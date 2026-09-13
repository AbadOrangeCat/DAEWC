"""Training receives only authorized rows. Test data are evaluated separately."""
from __future__ import annotations
import copy
import os
import platform
import random
import resource
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .protocol import metrics, source_threshold


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Floating-point kernels may differ across device families; hardware is logged.


def encode(tokenizer, rows, max_length):
    encoded = tokenizer([r["text"] for r in rows], truncation=True, padding="max_length",
                        max_length=max_length, return_tensors="pt")
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor([r["label"] for r in rows]), "ids": [r["id"] for r in rows]}


def batch(encoded, indices, device):
    return {k: v[indices].to(device) for k, v in encoded.items() if k != "ids"}


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def predict(model, encoded, domain, device, batch_size=128):
    model.eval()
    probabilities = []
    for start in range(0, len(encoded["labels"]), batch_size):
        b = batch(encoded, slice(start, start + batch_size), device)
        probabilities.extend(model(b["input_ids"], b["attention_mask"], domain).softmax(-1)[:, 1].cpu().tolist())
    return probabilities


def source_train(model, train, dev, cfg, device):
    model.configure_training("source", "source")
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=cfg["source_lr"], weight_decay=cfg["weight_decay"], eps=1e-8)
    best, best_loss, history, stale = None, float("inf"), [], 0
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(cfg["source_epochs"]):
        model.train()
        order = torch.randperm(len(train["labels"]))
        losses = []
        for indices in order.split(cfg["batch_size"]):
            b = batch(train, indices, device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(b["input_ids"], b["attention_mask"], "source"), b["labels"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            opt.step()
            losses.append(float(loss.detach().cpu()))
        probabilities = predict(model, dev, "source", device, cfg["eval_batch_size"])
        y, p = dev["labels"].numpy(), np.clip(probabilities, 1e-7, 1 - 1e-7)
        dev_loss = float(-(y * np.log(p) + (1-y) * np.log(1-p)).mean())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "dev_loss": dev_loss})
        print(f"Source epoch {epoch + 1}: dev loss {dev_loss:.5f}", flush=True)
        if dev_loss < best_loss - 1e-6:
            best_loss, stale = dev_loss, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= cfg["source_patience"]:
            break
    model.load_state_dict(best)
    probabilities = predict(model, dev, "source", device, cfg["eval_batch_size"])
    threshold = source_threshold(dev["labels"].numpy(), probabilities)
    synchronize(device)
    return threshold, {"history": history, "seconds": time.perf_counter() - started,
                       "selection": "minimum source development cross-entropy",
                       "threshold_selection": "source development macro-F1, fixed grid .05:.01:.95"}


def empirical_fisher(model, train, names, sample_size, seed, device):
    """Mean squared gradients of per-example observed-label log likelihood.

    No softened labels, batch-gradient squaring, or global importance rescaling.
    Only source training examples enter the estimate.
    """
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(train["labels"]), min(sample_size, len(train["labels"])), replace=False)
    original_flags = {n: p.requires_grad for n, p in model.named_parameters()}
    selected_params = []
    for n, p in model.named_parameters():
        p.requires_grad_(n in names)
        if n in names:
            selected_params.append((n, p))
    model.eval()
    star = {n: p.detach().cpu().clone() for n, p in selected_params}
    fisher = {n: torch.zeros_like(p, device="cpu") for n, p in selected_params}
    synchronize(device)
    started = time.perf_counter()
    for i in selected:
        b = batch(train, slice(int(i), int(i)+1), device)
        loss = F.cross_entropy(model(b["input_ids"], b["attention_mask"], "source"), b["labels"])
        grads = torch.autograd.grad(loss, [p for _, p in selected_params])
        for (n, _), grad in zip(selected_params, grads):
            fisher[n].add_(grad.detach().cpu().square() / len(selected))
    for n, p in model.named_parameters():
        p.requires_grad_(original_flags[n])
    synchronize(device)
    return {"star": star, "fisher": fisher, "sample_ids": [train["ids"][i] for i in selected],
            "seconds": time.perf_counter() - started, "estimator": "observed_label_per_example"}


def penalty(model, fisher, names, lam, alpha, damping):
    value = next(model.parameters()).new_zeros(())
    for name, p in model.named_parameters():
        if name in names:
            difference = (p - fisher["star"][name]).square()
            value = value + .5 * (lam * (fisher["fisher"][name] + damping) * difference + alpha * difference).sum()
    return value


def adapt(model, target_train, domain, method, cfg, fisher, device):
    """Fixed final checkpoint; no target dev, test or source example access."""
    model.to(device)
    model.configure_training(domain, method)
    teacher = copy.deepcopy(model).eval() if method == "lwf" else None
    if teacher is not None:
        for p in teacher.parameters():
            p.requires_grad_(False)
    cal = model.calibration_names()
    protected = model.shared_names() if method == "full_ewc" else cal
    lam = cfg["ewc_lambda"] if method in {"daewc", "no_gate", "no_proximity", "full_ewc"} else 0
    alpha = cfg["proximity_alpha"] if method in {"daewc", "no_gate", "no_ewc"} else 0
    importance = {key: {n: t.to(device) for n, t in fisher[key].items() if n in protected}
                  for key in ["star", "fisher"]}
    groups = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder.") and ".routes." not in name:
            lr = cfg["full_lr"] if method in {"full", "full_ewc", "lwf", "scratch"} else cfg["calibration_lr"]
        else:
            lr = cfg["module_lr"]
        # Shared calibration has no implicit weight decay toward zero.
        groups.append({"params": [p], "lr": lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8)
    n = len(target_train["labels"])
    order, cursor, steps, examples_seen, losses, reg_losses = torch.randperm(n), 0, 0, 0, [], []
    model.train()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    started = time.perf_counter()
    for step in range(cfg["adapt_steps"]):
        if cursor >= n:
            order, cursor = torch.randperm(n), 0
        indices = order[cursor:cursor + cfg["batch_size"]]
        cursor += len(indices)
        examples_seen += len(indices)
        b = batch(target_train, indices, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(b["input_ids"], b["attention_mask"], domain)
        loss = F.cross_entropy(logits, b["labels"])
        regularizer = penalty(model, importance, protected, lam, alpha, cfg["fisher_damping"]) if lam or alpha else loss.new_zeros(())
        if teacher is not None:
            temperature = cfg["lwf_temperature"]
            with torch.no_grad():
                teacher_prob = (teacher(b["input_ids"], b["attention_mask"], "source") / temperature).softmax(-1)
            # Source-domain route; target adapters never enter the distillation path.
            student_log_prob = (model(b["input_ids"], b["attention_mask"], "source") / temperature).log_softmax(-1)
            regularizer = regularizer + cfg["lwf_weight"] * temperature**2 * F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
        (loss + regularizer).backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["gradient_clip"])
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        reg_losses.append(float(regularizer.detach().cpu()))
        steps += 1
    synchronize(device)
    elapsed = time.perf_counter() - started
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    peak = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else None
    return {"optimizer_steps": steps, "seconds": elapsed, "seconds_per_step": elapsed/steps,
            "completed_passes_equivalent": examples_seen / n, "examples_seen_with_repetition": examples_seen,
            "loss_history": losses, "regularization_history": reg_losses,
            "trainable_names": trainable, "trainable_parameters": sum(trainable.values()),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "ewc_lambda_actual": lam, "proximity_alpha_actual": alpha,
            "peak_cuda_allocated_bytes": peak,
            "process_lifetime_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if platform.system() == "Darwin" else 1024),
            "memory_scope": "CUDA allocation peak per adaptation if CUDA; process lifetime RSS otherwise; no MPS peak claim",
            "source_replay": False, "unlabeled_target": False, "target_dev_labels_used": 0,
            "candidate_count": 1, "checkpoint_selection": "fixed final optimizer step"}
