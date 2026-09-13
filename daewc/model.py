"""One DAEWC definition, with per-block residual adapters and feature gates."""
from __future__ import annotations
import copy
import torch
from torch import nn
from transformers import BertConfig, BertModel, BertTokenizer


class LowRank(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.normal_(self.down.weight, std=.02)
        nn.init.zeros_(self.up.weight)
        self.scale = 2.0  # LoRA alpha = 2 * rank.

    def forward(self, x):
        return self.up(self.down(x)) * self.scale


class RoutedLinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.routes = nn.ModuleDict()
        self.active = None

    def forward(self, x):
        value = self.base(x)
        if self.active in self.routes:
            value = value + self.routes[self.active](x)
        return value


class ResidualAdapter(nn.Module):
    def __init__(self, width, rank, dropout):
        super().__init__()
        self.down = nn.Linear(width, rank)
        self.up = nn.Linear(rank, width)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.dropout(self.up(torch.nn.functional.gelu(self.down(x))))


class DomainPath(nn.Module):
    def __init__(self, width, layers, rank, embedding_dim, dropout, gate=True):
        super().__init__()
        self.adapters = nn.ModuleList([ResidualAdapter(width, rank, dropout) for _ in range(layers)])
        self.use_gate = gate
        if gate:
            self.embedding = nn.Parameter(torch.randn(embedding_dim) * .02)
            self.gates = nn.ModuleList([nn.Linear(embedding_dim, width) for _ in range(layers)])
            for layer in self.gates:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, layer):
        x = self.adapters[layer](x)
        if self.use_gate:
            x = x * (2 * torch.sigmoid(self.gates[layer](self.embedding)))
        return x


class Classifier(nn.Module):
    def __init__(self, model_path=None, config=None, pretrained=True):
        super().__init__()
        if config is None:
            config = BertConfig.from_pretrained(model_path)
        config._attn_implementation = "eager"
        self.encoder = (BertModel.from_pretrained(model_path, config=config, add_pooling_layer=False)
                        if pretrained else BertModel(config, add_pooling_layer=False))
        self.width = config.hidden_size
        self.layers = config.num_hidden_layers
        for layer in self.encoder.encoder.layer:
            layer.attention.self.query = RoutedLinear(layer.attention.self.query)
            layer.attention.self.value = RoutedLinear(layer.attention.self.value)
        self.paths = nn.ModuleDict()
        self.heads = nn.ModuleDict({"source": nn.Linear(self.width, 2)})
        self.method_by_domain = {"source": "source"}
        self.head_dropout = nn.Dropout(.1)

    def add_domain(self, domain, method, cfg):
        if domain in self.heads:
            raise ValueError(f"Domain already exists: {domain}")
        self.heads[domain] = copy.deepcopy(self.heads["source"])
        self.method_by_domain[domain] = method
        if method in {"daewc", "adapter", "no_gate", "no_ewc", "no_proximity", "no_regularizers"}:
            self.paths[domain] = DomainPath(self.width, self.layers, cfg["adapter_rank"],
                                            cfg["domain_embedding_dim"], cfg["adapter_dropout"],
                                            gate=method not in {"no_gate", "adapter"})
        if method == "lora":
            for layer in self.encoder.encoder.layer:
                for name in ["query", "value"]:
                    getattr(layer.attention.self, name).routes[domain] = LowRank(self.width, cfg["lora_rank"])

    def forward(self, input_ids, attention_mask, domain="source"):
        for layer in self.encoder.encoder.layer:
            layer.attention.self.query.active = domain
            layer.attention.self.value.active = domain
        h = self.encoder.embeddings(input_ids=input_ids)
        mask = (1 - attention_mask[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
        for i, layer in enumerate(self.encoder.encoder.layer):
            h = layer(h, attention_mask=mask)
            if isinstance(h, tuple):  # Compatibility with transformers 4.x.
                h = h[0]
            if domain in self.paths:
                h = self.paths[domain](h, i)
        # Masked mean pooling; padding never contributes to the representation.
        m = attention_mask.to(h.dtype).unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1)
        return self.heads[domain](self.head_dropout(pooled))

    def calibration_names(self):
        return {name for name, _ in self.named_parameters()
                if name.startswith("encoder.encoder.layer.") and ".routes." not in name
                and ("LayerNorm." in name or name.endswith("bias"))}

    def shared_names(self):
        return {name for name, _ in self.named_parameters()
                if name.startswith("encoder.") and ".routes." not in name}

    def configure_training(self, domain, method):
        calibration = self.calibration_names()
        for name, parameter in self.named_parameters():
            target_specific = (name.startswith(f"heads.{domain}.") or
                               name.startswith(f"paths.{domain}.") or f".routes.{domain}." in name)
            full = method in {"full", "full_ewc", "lwf", "scratch"}
            shared = name in self.shared_names() and (full or method == "source")
            calibrate = name in calibration and method in {"daewc", "no_gate", "no_ewc", "no_proximity", "no_regularizers"}
            parameter.requires_grad_(target_specific or shared or calibrate)


def tokenizer_from(path):
    return BertTokenizer.from_pretrained(path, do_lower_case=True)
