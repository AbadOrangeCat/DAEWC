"""Create a flat Overleaf upload set without changing the scientific content."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def build(manuscript, output):
    manuscript = manuscript.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    assets = {}

    def expand(relative, stack=()):
        path = (manuscript / relative).resolve()
        if not path.is_relative_to(manuscript) or path in stack:
            raise ValueError(f"Invalid or recursive input: {relative}")
        content = path.read_text()
        sources[str(path.relative_to(manuscript))] = hashlib.sha256(path.read_bytes()).hexdigest()
        def include(match):
            name = match.group(1)
            if not name.endswith('.tex'):
                name += '.tex'
            return '\n% Source: ' + name + '\n' + expand(name, stack + (path,))
        def figure(match):
            name = match.group(1)
            source = (manuscript / name).resolve()
            if not source.is_relative_to(manuscript):
                raise ValueError(f"Invalid figure path: {name}")
            target = 'revision_' + source.name
            if target in assets and assets[target] != source:
                raise ValueError(f"Conflicting figure names: {target}")
            assets[target] = source
            return match.group(0).replace(name, target)
        content = re.sub(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', figure, content)
        return re.sub(r'\\input\{([^}]+)\}', include, content)

    text = expand('manuscript.tex')
    (output / 'DAEWC.tex').write_text(text)
    for name, source in assets.items():
        shutil.copy2(source, output / name)
    for name in ['references.bib', 'highlights.txt']:
        shutil.copy2(manuscript / name, output / name)
    generated = ['DAEWC.tex', 'references.bib', 'highlights.txt', *assets]
    # Carry the original journal template and its front-matter icons with the paper.
    template_files = [manuscript / name for name in ('cas-sc.cls', 'cas-common.sty')]
    template_files += sorted((manuscript / 'thumbnails').glob('*.jpeg'))
    for source in template_files:
        if source.is_file():
            name = str(source.relative_to(manuscript))
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            generated.append(name)
    manifest = {'main_file': 'DAEWC.tex', 'compiler': 'XeLaTeX',
                'source_sha256': sources,
                'files': {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                          for name in sorted(generated)}}
    (output / 'SYNC_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manuscript', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.manuscript, args.out)
    print(f"Prepared {len(result['files'])} files; select DAEWC.tex and XeLaTeX in Overleaf.")
