"""Build a reviewed-source ZIP using Git ignore rules, with clean notebooks.

Does not initialize or modify the source project's Git repository or push anything.
This is a basic screen, not a guarantee of secret detection or redistribution rights.
"""
import argparse
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}'),
            re.compile(r'github_pat_[A-Za-z0-9_]{20,}'),
            re.compile(r'AKIA[A-Z0-9]{16}'),
            re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
TEXT_EXTENSIONS = {'.py', '.md', '.txt', '.json', '.toml', '.yaml', '.yml', '.csv', '.ipynb'}


def public_paths(root):
    git = shutil.which('git')
    if not git:
        raise RuntimeError('Git is required to evaluate .gitignore rules.')
    paths = [p for p in sorted(root.rglob('*')) if p.is_file() and not p.is_symlink()
             and '.git' not in p.relative_to(root).parts]
    names = [p.relative_to(root).as_posix() for p in paths]
    with tempfile.TemporaryDirectory(prefix='battery-gitignore-') as tmp:
        subprocess.run([git, 'init', '-q', tmp], check=True)
        result = subprocess.run([git, '-c', 'core.excludesFile=/dev/null',
                                 f'--git-dir={Path(tmp)/".git"}', f'--work-tree={root}',
                                 'check-ignore', '--no-index', '-z', '--stdin'],
                                input='\0'.join(names).encode()+b'\0', capture_output=True,
                                cwd=root)
        if result.returncode not in [0, 1]:
            raise RuntimeError('Git ignore evaluation failed.')
        ignored = set(result.stdout.decode().split('\0'))
    return [p for p, name in zip(paths, names) if name not in ignored]


def clean_notebook(data):
    notebook = json.loads(data.decode('utf-8'))
    notebook['metadata'] = {k: v for k, v in notebook.get('metadata', {}).items()
                            if k in ['kernelspec', 'language_info']}
    for cell in notebook.get('cells', []):
        cell['metadata'] = {}
        cell.pop('attachments', None)
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    return (json.dumps(notebook, ensure_ascii=False, indent=1)+'\n').encode('utf-8')


def run(output, root=ROOT):
    root = Path(root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new output ZIP; existing packages are preserved.')
    payloads, warnings, blockers = [], [], []
    for path in public_paths(root):
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        if path.suffix == '.ipynb':
            data = clean_notebook(data)
        if len(data) > 100*1024*1024:
            blockers.append(f'{relative}: exceeds 100 MiB')
        elif len(data) > 50*1024*1024:
            warnings.append(f'{relative}: exceeds 50 MiB')
        if path.suffix in TEXT_EXTENSIONS or path.name.startswith('.env'):
            content = data.decode('utf-8', errors='replace')
            for number, line in enumerate(content.splitlines(), 1):
                if any(pattern.search(line) for pattern in PATTERNS):
                    blockers.append(f'{relative}:{number}: possible credential; value suppressed')
                if ('/'+'Users/') in line or ('/'+'var/folders/') in line or re.search(r'C:\\\\Users\\\\', line):
                    warnings.append(f'{relative}:{number}: possible local path; review manually')
        payloads.append((relative, data))
    if blockers:
        raise RuntimeError('Public bundle blocked:\n'+'\n'.join(blockers))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for relative, data in payloads:
            archive.writestr('ess-battery-project/'+relative, data)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert 'ess-battery-project/.gitignore' in names
        assert 'ess-battery-project/.env.example' in names
        assert 'ess-battery-project/.env' not in names
        assert not any(name.endswith(('.joblib','.parquet','.mat','.sqlite3','.log','.pdf')) for name in names)
    print(f'Created: {output}; files={len(payloads)}; basic scan passed')
    for warning in warnings:
        print('REVIEW:', warning)
    print('Manual copyright/data/identity review is still required. Nothing was uploaded.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='dist/ess_battery_github.zip')
    args = parser.parse_args()
    run(args.out)
