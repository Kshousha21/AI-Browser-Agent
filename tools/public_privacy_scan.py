"""Content-safe publication audit. Reports categories and paths, never matched values."""
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def audit():
    warnings = []
    def git(*args):
        result = subprocess.run(['git','-C',str(ROOT),*args],text=True,encoding='utf-8',capture_output=True,check=True)
        if result.stderr.strip(): warnings.append('Some directory entries could not be inventoried; audit is incomplete there.')
        return result.stdout.splitlines()
    tracked=sorted(set(git('ls-files') + git('ls-files','--others','--exclude-standard')))
    suspect=[]; private_paths=[]; internal_urls=[]; binaries=Counter()
    internal=re.compile(r"https?://(?:[^/\s]+\.)?(?:intranet|internal)[./]|https?://[^/\s]*\.(?:local|corp)(?=[:/\s])",re.I)
    secret=re.compile(r'\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16}|eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,})\b')
    path_re=re.compile(r'(?:[A-Z]:[\\/](?:Users|home)[\\/][^\s"\'<>]+|\\\\[A-Za-z0-9_.-]+\\[^\s"\'<>]+)',re.I)
    for name in tracked:
        file=ROOT/name
        if not file.is_file():continue
        if file.suffix.lower() in {'.pdf','.png','.jpg','.jpeg','.log','.jsonl','.sqlite','.sqlite3'}:
            binaries[file.suffix.lower()]+=1
        if file.name=='.env' or file.suffix.lower() in {'.key','.pem','.pfx','.p12'}:
            suspect.append({'path':name,'kind':'credential_file'})
            continue
        try:text=file.read_text(encoding='utf-8')
        except (UnicodeError,OSError):continue
        if path_re.search(text):private_paths.append(name)
        if internal.search(text):internal_urls.append(name)
        if secret.search(text):suspect.append({'path':name,'kind':'possible_secret_pattern'})
    ignored=git('ls-files','--others','--ignored','--exclude-standard')
    artifacts=Counter()
    for name in ignored:
        parts=Path(name).parts
        if any(p in {'.venv','node_modules','.npm-cache','.git'} for p in parts):continue
        if Path(name).name=='.env':artifacts['environment_files']+=1
        elif Path(name).suffix.lower()=='.pdf':artifacts['pdfs']+=1
        elif Path(name).suffix.lower() in {'.png','.jpg','.jpeg','.webp'}:artifacts['images_or_screenshots']+=1
        elif Path(name).suffix.lower() in {'.log','.jsonl'}:artifacts['logs']+=1
        elif Path(name).suffix.lower() in {'.sqlite','.sqlite3','.db'}:artifacts['databases']+=1
    return {'working_tree_files_scanned':sum((ROOT/name).is_file() for name in tracked),
        'possible_secret_files':suspect,'internal_url_reference_files':internal_urls,'inventory_warnings':sorted(set(warnings)),'workstation_path_reference_files':private_paths,
        'tracked_artifact_counts':dict(binaries),'ignored_private_artifact_counts':dict(artifacts),
        'limits':'Tracked and nonignored untracked working-tree text only. Ignored private artifacts inventoried without reading contents. Git history and external local stores not content-scanned. Pattern matches require human review; no secret values emitted.'}

if __name__=='__main__':
    print(json.dumps(audit(),indent=2))
