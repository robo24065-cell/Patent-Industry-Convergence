# -*- coding: utf-8 -*-
"""커밋 가드: 공모전 제공 데이터·파생 데이터·API 키가 저장소에 들어가지 않게 막는다.
사용: python .githooks/guard.py [--staged | 파일 ...]
- 금지 확장자(데이터·압축·DB), 5MB 초과 파일, 허용 폴더 밖 CSV/JSON 차단
- 저장소 밖 API키_기록.txt(있을 때만)에서 키 후보 문자열을 읽어 파일 내용에 있으면 차단(키 값은 출력하지 않음)"""
import os, re, subprocess, sys, urllib.parse, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ROOT = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True, encoding='utf-8').stdout.strip() or '.'
BAD_EXT = {'.parquet', '.sqlite', '.db', '.npy', '.npz', '.pkl', '.pickle', '.feather', '.h5', '.hdf5', '.zip', '.gz', '.7z', '.tar', '.xlsx', '.xls', '.tsv', '.sqlite-wal', '.sqlite-shm', '.arrow', '.dta', '.sav'}
CSV_OK = ('분석/보고서_20261012/figs/', '분석/보고서_20261012/facts/')
MAX = 5 * 1024 * 1024
def files():
    if len(sys.argv) > 1 and sys.argv[1] != '--staged':
        return sys.argv[1:]
    out = subprocess.run(['git', '-c', 'core.quotepath=false', 'diff', '--cached', '--name-only', '--diff-filter=ACMR'], capture_output=True, text=True, encoding='utf-8', cwd=ROOT).stdout
    return [l for l in out.splitlines() if l.strip()]
def keys():
    ks = set()
    for name in ('API키_기록.txt', 'API키_기록.txt.bak_20260929'):
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            for tok in re.findall(r'[A-Za-z0-9%+/=_\-]{16,}', open(p, encoding='utf-8', errors='ignore').read()):
                if tok.startswith(('http', 'www')) or tok.isdigit():
                    continue
                ks.add(tok); ks.add(urllib.parse.unquote(tok)); ks.add(urllib.parse.quote(urllib.parse.unquote(tok), safe=''))
    return {k for k in ks if len(k) >= 16}
GEN = re.compile(r'(serviceKey|crtfc_key|authKey|api[_-]?key|ServiceKey|accessKey|secret)\s*[=:]\s*["\'][A-Za-z0-9%+/=]{20,}["\']', re.I)
bad = []
K = keys()
for f in files():
    p = os.path.join(ROOT, f)
    if not os.path.isfile(p):
        continue
    ext = os.path.splitext(f)[1].lower()
    low = f.replace(chr(92), '/')
    if ext in BAD_EXT: bad.append((f, 'data/archive extension')); continue
    if os.path.getsize(p) > MAX: bad.append((f, 'file > 5MB')); continue
    if ext in ('.csv', '.json', '.jsonl') and not (low.startswith(CSV_OK) or low.endswith('LEDGER.jsonl')): bad.append((f, 'csv/json outside allowed folders')); continue
    if os.path.basename(f) in ('API키_기록.txt', 'collect_kosis_new.py', 'collect_registry.py') or 'API키' in f: bad.append((f, 'blocked by name')); continue
    if ext in ('.png', '.pdf', '.docx', '.doc'): continue
    try: txt = open(p, encoding='utf-8', errors='ignore').read()
    except Exception: continue
    if any(k in txt for k in K): bad.append((f, 'contains a key string from API키_기록.txt')); continue
    if GEN.search(txt): bad.append((f, 'looks like a hard-coded API key'))
if bad:
    print('GUARD BLOCKED', len(bad))
    for f, why in bad: print(' -', f, ':', why)
    sys.exit(1)
print('GUARD OK', len(K), 'key patterns checked')
