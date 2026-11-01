# C19-T3 s2: 스크립트 전수 색인 + 원천 파일 계보(자동 1차). 읽기 전용.
# 범위: 모델경쟁_20260927 전체 .py, 검증루프_20261012 C01–C18 .py (C19 제외)
# 출력: cache/s2_scripts.csv (스크립트별 메타 직접 읽기·US/EP 필터 단서), cache/s2_lineage.json (원천 파일 → 생산 스크립트 → 상류 메타 읽기 스크립트)
import os, sys, re, csv, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
roots = [MC] + sorted(glob.glob(VL + r'\C[0-1][0-9]'))
files = []
for r in roots:
    for dp, dn, fn in os.walk(r):
        for f in fn:
            if f.endswith('.py'):
                files.append(os.path.join(dp, f))
out('scripts', len(files))
META_PAT = re.compile(r"_shared[\\/]+meta|[\\/]meta[\\/]\*\.parquet|'\.\./meta/|공모전데이터|융합데이터_|C18_META|meta_V1E|meta_V1\.parquet|\{META\}|META\b")
DATA = re.compile(r"[A-Za-z0-9_\-\.가-힣]+\.(?:parquet|csv|json|npy|npz|pkl|pickle|feather|sqlite|joblib|txt)\b")
WR = re.compile(r"to_parquet|to_csv|json\.dump|\.dump\(|np\.save|savez|\.write\(|COPY\s*\(|copy\s*\(|to_json|to_feather|savefig|joblib\.dump|pickle\.dump|\.save\(|,\s*['\"]w['\"]|mode=['\"]w|open\([^)]*['\"]w")
CC_KR_ONLY = re.compile(r"cc\s*=\s*'KR'|cc\s*==\s*['\"]KR['\"]|cc\s*in\s*\(\s*'KR'\s*\)|\.cc\s*==\s*['\"]KR['\"]")
CC_FOREIGN = re.compile(r"'US'|\"US\"|'EP'|\"EP\"|cc\s*<>\s*'KR'|cc\s*!=\s*['\"]KR['\"]|cc\s*in\s*\(\s*'US'|GROUP BY cc|groupby\(\[?['\"]cc|by cc|cc\s*IN\s*\(")
rows = []; produces = {}; mentions = {}
for p in files:
    try:
        t = open(p, encoding='utf-8', errors='replace').read()
    except Exception as e:
        continue
    rel = os.path.relpath(p, AN)
    lines = t.split('\n')
    meta_lines = [i + 1 for i, s in enumerate(lines) if META_PAT.search(s) and not s.strip().startswith('#')]
    w_out = set(); r_in = set()
    for s in lines:
        if s.strip().startswith('#'): continue
        ds = set(DATA.findall(s))
        if not ds: continue
        (w_out if WR.search(s) else r_in).update(ds)
    # 변수에 경로를 담아 나중에 쓰는 경우를 위해, 쓰기 줄이 변수만 쓰면 놓친다 → 'mentions'로 보조
    for d in w_out: produces.setdefault(d, set()).add(rel)
    for d in w_out | r_in: mentions.setdefault(d, set()).add(rel)
    rows.append(dict(script=rel, n_lines=len(lines), reads_meta=int(bool(meta_lines)), meta_lines=';'.join(map(str, meta_lines[:12])),
                     kr_only_hint=int(bool(CC_KR_ONLY.search(t))), foreign_hint=int(bool(CC_FOREIGN.search(t))),
                     family_hint=int('family' in t), outputs=';'.join(sorted(w_out))[:600], inputs=';'.join(sorted(r_in))[:600]))
with open(C + r'\s2_scripts.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
S = {r['script']: r for r in rows}
json.dump(dict(produces={k: sorted(v) for k, v in produces.items()}, mentions={k: sorted(v) for k, v in mentions.items()}),
          open(C + r'\s2_index.json', 'w', encoding='utf-8'), ensure_ascii=False)
from collections import Counter
out('reads_meta', sum(r['reads_meta'] for r in rows), Counter(r['script'].split(os.sep)[1] if r['script'].startswith('모델경쟁') else r['script'].split(os.sep)[1] for r in rows if r['reads_meta']))
