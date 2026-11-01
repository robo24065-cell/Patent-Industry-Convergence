# C19-T3 s3: 부록 출처 칸의 원천 파일을 절대 경로로 풀고, 생산 스크립트 후보와 상류(2단계) 메타 읽기 여부를 붙인다 → cache/s3_sources.csv
import os, sys, re, csv, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
A = json.load(open(C + r'\s1_app.json', encoding='utf-8'))['APP']
IDX = json.load(open(C + r'\s2_index.json', encoding='utf-8'))
SC = {r['script']: r for r in csv.DictReader(open(C + r'\s2_scripts.csv', encoding='utf-8-sig'))}
SC = {k: v for k, v in SC.items() if not k.startswith(r'검증루프_20261012\C19')}
def resolve(tok):
    t = tok.strip().rstrip(',').strip()
    t = re.sub(r'\(([^)]*)\)$', '', t) if not t.startswith('(') else t
    t = t.replace('(MC)', MC).replace('(분석)', AN).replace('(VL)', VL)
    if t.startswith('분석\\'): t = B + '\\' + t
    elif t.startswith('보고서_20261012'): t = AN + '\\' + t
    elif t.startswith('검증루프'): t = AN + '\\' + t
    elif t.startswith('수집파이프라인'): t = B + '\\' + t
    t = re.sub(r'\((?:range_168|regdate|원문.*|rgst_no.*)\)?$', '', t).strip()
    return t
rows = []
seen = {}
for k, v in A.items():
    parts = [p for p in re.split(r';\s*|,\s+(?=\(|[A-Za-z_가-힣0-9]+\.(?:csv|json|md|txt|parquet))', v['src']) if p.strip()]
    prev_dir = None
    for p in parts:
        ap = resolve(p)
        if not os.path.isabs(ap) and prev_dir:   # 'variance_decomp.csv'처럼 앞 경로를 잇는 짧은 이름
            ap = os.path.join(prev_dir, ap)
        if os.path.isabs(ap): prev_dir = os.path.dirname(ap)
        ex = os.path.exists(ap)
        base = os.path.basename(ap)
        prods = IDX['produces'].get(base, [])
        # 같은 상위 폴더(트랙/보고서) 안 생산자를 우선
        apr = os.path.relpath(ap, AN) if ap.startswith(AN) else ap
        top = os.sep.join(apr.split(os.sep)[:3])
        near = [s for s in prods if s.startswith(os.sep.join(apr.split(os.sep)[:2]))]
        cand = near or prods
        info = []
        for s in cand[:8]:
            r = SC.get(s)
            if not r: continue
            ups = []
            for inp in (r['inputs'] or '').split(';'):
                if not inp: continue
                for s2 in IDX['produces'].get(inp, [])[:6]:
                    r2 = SC.get(s2)
                    if r2 and r2['reads_meta'] == '1': ups.append(f'{inp}<-{s2}')
            info.append(dict(s=s, meta=r['reads_meta'], kr=r['kr_only_hint'], fo=r['foreign_hint'], up=ups[:6]))
        rows.append(dict(app_id=k, app_line=v['line'], src_raw=p.strip(), src_abs=ap, exists=int(ex), n_prod=len(prods), n_near=len(near),
                         producers=json.dumps(info, ensure_ascii=False)))
with open(C + r'\s3_sources.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
u = {}
for r in rows: u.setdefault(r['src_abs'], r)
out('rows', len(rows), 'unique', len(u), 'missing', [os.path.relpath(k, AN) if k.startswith(AN) else k for k, r in u.items() if not r['exists']])
out('no producer', len([1 for r in u.values() if r['n_prod'] == 0]))
