# C17-T3 verify_0: 30% 독립 재판정 표본 추출(seed 20261017) + 근거 파일 줄 직접 출력(코더 A·B 출력 없이 원 파일을 다시 연다)
import os, sys, re, random, json
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
MC = os.path.join(ROOT, "모델경쟁_20260927")
W = os.path.join(ROOT, "검증루프_20261012", "C17", "C17-T3")
pm = pd.read_csv(os.path.join(W, 'pit_map_v2.csv'), encoding='utf-8-sig')
ids = pm.요소ID.tolist()
random.seed(20261017); samp = sorted(random.sample(ids, 27))
pd.DataFrame(dict(요소ID=samp)).to_csv(os.path.join(W, 'verify_0', 'v0_sample.csv'), index=False, encoding='utf-8-sig')
print('SAMPLE', samp)
def dump(spec, trackdir):
    # spec 예: "(MC)\T1_산업대기실\lib.py:59-63; R3\final_run_r3.py:75(...)"
    out = []
    cur = trackdir
    for part in re.split(r';\s*', str(spec)):
        m = re.match(r'(?:\(MC\)\\)?([^:(]+?\.(?:py|md|json)):([\d,\-·]+)', part.strip())
        if not m: continue
        rel = m.group(1)
        if rel.startswith(('T1_', 'T3_', 'T4_', 'N4_', 'N5_', 'N6_', 'N7_', 'N8_', 'N9_', '_shared')):
            path = os.path.join(MC, rel); cur = os.path.join(MC, rel.split('\\')[0])
        else:
            path = os.path.join(cur, rel)
        if not os.path.exists(path): out.append(f'  [없음] {path}'); continue
        L = open(path, encoding='utf-8', errors='replace').read().splitlines()
        for rg in re.split(r'[,·]', m.group(2)):
            if not rg: continue
            a, b = (rg.split('-') + [rg])[:2] if '-' in rg else (rg, rg)
            a, b = int(a), int(b)
            for i in range(max(1, a), min(len(L), b) + 1): out.append(f'  {rel}:{i}: {L[i-1][:220]}')
    return out
for k in samp:
    r = pm[pm.요소ID == k].iloc[0]
    trackdir = None
    print(f'\n===== {k} | {r.요소[:150]}')
    for line in dump(r['근거(파일:행)'], trackdir)[:24]: print(line)
