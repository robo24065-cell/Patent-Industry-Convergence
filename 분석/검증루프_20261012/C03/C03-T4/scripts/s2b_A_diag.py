# s2b_A: 사후 진단(판정 밖) — KR 문서 vis 조건 제거, 창 12개월로 확대 시 양성 수
import sys
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C03\C03-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from libA import *
docs, L = load(('PUB',))
out = []
for t0 in DEV:
    T = f'{t0}1231'
    dd = docs.rename(columns={'ksic': 'k'})
    vis_T = dd[dd.vis <= T]; cnt = vis_T.groupby('cd').appl_no.nunique(); F = set(cnt[cnt >= 3].index)
    dom = vis_T[vis_T.cd.isin(F)][['cd', 'k']].drop_duplicates()
    j = L.merge(dd[['cd', 'appl_no', 'k', 'vis']], left_on='kr_appl', right_on='appl_no'); j = j[j.cd.isin(F)]
    pre = j[(j.pub_date <= T) & (j.vis <= T)][['cd', 'k']].drop_duplicates()
    risk = dom.merge(pre, on=['cd', 'k'], how='left', indicator=True); risk = risk[risk._merge == 'left_only'][['cd', 'k']]
    r = {'t0': t0}
    for nm, we, visc in [('H1_novis', f'{t0+1}0630', False), ('Y12_vis', f'{t0+1}1231', True)]:
        w = j[(j.pub_date > T) & (j.pub_date <= we)]
        if visc: w = w[w.vis <= we]
        r[nm] = len(w[['cd', 'k']].drop_duplicates().merge(risk, on=['cd', 'k']))
    out.append(r); print(r, flush=True)
R = pd.DataFrame(out); print(R.sum())
R.to_csv(C + r'\s2b_A_posthoc_diag.csv', index=False, encoding='utf-8-sig')
