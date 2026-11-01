# s7: 데이터 편향 점검(채점 뒤). 잠금 위험집합·양성 분포 vs 개발, 탐지 vs 미탐지(첫 공개 국내 법인 상호), NPS form 점검.
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libC import *
rows = []
def add(check, group, metric, dev, lock, note=''):
    rows.append(dict(check=check, group=group, metric=metric, dev=dev, lock=lock, note=note))
def tvd(a, b):
    ix = a.index.union(b.index)
    return round(float(0.5 * (a.reindex(ix, fill_value=0) - b.reindex(ix, fill_value=0)).abs().sum()), 4)
def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return round(float((a.mean() - b.mean()) / sp), 3) if sp > 0 else 0.0
u = pd.read_parquet(os.path.join(C, 'units.parquet'), columns=['key', 's0', 'y', 'E', 'ind2', 'sido'])
lab = pd.read_parquet(os.path.join(C, 'lock_labels.parquet'))
L = u[u.s0 == LOCK].drop(columns='y').merge(lab, on='key', validate='one_to_one')
D = u[u.s0.isin(DEV)]
for d in (L, D):
    d['size'] = pd.cut(d.E, [4, 9, 29, 99, 1e9], labels=['5-9', '10-29', '30-99', '100+']).astype(str)
for var in ['ind2', 'size', 'sido']:
    dr, lr = D[var].value_counts(normalize=True), L[var].value_counts(normalize=True)
    dp, lp = D[D.y == 1][var].value_counts(normalize=True), L[L.y == 1][var].value_counts(normalize=True)
    t1, t2 = tvd(dr, lr), tvd(dp, lp)
    add('분포', '위험집합', f'TVD {var} (개발 원점 합 vs 잠금)', None, t1, '경고' if t1 >= 0.10 else '')
    add('분포', '양성', f'TVD {var} (개발 양성 vs 잠금 양성)', None, t2, '경고' if t2 >= 0.10 else '')
    add('분포', '양성 vs 위험집합', f'TVD {var} 양성-위험집합', tvd(dp, dr), tvd(lp, lr))
    if var == 'size':
        for k in ['5-9', '10-29', '30-99', '100+']:
            add('분포', '양성률', f'size {k} 양성률', round(float(D[D['size'] == k].y.mean()), 6), round(float(L[L['size'] == k].y.mean()), 6))
add('규모', '위험집합', '양성률', round(float(D.y.mean()), 6), round(float(L.y.mean()), 6), f'개발 원점별 {D.groupby("s0").y.mean().round(6).min()}–{D.groupby("s0").y.mean().round(6).max()}')
add('규모', '위험집합', 'E0 중앙값 위험집합/양성', f'{D.E.median()} / {D[D.y == 1].E.median()}', f'{L.E.median()} / {L[L.y == 1].E.median()}')
# 탐지 vs 미탐지: 창 안 첫 공개 국내 법인 표지 상호(CA 존재)
ca = load_ca(); nm2b6, _ = name_map(ca); ca_names = set(ca.nname)
docs = load_docs(); docs['b6'] = docs.nname.map(nm2b6)
fva = docs[docs.vis.notna()].groupby('nname').vis.min()
pos_dev = pd.read_parquet(os.path.join(C, 'dev_pos.parquet'))
def grp(s0, posset):
    ref = REF[s0]; end = add_months(ref, 6)
    w = docs[docs.pub.notna() & (docs.pub > ref) & (docs.pub <= end) & docs.kor_corp]
    w = w[w.nname.map(fva) > ref].copy(); w['key'] = w.b6 + '|' + w.nname
    g = w.groupby('nname').agg(key=('key', 'first'), n=('appl_no', 'size'), dom=('dom_prio', 'mean'), km=('ksic', lambda s: s.value_counts().index[0])).reset_index()
    g = g[g.nname.isin(ca_names)]
    g['det'] = g.key.isin(posset)
    return g
GD = pd.concat([grp(s0, set(pos_dev[pos_dev.s0 == s0].key)) for s0 in DEV]); GL = grp(LOCK, set(L[L.y == 1].key))
for nm, f in [('SMD log1p(창 문서 수) 탐지-미탐지', lambda g: smd(np.log1p(g[g.det].n), np.log1p(g[~g.det].n))),
              ('SMD 국내우선 비율 탐지-미탐지', lambda g: smd(g[g.det].dom, g[~g.det].dom)),
              ('KSIC 최빈 TVD 탐지-미탐지', lambda g: tvd(g[g.det].km.value_counts(normalize=True), g[~g.det].km.value_counts(normalize=True))),
              ('탐지율(CA 존재 첫 공개 상호 중)', lambda g: round(float(g.det.mean()), 4)),
              ('상호 수', lambda g: int(len(g)))]:
    add('연결 편향', '첫 공개 국내법인 상호', nm, f(GD), f(GL))
# NPS form 점검(2024-12만; 2025-12는 열지 않음)
lg = json.load(open(os.path.join(C, 's1_log.json'), encoding='utf-8'))
fc = {r['s0']: r['form_counts_raw'] for r in lg['records']}
add('NPS 범위', 'S0 파일 form', 'form 값 분포(원 파일 전체 행)', json.dumps({k: v for k, v in fc.items() if k != LOCK}), json.dumps(fc[LOCK]),
    '전부 1이면 form==1 필터는 2024-12 이전 위험집합에 영향 없음. 2025-12 form=2 56,948행은 S0=2024-12 위험집합·피처에 쓰지 않음(C04 verify_3 수치)')
df = pd.DataFrame(rows)
df.to_csv(os.path.join(W, 'bias_lock.csv'), index=False, encoding='utf-8-sig')
print(df.to_string())
