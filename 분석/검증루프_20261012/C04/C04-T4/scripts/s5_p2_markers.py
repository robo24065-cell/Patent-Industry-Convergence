# s5: P2 운영 표지 M(기준일 이후 활동)으로 사업자번호 경로의 사후 의존성 판정(PREREG 변경 기록 ①). 개발 자료만(관측 상한 2023-12-31).
import sys, json
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C04\C04-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libA import *
CAP = '20231231'
ca = load_ca(); nm2b6, nst = name_map(ca)
ca_names = set(ca.nname)
docs = load_docs(); docs['b6'] = docs.nname.map(nm2b6)
raw = pd.read_parquet(B + r'\공모전데이터\작업데이터\kr_patents.parquet', columns=['출원인명'])
docs['kor_corp'] = raw['출원인명'].fillna('').str.contains(r'주식회사|\(주\)|㈜|유한회사').values & docs.dom_prio.values
docs = docs[docs.kor_corp & docs.vis.notna()]
u = pd.read_parquet(C + r'\A_units_dev.parquet')
rows = []
def add(cand, metric, value, note=''):
    rows.append(dict(candidate=cand, check='P2c(M)', metric=metric, value=value, note=note)); print(cand, metric, value, note, flush=True)
def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return round(float((a.mean() - b.mean()) / sp), 3) if sp > 0 else 0.0
# ---- C04-A/C
allA = []
for s0 in DEV_A:
    ref = REF[s0]
    dv = docs[docs.vis <= ref]
    later = set(docs[(docs.vis > ref) & (docs.vis <= CAP)].nname)
    g = dv.groupby('nname').agg(n=('appl_no', 'size'))
    g = g[g.index.isin(ca_names) & (g.index.str.len() >= 2)]
    g['has_biz_ok'] = g.index.isin(set(nm2b6))
    g['M'] = g.index.isin(later).astype(int)
    linked_names = set(u[u.s0 == s0].nname)
    g['linked'] = g.index.isin(linked_names)
    g['s0'] = s0
    allA.append(g)
    add('A/C', f'{s0} 연결 vs 비연결(사업자번호 없는 CA 상호 포함) M 평균', f'{g[g.linked].M.mean():.3f} / {g[~g.linked].M.mean():.3f}',
        f'SMD {smd(g[g.linked].M, g[~g.linked].M)}; 사업자번호(name_ok) 보유율 M=1 {g[g.M==1].has_biz_ok.mean():.3f} / M=0 {g[g.M==0].has_biz_ok.mean():.3f}')
G = pd.concat(allA)
smdA = smd(G[G.linked].M, G[~G.linked].M)
add('A/C', 'POOLED SMD(M) 연결-비연결', smdA, f'연결 {G[G.linked].M.mean():.3f} / 비연결 {G[~G.linked].M.mean():.3f}; name_ok 보유율 M=1 {G[G.M==1].has_biz_ok.mean():.3f} / M=0 {G[G.M==0].has_biz_ok.mean():.3f}')
uu = u.merge(G.reset_index()[['nname', 's0', 'M']], on=['nname', 's0'], how='left')
for y in ['yA', 'yC']:
    r1, r0 = uu.loc[uu.M == 1, y].mean(), uu.loc[uu.M == 0, y].mean()
    add('A' if y == 'yA' else 'C', f'{y} 양성률 M=1/M=0(연결 단위)', round(float(r1 / r0), 3), f'{r1:.4f} / {r0:.4f} (n {int((uu.M==1).sum())} / {int((uu.M==0).sum())})')
# 원점 고정 추적 기간 민감도: M2 = 기준일 후 24개월 안(개발 S1과 같은 창) — 결과 창과 겹친다는 점 명시
allA2 = []
for s0 in DEV_A:
    ref = REF[s0]; e2 = add_months(ref, 24)
    later2 = set(docs[(docs.vis > ref) & (docs.vis <= e2)].nname)
    g = G[G.s0 == s0].copy(); g['M2'] = g.index.isin(later2).astype(int); allA2.append(g)
G2 = pd.concat(allA2)
add('A/C', 'POOLED SMD(M2: 24개월) 연결-비연결', smd(G2[G2.linked].M2, G2[~G2.linked].M2), f'name_ok 보유율 M2=1 {G2[G2.M2==1].has_biz_ok.mean():.3f} / M2=0 {G2[G2.M2==0].has_biz_ok.mean():.3f}')
# ---- C04-B
pos = pd.read_parquet(C + r'\B_dev_pos.parquet')
first_vis = docs.groupby('nname').vis.min()
allB = []
for s0 in DEV_B:
    ref = REF[s0]; end = add_months(ref, 6); fu = add_months(end, 18)
    if fu > CAP:
        continue
    w = docs[(docs.pub.notna()) & (docs.pub > ref) & (docs.pub <= end)]
    w = w[w.nname.map(first_vis) > ref]
    g = w.groupby('nname').size().rename('n').to_frame()
    g = g[g.index.isin(ca_names) & (g.index.str.len() >= 2)]
    later = set(docs[(docs.vis > end) & (docs.vis <= fu)].nname)
    g['M'] = g.index.isin(later).astype(int)
    det = set(pos[pos.s0 == s0].key.str.split('|').str[1])
    g['detected'] = g.index.isin(det)
    g['has_biz_ok'] = g.index.isin(set(nm2b6))
    g['s0'] = s0
    allB.append(g)
    add('B', f'{s0} 첫 공개 상호 탐지율 M=1 / M=0', f'{g[g.M==1].detected.mean():.3f} / {g[g.M==0].detected.mean():.3f}', f'n {int((g.M==1).sum())} / {int((g.M==0).sum())}; name_ok M=1 {g[g.M==1].has_biz_ok.mean():.3f} / M=0 {g[g.M==0].has_biz_ok.mean():.3f}')
GB = pd.concat(allB)
add('B', 'POOLED SMD(M) 탐지-미탐지', smd(GB[GB.detected].M, GB[~GB.detected].M), f'탐지 {GB[GB.detected].M.mean():.3f} / 미탐지 {GB[~GB.detected].M.mean():.3f}')
d1, d0 = GB[GB.M == 1].detected.mean(), GB[GB.M == 0].detected.mean()
add('B', 'POOLED 탐지율 비 M=1/M=0', round(float(d1 / d0), 3), f'{d1:.4f} / {d0:.4f}')
add('B', 'POOLED name_ok(사업자번호) 보유율 M=1/M=0', f"{GB[GB.M==1].has_biz_ok.mean():.3f} / {GB[GB.M==0].has_biz_ok.mean():.3f}")
pd.DataFrame(rows).to_csv(C + r'\s5_p2_markers.csv', index=False, encoding='utf-8-sig')
