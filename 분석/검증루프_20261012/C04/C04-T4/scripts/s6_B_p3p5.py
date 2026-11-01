# s6: C04-B P3(연결 편향)·P5-a(사건 (기업키,S0,KSIC) 자카드 vs 혼동 보존 귀무·UB). 개발 원점만.
import sys, json, time
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C04\C04-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libA import *
T0 = time.time(); NREP = 20
ca = load_ca(); nm2b6, nst = name_map(ca); ca_names = set(ca.nname)
docs = load_docs(); docs['b6'] = docs.nname.map(nm2b6)
raw = pd.read_parquet(B + r'\공모전데이터\작업데이터\kr_patents.parquet', columns=['출원인명'])
docs['kor_corp'] = raw['출원인명'].fillna('').str.contains(r'주식회사|\(주\)|㈜|유한회사').values & docs.dom_prio.values
pos = pd.read_parquet(C + r'\B_dev_pos.parquet')
first_vis = docs[docs.vis.notna()].groupby('nname').vis.min()
rows = []
def add(check, metric, value, note=''):
    rows.append(dict(candidate='B', check=check, metric=metric, value=value, note=note)); print(check, metric, value, note, flush=True)
def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return round(float((a.mean() - b.mean()) / sp), 3) if sp > 0 else 0.0
W = []; lr = {}
for s0 in DEV_B:
    ref = REF[s0]; end = add_months(ref, 6)
    assert end <= '20240630'
    w = docs[docs.pub.notna() & (docs.pub > ref) & (docs.pub <= end) & docs.kor_corp]
    w = w[w.nname.map(first_vis) > ref].copy()
    d0 = load_nps(s0, cols=('name', 'biz6', 'stat', 'adpt_dt', 'n_emp', 'amt', 'indcd', 'ldong'))
    allkeys = set(d0.key)
    w['key'] = w.b6 + '|' + w.nname
    nmset = w.groupby('nname').agg(key=('key', 'first'), n=('appl_no', 'size'), dom=('dom_prio', 'mean'),
                                   kmode=('ksic', lambda s: s.value_counts().index[0])).reset_index()
    nmset['in_nps'] = nmset.key.isin(allkeys)
    det = set(pos[pos.s0 == s0].key)
    nmset['detected'] = nmset.key.isin(det)
    nmset['in_ca'] = nmset.nname.isin(ca_names); nmset['s0'] = s0
    lr[s0] = round(float(nmset.in_nps.mean()), 4)
    W.append(nmset)
    w['s0'] = s0
G = pd.concat(W, ignore_index=True)
add('P3', '연결률(첫 공개 국내법인 상호 → S0 NPS 활성 기업키, E0 무관) 원점별', json.dumps(lr), f'평균 {np.mean(list(lr.values())):.4f}, 최고/최저 {max(lr.values())/min(lr.values()):.3f}')
add('P3', '탐지율(E0≥5·NTIS 제외 위험집합 양성으로 잡힘) 평균', round(float(G.detected.mean()), 4), f'CA 존재율 {G.in_ca.mean():.3f}')
Gc = G[G.in_ca]
add('P3', 'SMD 탐지-미탐지 log1p(창 문서 수)', smd(np.log1p(Gc[Gc.detected].n), np.log1p(Gc[~Gc.detected].n)))
add('P3', 'SMD 탐지-미탐지 국내우선 비율', smd(Gc[Gc.detected].dom, Gc[~Gc.detected].dom))
pl = Gc[Gc.detected].kmode.value_counts(normalize=True); pn = Gc[~Gc.detected].kmode.value_counts(normalize=True); ix = pl.index.union(pn.index)
add('P3', 'KSIC 최빈 TVD 탐지-미탐지(풀링)', round(float(0.5 * (pl.reindex(ix, fill_value=0) - pn.reindex(ix, fill_value=0)).abs().sum()), 3))
cp = Gc[Gc.detected].groupby(['s0', 'kmode']).size(); cu = Gc[~Gc.detected].groupby(['s0', 'kmode']).size()
add('P3', '양성 가중 KSIC×원점 칸 공존율', round(float(cp[cp.index.isin(cu.index)].sum() / cp.sum()), 4))
# ---- P5-a
lab = pd.read_parquet(B + r'\분석\검증루프_20261012\C02\C02-T4\cache\kr_publab_src.parquet')
lab['mg'] = lab.sg.str.split('/').str[0]; lab['sc'] = lab.sg.str[:4]; lab = lab.set_index('appl_no')
def hard(t0, start, nmin):
    sub = lab[(lab.appl_year <= t0)]
    lv = ['sg', 'mg', 'sc'][['sg', 'mg', 'sc'].index(start):]
    out = pd.Series(np.nan, index=lab.index, dtype=object); todo = pd.Series(True, index=lab.index)
    for l in lv:
        s = sub[sub[l].notna()]
        tab = s.groupby([l, 'k_free']).size().reset_index(name='n'); tot = tab.groupby(l).n.sum()
        mode = tab.sort_values([l, 'n', 'k_free'], ascending=[True, False, True]).drop_duplicates(l).set_index(l).k_free
        v = lab[l].map(mode[mode.index.isin(tot[tot >= nmin].index)])
        hit = todo & v.notna(); out[hit] = v[hit]; todo &= ~hit
    return out
pk = docs.drop_duplicates('appl_no').set_index('appl_no').ksic
sg = lab.sg.reindex(pk.index); dfu = pd.DataFrame({'key': sg, 'k': pk}).dropna()
ubL = sg.map(dfu.groupby('key').k.agg(lambda s: s.value_counts().idxmax())).fillna(pk)
# 양성 기업의 창 문서
PD = []
for s0 in DEV_B:
    ref = REF[s0]; end = add_months(ref, 6)
    w = docs[docs.pub.notna() & (docs.pub > ref) & (docs.pub <= end) & docs.b6.notna()].copy()
    w['key'] = w.b6 + '|' + w.nname
    w = w[w.key.isin(set(pos[pos.s0 == s0].key))]
    w['s0'] = s0; PD.append(w[['s0', 'key', 'appl_no', 'ksic']])
PD = pd.concat(PD, ignore_index=True)
ev = lambda df, col: set(zip(df.s0, df.key, df[col]))
Ep = ev(PD, 'ksic')
res = {}
for A in ['TXT-FREE', 'FREE-IPC', 'H-sc-1', 'UB']:
    parts = []; nul_i = np.zeros(NREP); nul_u = np.zeros(NREP); I = Un = 0; ag = []
    for s0 in DEV_B:
        t0 = int(s0[:4]); x = PD[PD.s0 == s0].copy()
        L = lab.k_free if A == 'TXT-FREE' else hard(t0, 'sg', 3) if A == 'FREE-IPC' else hard(t0, 'sc', 1) if A == 'H-sc-1' else ubL
        x['pub'] = x.appl_no.map(L).astype(object)
        Ea, Eo = ev(x.dropna(subset=['pub']), 'pub'), ev(x, 'ksic')
        I += len(Ea & Eo); Un += len(Ea | Eo); ag.append(float((x.pub.astype(str) == x.ksic).mean()))
        if A != 'UB':
            x = x.reset_index(drop=True); grp = x.groupby('ksic').indices
            for rep in range(NREP):
                rng = np.random.default_rng(20261012 + rep); newp = x.pub.values.copy()
                for kk, ix in grp.items():
                    newp[ix] = newp[ix][rng.permutation(len(ix))]
                xn = x.assign(pub=newp).dropna(subset=['pub'])
                En = ev(xn, 'pub'); nul_i[rep] += len(En & Eo); nul_u[rep] += len(En | Eo)
    r = dict(J_pub=round(I / Un, 4), doc_agree=round(float(np.mean(ag)), 4))
    if A != 'UB':
        nj = nul_i / nul_u; r.update(null_q05=round(float(np.quantile(nj, 0.05)), 4), null_mean=round(float(nj.mean()), 4), null_min=round(float(nj.min()), 4), null_max=round(float(nj.max()), 4))
    res[A] = r
    add('P5a', f'{A}', json.dumps(r))
pd.DataFrame(rows).to_csv(C + r'\s6_B_p3p5.csv', index=False, encoding='utf-8-sig')
json.dump(res, open(C + r'\s6_B_p5.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('done', round(time.time() - T0))
