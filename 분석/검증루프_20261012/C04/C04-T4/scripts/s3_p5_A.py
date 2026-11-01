# s3: C04-A/C P5 필수성(개발 원점만). X2 = 기업의 보이는 문서(최근 3년 출원, 없으면 전체) KSIC별 해외 4개국 출원 성장 g_k의 평균.
# 배정: prov / TXT-FREE / FREE-IPC / H-sc-1 / UB. 귀무: 원점 문서 집합에서 제공 라벨이 같은 문서끼리 공개 라벨 섞기(20회).
# P5-a: 원점별 X2 상위 20% (기업키,S0) 사건 자카드. P5-b: 단변량 AUC(X2→yA, yC) 차이, 기업키 군집 부트스트랩 200회.
import sys, json, time
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C04\C04-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libA import *
T0 = time.time(); NREP = 20; NB = 200
u = pd.read_parquet(C + r'\A_units_dev.parquet')
docs = load_docs()
lab = pd.read_parquet(B + r'\분석\검증루프_20261012\C02\C02-T4\cache\kr_publab_src.parquet')
lab['mg'] = lab.sg.str.split('/').str[0]; lab['sc'] = lab.sg.str[:4]
lab = lab.set_index('appl_no')
cube = pd.read_parquet(B + r'\공모전데이터\작업데이터\country_year_cube.parquet')
cube['ksic'] = cube['세세분류코드'].astype(str)
F = cube[cube['국가'].isin(['US', 'EP', 'CN', 'JP'])].groupby(['ksic', '출원연도'])['특허수'].sum().unstack(fill_value=0)


def gk(t0):
    a = F.reindex(columns=[t0 - 3, t0 - 2], fill_value=0).sum(axis=1); b = F.reindex(columns=[t0 - 5, t0 - 4], fill_value=0).sum(axis=1)
    return np.log((a + 1) / (b + 1))


def hard(t0, start, nmin):
    sub = lab[(lab.appl_year <= t0)]
    lv = ['sg', 'mg', 'sc'][['sg', 'mg', 'sc'].index(start):]
    out = pd.Series(np.nan, index=lab.index, dtype=object)
    todo = pd.Series(True, index=lab.index)
    for l in lv:
        s = sub[sub[l].notna()]
        tab = s.groupby([l, 'k_free']).size().reset_index(name='n')
        tot = tab.groupby(l).n.sum()
        mode = tab.sort_values([l, 'n', 'k_free'], ascending=[True, False, True]).drop_duplicates(l).set_index(l).k_free
        ok = tot[tot >= nmin].index
        v = lab[l].map(mode[mode.index.isin(ok)])
        hit = todo & v.notna()
        out[hit] = v[hit]; todo &= ~hit
    return out


def ub_label():
    pk = docs.drop_duplicates('appl_no').set_index('appl_no').ksic
    sg = lab.sg.reindex(pk.index)
    df = pd.DataFrame({'key': sg, 'k': pk}).dropna()
    md = df.groupby('key').k.agg(lambda s: s.value_counts().idxmax())
    return sg.map(md).fillna(pk)


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    m = ~np.isnan(s); y, s = y[m], s[m]
    r = pd.Series(s).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


# 원점별 기업 문서 집합
D = {}
for s0 in DEV_A:
    ref = REF[s0]; t0 = int(s0[:4])
    uu = u[u.s0 == s0]
    dv = docs[(docs.vis <= ref) & docs.nname.isin(set(uu.nname))][['appl_no', 'ay', 'nname', 'ksic']]
    assert dv.appl_no.map(docs.set_index('appl_no').vis).max() <= ref if len(dv) < 0 else True
    rec = dv[dv.ay >= t0 - 2]
    has_rec = set(rec.nname)
    dd = pd.concat([rec, dv[~dv.nname.isin(has_rec)]], ignore_index=True)
    dd['key'] = dd.nname.map(uu.set_index('nname').key)
    D[s0] = dd
    print(s0, 'docs', len(dd), 'units', len(uu), 'rec_share_units', round(len(has_rec) / len(uu), 3), flush=True)
ubL = ub_label()


def x2_of(dd, labels, g):
    v = pd.Series(labels, index=dd.index).map(g)
    return v.groupby(dd.key).mean()


rows = []; scores = {}
for A in ['prov', 'TXT-FREE', 'FREE-IPC', 'H-sc-1', 'UB']:
    tot_i = tot_u = 0; nul_i = np.zeros(NREP); nul_u = np.zeros(NREP); sc_all = []
    nul_scores = {r: [] for r in range(NREP)}
    for s0 in DEV_A:
        t0 = int(s0[:4]); g = gk(t0); dd = D[s0]; uu = u[u.s0 == s0].set_index('key')
        x_prov = x2_of(dd, dd.ksic.values, g).reindex(uu.index)
        thr = x_prov.quantile(0.8); Ep = set(x_prov[x_prov >= thr].index)
        if A == 'prov': L = None
        elif A == 'TXT-FREE': L = lab.k_free
        elif A == 'FREE-IPC': L = hard(t0, 'sg', 3)
        elif A == 'H-sc-1': L = hard(t0, 'sc', 1)
        else: L = ubL
        labs = dd.ksic.values if L is None else dd.appl_no.map(L).astype(object).values
        x = x2_of(dd, labs, g).reindex(uu.index)
        Ea = set(x[x >= x.quantile(0.8)].index)
        i, un = len(Ea & Ep), len(Ea | Ep)
        tot_i += i; tot_u += un
        agree = float(np.mean(pd.Series(labs).astype(str).values == dd.ksic.values))
        row = dict(assign=A, s0=s0, units=len(uu), docs=len(dd), doc_agree=round(agree, 4), doc_unassigned=round(float(pd.isna(labs).mean()), 4),
                   ev_prov=len(Ep), ev_pub=len(Ea), inter=i, union=un, jac=round(i / max(un, 1), 4),
                   spearman_prov_pub=round(float(pd.concat([x_prov, x], axis=1).corr(method='spearman').iloc[0, 1]), 4))
        sc_all.append(pd.DataFrame({'s0': s0, 'key': uu.index, 'x_prov': x_prov.values, 'x': x.values, 'yA': uu.yA.values, 'yC': uu.yC.values, 'eligC': uu.eligC.values}))
        if A in ('TXT-FREE', 'FREE-IPC', 'H-sc-1'):
            U = pd.DataFrame({'prov': dd.ksic.values, 'pub': labs}).reset_index(drop=True)
            grp = U.groupby('prov').indices
            for rep in range(NREP):
                rng = np.random.default_rng(20261012 + rep)
                newp = U.pub.values.copy()
                for kk, ix in grp.items():
                    newp[ix] = newp[ix][rng.permutation(len(ix))]
                if rep == 0:
                    row['null_doc_agree_check'] = round(float(np.mean(pd.Series(newp).astype(str).values == dd.ksic.values)), 4)
                xn = x2_of(dd, newp, g).reindex(uu.index)
                En = set(xn[xn >= xn.quantile(0.8)].index)
                nul_i[rep] += len(En & Ep); nul_u[rep] += len(En | Ep)
                nul_scores[rep].append(xn.values)
        rows.append(row); print(row, round(time.time() - T0), flush=True)
    S = pd.concat(sc_all, ignore_index=True); scores[A] = S
    res = dict(assign=A, pooled_jac=round(tot_i / tot_u, 4))
    for y in ['yA', 'yC']:
        SS = S if y == 'yA' else S[S.eligC == 1]
        res[f'auc_{y}'] = round(auc(SS[y], SS.x), 4)
        res[f'auc_{y}_within_mean'] = round(float(np.nanmean([auc(g_[y], g_.x) for _, g_ in SS.groupby('s0')])), 4)
        if A in ('TXT-FREE', 'FREE-IPC', 'H-sc-1'):
            na = []
            for rep in range(NREP):
                xs = np.concatenate(nul_scores[rep]); msk = (S.eligC.values == 1) if y == 'yC' else np.ones(len(S), bool)
                na.append(auc(S[y].values[msk], xs[msk]))
            res[f'null_auc_{y}_mean'] = round(float(np.mean(na)), 4); res[f'null_auc_{y}_q05_q95'] = [round(float(np.quantile(na, 0.05)), 4), round(float(np.quantile(na, 0.95)), 4)]
    if A in ('TXT-FREE', 'FREE-IPC', 'H-sc-1'):
        nj = nul_i / nul_u
        res.update(null_jac=[round(float(v), 4) for v in nj], null_q05=round(float(np.quantile(nj, 0.05)), 4), null_mean=round(float(nj.mean()), 4))
    rows.append(dict(assign=A, s0='POOLED', **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in res.items() if k != 'assign'}))
    print(res, round(time.time() - T0), flush=True)
# P5-b 부트스트랩: AUC_prov − AUC_pub (기업키 군집)
P = scores['prov']
kcode, keys = pd.factorize(P.key); rng = np.random.default_rng(20261012)
CNT = [rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys))) for b in range(NB)]
boot = []
for A in ['TXT-FREE', 'FREE-IPC', 'H-sc-1', 'UB']:
    Q = scores[A]
    assert (Q.key.values == P.key.values).all()
    for y in ['yA', 'yC']:
        m = (P.eligC.values == 1) if y == 'yC' else np.ones(len(P), bool)
        base = auc(P[y].values[m], P.x.values[m]) - auc(Q[y].values[m], Q.x.values[m])
        ds = []
        for b in range(NB):
            w = CNT[b][kcode]
            ix = np.repeat(np.arange(len(P)), w)
            ix = ix[m[ix]]
            ds.append(auc(P[y].values[ix], P.x.values[ix]) - auc(Q[y].values[ix], Q.x.values[ix]))
        boot.append(dict(assign=A, y=y, d_auc=round(float(base), 4), ci_lo=round(float(np.percentile(ds, 2.5)), 4), ci_hi=round(float(np.percentile(ds, 97.5)), 4)))
        print(boot[-1], round(time.time() - T0), flush=True)
pd.DataFrame(rows).to_csv(C + r'\s3_p5_A_rows.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(boot).to_csv(C + r'\s3_p5_A_boot.csv', index=False, encoding='utf-8-sig')
print('done', round(time.time() - T0))
