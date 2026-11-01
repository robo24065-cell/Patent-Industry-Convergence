# s6_C: C03-C P5 강화 필수성(개발 원점만). 인자: 배정 이름(TXT-FREE | FREE-IPC | H-sc-1 | UB)
# 공개 배정 hard()는 C02-T4 s3_essential.py와 같은 구현. 귀무: 원점 문서 집합 안에서 제공 라벨이 같은 문서끼리 공개 라벨을 섞는다(20회).
import sys, json, time
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C03\C03-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libC import *
A = sys.argv[1]; NREP = 20
T0 = time.time()
d = load()
lab = pd.read_parquet(C2 + r'\kr_publab_src.parquet')
lab['mg'] = lab.sg.str.split('/').str[0]; lab['sc'] = lab.sg.str[:4]
lab = lab.set_index('appl_no')
P = pd.read_parquet(C + r'\C_dev_pos.parquet')


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
        mp = mode[mode.index.isin(ok)]
        v = lab[l].map(mp)
        hit = todo & v.notna()
        out[hit] = v[hit]; todo &= ~hit
    return out


def ub_label():
    pk = d.drop_duplicates('appl_no').set_index('appl_no').ksic
    sg = lab.sg.reindex(pk.index)
    df = pd.DataFrame({'key': sg, 'k': pk}).dropna()
    md = df.groupby('key').k.agg(lambda s: s.value_counts().idxmax())
    return sg.map(md).fillna(pk)


def jac(Ea, Ep):
    i = len(Ea & Ep); return i, len(Ea | Ep)


rows = []
ubL = ub_label() if A == 'UB' else None
for t0 in DEV:
    Ep = set(zip(P[P.t0 == t0].cd, P[P.t0 == t0].k))
    if A == 'TXT-FREE': L = lab.k_free
    elif A == 'FREE-IPC': L = hard(t0, 'sg', 3)
    elif A == 'H-sc-1': L = hard(t0, 'sc', 1)
    else: L = ubL
    pub = d.appl_no.map(L).astype(object)
    st, pos, r = build(t0, d, lab=pub)
    Ea = set(zip(pos.cd, pos.k)); i, u = jac(Ea, Ep)
    # 원점 문서 집합(기준+창) 일치율·혼동 보존 귀무
    msk = (d.p >= f'{t0-2}0101') & (d.p <= f'{t0+1}0630')
    U = d[msk].drop_duplicates('appl_no')[['appl_no', 'ksic']].copy()
    U['pub'] = U.appl_no.map(L).values
    agree = float((U.pub == U.ksic).mean()); unas = float(U.pub.isna().mean())
    row = dict(t0=t0, assign=A, ev_pub=len(Ea), ev_prov=len(Ep), inter=i, union=u, jac=round(i / max(u, 1), 4),
               doc_agree=round(agree, 4), doc_unassigned=round(unas, 4), firm_jac=round(len({c for c, _ in Ea} & {c for c, _ in Ep}) / max(len({c for c, _ in Ea} | {c for c, _ in Ep}), 1), 4))
    if A != 'UB':
        U = U.sort_values('ksic').reset_index(drop=True)
        grp = U.groupby('ksic').indices
        for rep in range(NREP):
            rng = np.random.default_rng(20261012 + rep)
            newp = U.pub.values.copy()
            for kk, ix in grp.items():
                newp[ix] = newp[ix][rng.permutation(len(ix))]
            mp = dict(zip(U.appl_no, newp))
            nl = d.appl_no.map(mp).astype(object)
            st2, pos2, _ = build(t0, d, lab=nl)
            i2, u2 = jac(set(zip(pos2.cd, pos2.k)), Ep)
            row[f'null_inter_{rep}'] = i2; row[f'null_union_{rep}'] = u2
            if rep == 0:
                row['null_doc_agree_check'] = round(float((pd.Series(newp) == U.ksic).mean()), 4)
    rows.append(row); print({k: v for k, v in row.items() if not k.startswith('null_')}, round(time.time() - T0), flush=True)
R = pd.DataFrame(rows)
R.to_csv(C + rf'\s6_C_p5_{A}.csv', index=False, encoding='utf-8-sig')
res = dict(assign=A, pooled_jac=round(R.inter.sum() / R.union.sum(), 4), pooled_recall=round(R.inter.sum() / R.ev_prov.sum(), 4),
           ev_pub=int(R.ev_pub.sum()), ev_prov=int(R.ev_prov.sum()), doc_agree_mean=round(R.doc_agree.mean(), 4))
if A != 'UB':
    nj = [R[f'null_inter_{r}'].sum() / R[f'null_union_{r}'].sum() for r in range(NREP)]
    res.update(null_jac=[round(x, 4) for x in nj], null_q05=round(float(np.quantile(nj, 0.05)), 4), null_mean=round(float(np.mean(nj)), 4),
               null_max=round(float(np.max(nj)), 4))
res['elapsed'] = round(time.time() - T0, 1)
json.dump(res, open(C + rf'\s6_C_p5_{A}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(res)
