# s1: 위험집합(①10자리 NTIS 차단 포함) + 개발 레이블 + M0 원자료 + M1 침투율·흐름. 잠금 S0=2024-12는 위험집합·피처만(레이블 계산 없음).
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libC import *
T0 = time.time()
ca = load_ca(); nm2b6, _ = name_map(ca)
docs = load_docs(); docs['b6'] = docs.nname.map(nm2b6)
first_vis = docs[docs.b6.notna() & docs.vis.notna()].groupby('nname').vis.min()
nt = load_ntis()
# CA (정규화 상호, biz6) → 10자리 번호 집합
cab = ca[ca.has_biz & (ca.nname.str.len() >= 1)][['nname', 'biz6', 'bizno']].drop_duplicates()
cab['key'] = cab.biz6 + '|' + cab.nname
c04pos = pd.read_parquet(os.path.join(B, '분석', '검증루프_20261012', 'C04', 'C04-T4', 'cache', 'B_dev_pos.parquet'))
recs = []; U = []; P = []; FORMS = {}
for s0 in DEV + [LOCK]:
    ref = REF[s0]; t0 = int(s0[:4]); end = add_months(ref, 6)
    d0, nform = load_nps(s0, tag='s1_S0'); FORMS[s0] = nform
    f0 = firm_table(d0, ref)
    fE = f0[f0.E >= 5].copy()
    vis_names = first_vis[first_vis <= ref].index
    vis_keys = set(pd.Series(vis_names).map(nm2b6).astype(str) + '|' + pd.Series(vis_names))
    fE['linked'] = fE.index.isin(vis_keys)
    fv_key = pd.Series(fE.index.str.split('|').str[1], index=fE.index).map(first_vis)
    fE['flow1'] = (fE.linked & (fv_key > add_months(ref, -12)) & (fv_key <= ref)).astype(int)
    # M1 침투율·흐름(같은 S0 법인 E0≥5 전체 기준, 평활 m=50)
    def sm(grp, par, col, m=50):
        a = fE.groupby(grp)[col].agg(['sum', 'size']); p = fE.groupby(par)[col].mean()
        pm = fE[[grp, par]].drop_duplicates(grp).set_index(grp)[par].map(p)
        r = (a['sum'] + m * pm.reindex(a.index)) / (a['size'] + m)
        return fE[grp].map(r)
    fE['lnk'] = fE.linked.astype(int)
    fE['pen_ind6'] = np.log(sm('ind6', 'ind2', 'lnk') + 1e-6)
    fE['pen_sgg'] = np.log(sm('sgg', 'sido', 'lnk') + 1e-6)
    fE['flow_ind6'] = np.log(sm('ind6', 'ind2', 'flow1') + 1e-6)
    # 차단
    ntk6 = set(nt[nt.year.between(t0 - 5, t0 - 3) & nt.is_tc].key6)
    nt10 = set(nt[nt.year.between(t0 - 5, t0 - 3)].bz)
    blk10_keys = set(cab[cab.bizno.isin(nt10)].key)
    risk0 = fE[~fE.linked]
    r_i = risk0[~risk0.index.isin(ntk6)]
    b10 = r_i.index.isin(blk10_keys)
    risk = r_i[~b10].copy()
    # 직전 스냅샷
    if s0 in PREV:
        dp, _ = load_nps(PREV[s0], cols=('name', 'biz6', 'stat', 'form', 'n_emp'), tag='s1_prev')
        Ep = dp.groupby('key').n_emp.sum()
        risk['E_prev'] = risk.index.map(Ep)
        risk['newkey'] = risk.E_prev.isna().astype(float)
        risk['demp'] = np.log((risk.E + 1) / (risk.E_prev.fillna(0) + 1))
        risk['noprev'] = 0
    else:
        risk['E_prev'] = np.nan; risk['newkey'] = np.nan; risk['demp'] = np.nan; risk['noprev'] = 1
    rec = dict(s0=s0, ref=ref, end=end, nps_E0ge5=len(fE), linked=int(fE.linked.sum()), excl_ntis_i=int(len(risk0) - len(r_i)),
               excl_ntis10=int(b10.sum()), risk=len(risk), form_counts_raw=nform)
    if s0 in DEV:
        assert end <= DEV_PUB_MAX
        w = docs[docs.pub.notna() & (docs.pub > ref) & (docs.pub <= end) & docs.b6.notna()]
        assert w.pub.max() <= DEV_PUB_MAX
        wl = w[w.nname.map(first_vis) > ref].copy()
        wl['key'] = wl.b6 + '|' + wl.nname
        pos_i = set(wl.key) & set(r_i.index)
        pos = set(wl.key) & set(risk.index)
        risk['y'] = risk.index.isin(pos).astype(np.int8)
        c4 = set(c04pos[c04pos.s0 == s0].key)
        rec.update(pos_before_blk10=len(pos_i), pos=len(pos), pos_removed_by_blk10=len(pos_i - pos), c04_pos=len(c4),
                   c04_match=len(c4 & pos_i), pos_rate=round(len(pos) / len(risk), 6),
                   risk_before_blk10=len(r_i))
        P.append(pd.DataFrame({'s0': s0, 'key': sorted(pos)}))
    else:
        risk['y'] = -1  # 잠금: 레이블 미계산
    risk['s0'] = s0
    U.append(risk.reset_index().rename(columns={'index': 'key'}))
    recs.append(rec); print(rec, round(time.time() - T0), flush=True)
UU = pd.concat(U, ignore_index=True)
keep = ['key', 's0', 'y', 'E', 'n_wkpl', 'n_new', 'n_loss', 'age', 'ind6', 'ind4', 'ind2', 'sido', 'sgg', 'E_prev', 'newkey', 'demp', 'noprev',
        'pen_ind6', 'pen_sgg', 'flow_ind6']
UU[keep].to_parquet(os.path.join(C, 'units.parquet'), index=False)
PP = pd.concat(P, ignore_index=True)
PP.to_parquet(os.path.join(C, 'dev_pos.parquet'), index=False)
dv = [r for r in recs if r['s0'] in DEV]
summ = dict(records=recs, P1=dict(unique=int(PP.key.nunique()), origin_sum=int(len(PP)), per_origin={r['s0']: r['pos'] for r in dv},
            lt200=int(sum(r['pos'] < 200 for r in dv)), removed_total=int(sum(r['pos_removed_by_blk10'] for r in dv)),
            before_total=int(sum(r['pos_before_blk10'] for r in dv)), c04_total=int(sum(r['c04_pos'] for r in dv)),
            c04_match_total=int(sum(r['c04_match'] for r in dv))))
json.dump(summ, open(os.path.join(C, 's1_log.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(summ['P1'], 'done', round(time.time() - T0))
