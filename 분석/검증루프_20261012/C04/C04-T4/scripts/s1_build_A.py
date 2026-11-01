# s1: C04-A/C 개발 단위 구성(as-of 연결) + 결과(개발 S1 ≤ 2023-12) + 연결/비연결 출원인 표 + 잠금 S0 모집단 크기만.
import sys, json, time
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C04\C04-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libA import *
T0 = time.time()
ca = load_ca()
nm2b6, nst = name_map(ca)
docs = load_docs()
docs['b6'] = docs.nname.map(nm2b6)
log = dict(ca_rows=len(ca), ca_has_biz=int(ca.has_biz.sum()), ca_corp_tc=int(ca.corp_tc.sum()), name_ok=len(nm2b6),
           docs=len(docs), docs_vis_null=int(docs.vis.isna().sum()), docs_name_ok=int(docs.b6.notna().sum()))
print(log, flush=True)
U, A, R = [], [], []
for s0 in DEV_A + LOCK_S0:
    ref = REF[s0]
    dv = docs[(docs.vis.notna()) & (docs.vis <= ref)]
    assert dv.vis.max() <= ref
    g = dv[dv.b6.notna()].groupby('nname')
    ap = g.agg(n_vis=('appl_no', 'size'), first_ay=('ay', 'min'), b6=('b6', 'first'),
               reg_by_ref=('rg', lambda s: float((s.notna() & (s <= ref)).mean())),
               later_reg=('rg', lambda s: float((s.notna() & (s > ref)).mean())),
               dom_prio=('dom_prio', 'mean')).reset_index()
    ap['key'] = ap.b6 + '|' + ap.nname
    d0 = load_nps(s0)
    f0 = firm_table(d0)
    ap['linked'] = ap.key.isin(f0.index)
    ap['E0'] = ap.key.map(f0.E)
    ap['s0'] = s0
    # 출원인 KSIC 최빈(보이는 문서)
    km = dv[dv.nname.isin(set(ap.nname))].groupby(['nname', 'ksic']).size().reset_index(name='n').sort_values(['nname', 'n'], ascending=[True, False]).drop_duplicates('nname')
    ap['ksic_mode'] = ap.nname.map(km.set_index('nname').ksic)
    A.append(ap)
    u = ap[ap.linked & (ap.E0 >= 5)][['s0', 'key', 'nname', 'n_vis', 'first_ay', 'reg_by_ref', 'later_reg', 'dom_prio', 'ksic_mode']].copy()
    u = u.join(f0[['E', 'amt', 'n_wkpl', 'ind2', 'sido']], on='key').rename(columns={'E': 'E0', 'amt': 'amt0'})
    rec = dict(s0=s0, ref=ref, patent_names=len(ap), linked=int(ap.linked.sum()), link_rate=round(float(ap.linked.mean()), 4),
               units_E0ge5=len(u), units_E0ge10=int((u.E0 >= 10).sum()), nps_corp_keys=len(f0), nps_corp_keys_E0ge5=int((f0.E >= 5).sum()))
    if s0 in S1_OF:
        s1 = S1_OF[s0]
        assert s1 <= '202312'
        d1 = load_nps(s1)
        e = outcome_E1(set(u.key), d0, d1)
        u = u.join(e, on='key')
        u[['E1', 'amt1']] = u[['E1', 'amt1']].fillna(0); u['n_rows_by_ba'] = u.n_rows_by_ba.fillna(0)
        u['s1'] = s1
        u['yA'] = ((u.E1 >= 1.3 * u.E0) & (u.E1 - u.E0 >= 10)).astype(np.int8)
        # C04-C
        el = (u.E0 >= 10) & (u.E1 >= 0.8 * u.E0)
        w0 = u.amt0 / u.E0.replace(0, np.nan); w1 = u.amt1 / u.E1.replace(0, np.nan)
        u['lw'] = np.log(w1 / w0)
        thr = u.loc[el, 'lw'].quantile(0.9)
        u['yC'] = (el & (u.lw >= thr)).astype(np.int8)
        u['eligC'] = (u.E0 >= 10).astype(np.int8)
        rec.update(posA=int(u.yA.sum()), rateA=round(float(u.yA.mean()), 4), posC=int(u.yC.sum()), C_thr_lw=round(float(thr), 4),
                   exit_E1_0=round(float((u.E1 == 0).mean()), 4), via_ba_rows=int(u.n_rows_by_ba.sum()))
        # 상호 변경으로 S0에서 놓친 특허기업(개발 진단): S0 키 k_old(미연결) → S1 같은 ba의 새 키 k_new가 '보이는 문서가 있는 CA 상호'
        u0b, u1b = uniq_ba(d0), uniq_ba(d1)
        b2old = d0[d0.ba.isin(u0b)].set_index('ba').key
        d1x = d1[d1.ba.isin(u1b)].copy(); d1x['k_old'] = d1x.ba.map(b2old)
        rn = d1x[d1x.k_old.notna() & (d1x.key != d1x.k_old)]
        apk = set(ap.key)
        miss = rn[rn.key.isin(apk) & ~rn.k_old.isin(set(ap.loc[ap.linked, 'key']))].k_old.unique()
        mm = pd.DataFrame({'key': miss}).join(f0[['E']], on='key').rename(columns={'E': 'E0'})
        mm = mm[mm.E0 >= 5]
        em = outcome_E1(set(mm.key), d0, d1)
        mm = mm.join(em, on='key'); mm['E1'] = mm.E1.fillna(0)
        mm['yA'] = ((mm.E1 >= 1.3 * mm.E0) & (mm.E1 - mm.E0 >= 10)).astype(int)
        rec.update(missed_by_rename_2y=len(mm), missed_rate_vs_units=round(len(mm) / max(len(u), 1), 4),
                   missed_posA_rate=round(float(mm.yA.mean()), 4) if len(mm) else None,
                   renamed_linked_units_share=round(float(u.key.isin(set(rn.k_old)).mean()), 4))
        mm['s0'] = s0
        R.append(mm)
        del d1, d1x
    U.append(u)
    print(rec, round(time.time() - T0), flush=True)
    log[f'rec_{s0}'] = rec
units = pd.concat([x for x in U if 's1' in x.columns], ignore_index=True)
units.to_parquet(C + r'\A_units_dev.parquet', index=False)
lock_pop = pd.concat([x for x in U if 's1' not in x.columns], ignore_index=True)[['s0', 'E0']]
lock_pop.groupby('s0').agg(units_E0ge5=('E0', 'size'), units_E0ge10=('E0', lambda s: int((s >= 10).sum()))).to_csv(C + r'\A_lock_pop_counts.csv', encoding='utf-8-sig')
pd.concat(A, ignore_index=True).to_parquet(C + r'\A_applicants.parquet', index=False)
pd.concat(R, ignore_index=True).to_parquet(C + r'\A_missed_rename.parquet', index=False)
json.dump(log, open(C + r'\s1_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print('done', round(time.time() - T0))
