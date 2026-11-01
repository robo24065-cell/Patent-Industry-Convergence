# -*- coding: utf-8 -*-
"""C18-T1 verify_0 반박 재계산(독립 경로). 원천 읽기 전용(sqlite mode=ro). 쓰기는 verify_0 안만."""
import sys, pathlib, json, sqlite3, time
import pandas as pd, numpy as np
sys.dont_write_bytecode = True
t0 = time.time()
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
V = B/'분석/검증루프_20261012/C18/C18-T1/verify_0'
ro = lambda p: sqlite3.connect(pathlib.Path(p).as_uri() + '?mode=ro', uri=True)
END = '20250822'; o = {}
W = B/'공모전데이터/작업데이터'
KP = pd.read_parquet(W/'kr_patents.parquet', columns=['출원번호', '등록번호', '등록일자'])
KP.columns = ['a', 'r', 'd']
for c in KP: KP[c] = KP[c].astype(str).str.strip()
KR = pd.read_parquet(W/'kr_registered.parquet', columns=['출원번호', '등록번호', '등록일자'])
KR.columns = ['a', 'r', 'd']
for c in KR: KR[c] = KR[c].astype(str).str.strip()
isnum = KP.r.str.fullmatch(r'\d{13}')
o['kp_rows'] = len(KP); o['kp_rgst_13digit'] = int(isnum.sum()); o['kp_rgst_other_values_top'] = KP.loc[~isnum, 'r'].value_counts().head(5).to_dict()
o['kp_rgst_prefix'] = KP.loc[isnum, 'r'].str[:2].value_counts().to_dict()
o['krr_rows'] = len(KR); o['krr_prefix'] = KR.r.str[:2].value_counts().to_dict()
o['krr_vs_kp_nonblank_symdiff'] = len(set(KR.r) ^ set(KP.loc[isnum, 'r']))
o['krr_appl_not_in_kp'] = int((~KR.a.isin(set(KP.a))).sum())
blankA = set(KP.loc[~isnum, 'a'])
# registry.sqlite
g = ro(B/'수집파이프라인/registry.sqlite')
o['reg_tables'] = [r[0] for r in g.execute("select name from sqlite_master where type='table'")]
o['reg_status'] = g.execute('select status, count(*) from registry group by 1').fetchall()
pend = pd.read_sql('select rgst_no, appl_no from pending', g)
regno = set(r[0] for r in g.execute('select rgst_no from registry'))
g.close()
pend = pend.astype(str)
o['pending_n'] = len(pend); o['pending_vs_krr_symdiff'] = len(set(pend.rgst_no) ^ set(KR.r))
o['registry_vs_pending_symdiff'] = len(regno ^ set(pend.rgst_no))
o['blank_appl_in_pending'] = int(len(blankA & set(pend.appl_no)))
for f in sorted(W.glob('registry_snapshot_2026081*.sqlite')):
    c = ro(f)
    tabs = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
    res = {'tables': tabs}
    if 'pending' in tabs:
        cols = [r[1] for r in c.execute('pragma table_info(pending)')]
        res['pending_cols'] = cols
        pa = set(str(r[0]) for r in c.execute('select appl_no from pending')) if 'appl_no' in cols else set()
        pr = set(str(r[0]) for r in c.execute('select rgst_no from pending'))
        res['n'] = len(pr); res['blank_appl_in'] = len(pa & blankA); res['rgst_not_in_krr'] = len(pr - set(KR.r))
    if 'registry' in tabs:
        rr = set(str(r[0]) for r in c.execute('select rgst_no from registry'))
        res['registry_n'] = len(rr); res['registry_not_in_krr'] = len(rr - set(KR.r))
    c.close(); o['snap_' + f.stem[-8:]] = res
# registry_full
rc = ro(B/'수집파이프라인/registry_full.sqlite')
o['rf_tables'] = [r[0] for r in rc.execute("select name from sqlite_master where type='table'")]
RF = pd.read_sql("select rgst_no, applNo, rgstDate from patent", rc); rc.close()
RF = RF.astype(str)
o['rf_all_rows'] = len(RF); o['rf_prefix'] = RF.rgst_no.str[:2].value_counts().to_dict()
RFp = RF[RF.rgst_no.str[:2] == '10']
KRp = KR[KR.r.str[:2] == '10']
o['rf10_n'] = len(RFp); o['rf10_unique'] = RFp.rgst_no.nunique(); o['kr10_n'] = len(KRp)
o['rf10_vs_kr10_symdiff'] = len(set(RFp.rgst_no) ^ set(KRp.r))
o['rf_all_not_in_krr'] = len(set(RF.rgst_no) - set(KR.r))
# monthly RF/KR in gap months
rfm = RFp[RFp.rgstDate <= END].rgstDate.str[:6].value_counts(); krm = KRp[KRp.d <= END].d.str[:6].value_counts()
mm = pd.DataFrame({'rf': rfm, 'kr': krm}).fillna(0).sort_index()
o['rf_kr_ratio_201801_202508'] = {k: round(v, 4) for k, v in (mm.rf / mm.kr).loc['201801':'202508'].items() if v != 1}
o['rf_monthly_2023_2025'] = mm.loc['202301':'202508', 'rf'].astype(int).to_dict()
# right_list: '10', rgst_date 2015-01-01..END, appl in kp
bc = ro(B/'수집파이프라인/business_rights.sqlite')
rl = pd.read_sql(f"select rgst_no, rgst_date, appl_no from right_list where substr(rgst_no,1,2)='10' and rgst_date>='20150101' and rgst_date<='{END}'", bc)
rl = rl.astype(str)
P = set(KP.a)
rl = rl[rl.appl_no.isin(P)]
rlu = rl.sort_values('rgst_date').drop_duplicates('rgst_no')
o['rl_in_unique'] = len(rlu)
miss = rlu[~rlu.rgst_no.isin(set(RFp.rgst_no))].copy()
o['missing_n'] = len(miss); o['missing_by_year'] = miss.rgst_date.str[:4].value_counts().sort_index().to_dict()
o['missing_unique_appl'] = miss.appl_no.nunique()
miss['kp_blank'] = miss.appl_no.isin(blankA)
kpr = dict(zip(KP.a, KP.r)); miss['kp_r'] = miss.appl_no.map(kpr)
o['missing_kp_blank'] = int(miss.kp_blank.sum())
o['missing_kp_nonblank_examples'] = miss.loc[~miss.kp_blank, ['rgst_no', 'rgst_date', 'appl_no', 'kp_r']].head(5).to_dict('records')
o['missing_rgst_eq_kp_rgst'] = int((miss.kp_r == miss.rgst_no).sum())
o['missing_appl_in_pending'] = int(miss.appl_no.isin(set(pend.appl_no)).sum())
o['missing_rgst_in_pending'] = int(miss.rgst_no.isin(set(pend.rgst_no)).sum())
mmiss = miss.rgst_date.str[:6].value_counts().sort_index()
o['missing_monthly_202310_202508'] = mmiss.loc['202310':'202508'].to_dict()
# compare with s4 set
s4 = pd.read_parquet(V.parent/'cache/s4_missing_docs.parquet')
o['s4_vs_mine_appl_symdiff'] = len(set(s4.appl_no.astype(str)) ^ set(miss.appl_no))
o['s4_vs_mine_rgst_symdiff'] = len(set(s4.rgst_no.astype(str)) ^ set(miss.rgst_no))
miss.to_parquet(V/'r0_missing.parquet', index=False)
# C17 413
try:
    Dd = pd.read_parquet(B/'분석/검증루프_20261012/C16/C16-T1/cache/alias_docs.parquet')
    o['alias_cols'] = list(Dd.columns)
    Dd['appl_no'] = Dd.appl_no.astype(str)
    PU = set(Dd[(Dd.grp == 'P') & (~Dd.is_reg.astype(bool))].appl_no)
    o['PU_n'] = len(PU); o['PU_kp_blank'] = len(PU & blankA); o['PU_not_in_kp'] = len(PU - P)
    q = ','.join("'" + a + "'" for a in PU)
    h = pd.read_sql(f"select rgst_no, rgst_date, appl_no from right_list where appl_no in ({q})", bc).astype(str)
    o['C17_hit_rows'] = len(h); o['C17_hit_unique_appl'] = h.appl_no.nunique()
    hh = h.sort_values('rgst_date').drop_duplicates('appl_no')
    e = hh[hh.rgst_date <= END]
    o['C17_early'] = len(e); o['C17_early_prefix'] = e.rgst_no.str[:2].value_counts().to_dict()
    e10 = e[e.rgst_no.str[:2] == '10']
    o['C17_early10_in_missing_rgst'] = int(e10.rgst_no.isin(set(miss.rgst_no)).sum())
    o['C17_early10_in_missing_appl'] = int(e10.appl_no.isin(set(miss.appl_no)).sum())
    o['C17_early_in_RF_any'] = int(e.rgst_no.isin(set(RF.rgst_no)).sum())
    # any appl with multiple rgst (10 & 20)?
    o['C17_hits_appl_multi_rgst'] = int((h.groupby('appl_no').rgst_no.nunique() > 1).sum())
    e20 = e[e.rgst_no.str[:2] == '20']
    o['C17_early20_rows'] = e20.to_dict('records')
    o['C17_early20_appl_prefix'] = e20.appl_no.str[:2].tolist()
except Exception as ex:
    o['C17_err'] = repr(ex)
bc.close()
o['secs'] = round(time.time() - t0)
json.dump(o, open(V/'r0_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, default=str)[:6000])
