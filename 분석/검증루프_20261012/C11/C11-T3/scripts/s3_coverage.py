# -*- coding: utf-8 -*-
"""s3 (3) 커버리지 편향 + 우측 절단 + N5·N8 연결 정확도.
레코드 연결 = r3ros_rec_sci.n_us>0 (us_pn 3M–13M, 발행일 있음). 출원인 규모 = 같은 국가 안 첫 출원인 이름의 제공 레코드 수.
SMD(연결 vs 미연결): 출원연도, log 출원인 규모, IPC 개수, 해외 우선권 여부. TVD: KSIC(국가 안), 국가.
셀(T3 적격): ros_linked(=V0 ros_n>0) 1 vs 0 의 SMD(log kr_W, log tot_W, KR 0건, 부상).
우측 절단: 연속 번호 성질로 연도별 US 실용특허 등록 수(=다음 해 첫 번호 − 그해 첫 번호, KPSS 번호·발행일) 와 RoS·patent_sci 연도별 수 대조.
출력: res/s3_*.csv, res/s3_coverage.json"""
import json
from common import *
import numpy as np, pandas as pd
c = con()
R = {}
c.execute(f"""create table rs as select s.cc, s.appl_no, s.ksic, s.appl_year, (s.n_us>0) linked,
   split_part(coalesce(r.applicant,''),'|',1) app1, len(string_split(coalesce(r.ipc_all,''),'|')) n_ipc,
   (coalesce(r.prio_cc,'')<>'' and split_part(r.prio_cc,'|',1)<>s.cc) prio_foreign
   from read_parquet('{EXT}/r3ros_rec_sci.parquet') s join read_parquet('{W}/cache/rec.parquet') r using(cc, appl_no, ksic)""")
c.execute("create table asz as select cc, app1, count(*) n_app from rs where app1<>'' group by 1,2")
c.execute("create table rs2 as select rs.*, coalesce(ln(a.n_app),0) lsz from rs left join asz a using(cc, app1)")
R['n'] = c.sql("select cc, count(*), avg(linked::int) from rs2 group by 1 order by 1").fetchall()
# 출원연도별 연결률
lr = c.sql("select cc, appl_year, count(*) n, avg(linked::int) link_rate from rs2 where appl_year between 2000 and 2024 group by 1,2 order by 1,2").df()
lr.to_csv(W + "/res/s3_linkrate_cc_year.csv", index=False, encoding="utf-8-sig")
# 공유 csv 와 대조(원천 재계산)
sh = pd.read_csv(f"{EXT}/r3ros_link_rate_cc_year.csv")
m = lr.merge(sh[['cc', 'appl_year', 'link_rate']].rename(columns={'link_rate': 'shared'}), on=['cc', 'appl_year'])
R['linkrate_vs_shared_maxdiff'] = float((m.link_rate - m.shared).abs().max())
# 출원인 규모 사분위
q = c.sql("""select cc, ntile(4) over (partition by cc order by lsz) q, linked from rs2 where appl_year between 2005 and 2019""").df()
R['by_size_quartile'] = q.groupby(['cc', 'q']).linked.mean().round(4).unstack().to_dict('index')
# SMD (출원 2005–2019, T3 학습·개발 피처 창)
d = c.sql("select cc, ksic, appl_year, lsz, n_ipc, prio_foreign::int pf, linked from rs2 where appl_year between 2005 and 2019").df()
def smd(a, b):
    return float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)) if len(a) > 1 and len(b) > 1 else np.nan
rows = []
for cc, g in list(d.groupby('cc')) + [('ALL', d)]:
    L1, L0 = g[g.linked], g[~g.linked]
    r = dict(cc=cc, n=len(g), link_rate=g.linked.mean())
    for v in ('appl_year', 'lsz', 'n_ipc', 'pf'): r['smd_' + v] = smd(L1[v], L0[v])
    pa = L1.ksic.value_counts(normalize=True); pb = L0.ksic.value_counts(normalize=True)
    r['tvd_ksic'] = float(pa.subtract(pb, fill_value=0).abs().sum() / 2)
    if cc == 'ALL':
        pa = L1.cc.value_counts(normalize=True); pb = L0.cc.value_counts(normalize=True); r['tvd_cc'] = float(pa.subtract(pb, fill_value=0).abs().sum() / 2)
    rows.append(r)
S = pd.DataFrame(rows); S.round(4).to_csv(W + "/res/s3_bias_records.csv", index=False, encoding="utf-8-sig"); print(S.round(3).to_string())
# KSIC 별 연결률 범위
k = d.groupby('ksic').linked.mean(); R['ksic_linkrate_min_med_max'] = [float(k.min()), float(k.median()), float(k.max())]
kk = d[d.cc == 'KR'].groupby('ksic').linked.mean(); R['ksic_linkrate_KR_min_med_max'] = [float(kk.min()), float(kk.median()), float(kk.max())]
# 셀 수준 (T3 적격 셀): C09-T3 S0 예측(공식과 p 최대차 0 확인된 파일)의 kr_W·tot_W + V0 ros_n
L09 = BASE + "/분석/검증루프_20261012/C09/C09-T3/cache"
cells = []
for mode, files in (('dev', ['preds_S0_dev_2010-2011-2012-2013-2014.parquet', 'preds_S0_dev_2015-2016-2017-2018.parquet']), ('fin', ['preds_S0_final.parquet'])):
    p = pd.concat([pd.read_parquet(f"{L09}/{f}") for f in files]); sr = pd.read_parquet(f"{W}/cache/scen_ros_{mode}.parquet")[['cell', 't0', 'ros_n_V0', 'ros_n_V2']]
    p = p.merge(sr, on=['cell', 't0'], how='left'); assert p.ros_n_V0.notna().all()
    for t0, g in p.groupby('t0'):
        L1, L0 = g[g.ros_n_V0 > 0], g[g.ros_n_V0 == 0]
        cells.append(dict(t0=int(t0), n=len(g), linked=float((g.ros_n_V0 > 0).mean()), linked_V2=float((g.ros_n_V2 > 0).mean()),
                          smd_log_krW=smd(np.log1p(L1.kr_W), np.log1p(L0.kr_W)), smd_log_totW=smd(np.log1p(L1.tot_W), np.log1p(L0.tot_W)),
                          smd_kr0=smd((L1.kr_W == 0).astype(float), (L0.kr_W == 0).astype(float)),
                          emerg_linked=float(L1.yb.mean()), emerg_unlinked=float(L0.yb.mean()) if len(L0) else np.nan,
                          med_ros_n_linked=float(L1.ros_n_V0.median())))
CE = pd.DataFrame(cells); CE.round(4).to_csv(W + "/res/s3_bias_cells.csv", index=False, encoding="utf-8-sig"); print(CE.round(3).to_string())
# 우측 절단: 연도별 US 실용특허 등록 수(번호 연속성) vs RoS·patent_sci
kp = c.sql(f"select year(issue_date::date) y, min(patent_num) mn, max(patent_num) mx, count(*) n_kpss from read_parquet('{EXT}/quality_kpss_2025.parquet') group by 1 order by 1").df()
kp['grants_by_number'] = kp.mn.shift(-1) - kp.mn
c.execute(f"create table rosp as select distinct pn from read_parquet('{EXT}/r3ros_work/cit.parquet') where pn between 3000000 and 13000000")
c.execute(f"create table kpd as select patent_num pn, issue_date::date idt from read_parquet('{EXT}/quality_kpss_2025.parquet')")
ry = c.sql("select year(k.idt) y, count(*) n_ros from rosp r asof join kpd k on r.pn >= k.pn group by 1 order by 1").df()
py = c.sql(f"select issue_year y, count(*) n_patsci, sum((n_cit>0)::int) n_patsci_cit from read_parquet('{EXT}/r3ros_patent_sci.parquet') group by 1").df()
T = kp.merge(ry, on='y', how='left').merge(py, on='y', how='left')
T['ros_share_of_grants'] = T.n_ros / T.grants_by_number; T['kpss_share'] = T.n_kpss / T.grants_by_number
T['patsci_share_of_grants'] = T.n_patsci / T.grants_by_number; T['patsci_sci_share'] = T.n_patsci_cit / T.n_patsci
T = T[(T.y >= 2000) & (T.y <= 2025)]
T.round(4).to_csv(W + "/res/s3_truncation_issueyear.csv", index=False, encoding="utf-8-sig"); print(T.round(3).to_string())
# N5·N8: 같은 파일 사용 → KR 쌍 정확도(s2 표본), KPSS 연결 부분집합
F = f"read_parquet('{W}/cache/s2_fin.parquet')"
R['N5_KR_all'] = c.sql(f"select kind, count(*), sum((n_u>0)::int), avg(case when n_u>0 then ok::int end), avg(case when n_u>0 then m2::int end) from {F} where cc='KR' group by 1 order by 1").fetchall()
R['N8_KR_kpss'] = c.sql(f"""select kind, count(*), sum((n_u>0)::int), avg(case when n_u>0 then ok::int end), avg(case when n_u>0 then m2::int end)
   from {F} where cc='KR' and us_pn in (select patent_num from read_parquet('{EXT}/quality_kpss_2025.parquet')) group by 1 order by 1""").fetchall()
R['kpss_link_regex_same'] = c.sql(f"""select count(*), count(distinct us_pn) from read_parquet('{EXT}/quality_kpss_family_link.parquet')""").fetchone()
R['kpss_link_pairs_in_fam_us'] = c.sql(f"""select count(*), sum((f.us_pn is not null)::int) from (select distinct cc, appl_no, ksic, us_pn from read_parquet('{EXT}/quality_kpss_family_link.parquet')) k
   left join (select distinct cc, appl_no, ksic, us_pn from read_parquet('{EXT}/r3ros_work/fam_us.parquet') where src='family') f using(cc, appl_no, ksic, us_pn)""").fetchone()
print(R)
json.dump(R, open(W + "/res/s3_coverage.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log('done')
