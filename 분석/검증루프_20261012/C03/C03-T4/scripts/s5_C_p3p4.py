# s5_C: C03-C P3(기업 코드 연결 편향, 개발) + P4(잠금창 월별 건수만)
import sys, json, time
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C03\C03-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, duckdb
from libC import *
T0 = time.time(); out = {}
KRP = (B + r'\공모전데이터\작업데이터\kr_patents.parquet').replace(chr(92), '/')
DOCS = (C2 + r'\docs.parquet').replace(chr(92), '/')
dk = duckdb.connect(); dk.execute("set memory_limit='3GB'; set threads=2")
# ---- P4: 잠금 구간 월별 건수(집계만) ----
q = f"""with kp as (select 출원번호 appl_no, 공개일자 p, 출원연도 ay from read_parquet('{KRP}') where length(공개일자)=8 and 공개일자>='20240101'),
        dc as (select distinct cd, appl_no from read_parquet('{DOCS}') where substr(cd,1,1)='1'),
        anyc as (select distinct appl_no from read_parquet('{DOCS}'))
        select substr(kp.p,1,6) ym, count(*) n_pub_all, sum((kp.appl_no in (select appl_no from anyc))::int) n_pub_coded,
               count(distinct case when dc.cd is not null then kp.appl_no end) n_pub_corp_docs,
               count(distinct dc.cd) n_corp_firms
        from kp left join dc using (appl_no) group by 1 order by 1"""
M = dk.sql(q).df(); M.to_csv(C + r'\s5_C_monthly_2024_2026.csv', index=False, encoding='utf-8-sig')
print(M.to_string())
# 잠금 창 문서의 출원연도 구성(건수만): T1 최종 결과창(출원 2020–2023)과의 문서 겹침
q2 = f"""select 출원연도 ay, count(*) n from read_parquet('{KRP}') kp where length(공개일자)=8 and 공개일자 between '20250101' and '20250630'
         and 출원번호 in (select appl_no from read_parquet('{DOCS}') where substr(cd,1,1)='1') group by 1 order by 1"""
AY = dk.sql(q2).df(); out['lock_docs_corp_by_appl_year'] = AY.values.tolist()
# 잠금창 적격 기업 수(창 ≥3건) — 건수만
q3 = f"""with x as (select dc.cd, count(distinct kp.출원번호) n from read_parquet('{KRP}') kp join (select distinct cd, appl_no from read_parquet('{DOCS}') where substr(cd,1,1)='1') dc on kp.출원번호=dc.appl_no
         where length(kp.공개일자)=8 and kp.공개일자 between '{{s}}' and '{{e}}' group by 1) select count(*) from x where n>=3"""
out['firms_ge3_window'] = {y: dk.sql(q3.format(s=f'{y}0101', e=f'{y}0630')).fetchone()[0] for y in range(2019, 2026)}
mm = M.set_index('ym')
lock = mm.loc[[f'2025{m:02d}' for m in range(1, 7)], 'n_pub_corp_docs']; prev = mm.loc[[f'2024{m:02d}' for m in range(1, 7)], 'n_pub_corp_docs']
out['P4a_end_trunc'] = round(float(lock.iloc[3:].median() / lock.iloc[:3].median()), 4)
out['P4b_same_period'] = round(float(lock.sum() / prev.sum()), 4)
out['P4b_firms_ge3'] = round(out['firms_ge3_window'][2025] / out['firms_ge3_window'][2024], 4)
lk_all = mm.loc[[f'2025{m:02d}' for m in range(1, 7)]]; pv_all = mm.loc[[f'2024{m:02d}' for m in range(1, 7)]]
out['diag_code_coverage_2025H1_vs_2024H1'] = [round(float(lk_all.n_pub_coded.sum() / lk_all.n_pub_all.sum()), 4), round(float(pv_all.n_pub_coded.sum() / pv_all.n_pub_all.sum()), 4)]
out['diag_all_pub_ratio'] = round(float(lk_all.n_pub_all.sum() / pv_all.n_pub_all.sum()), 4)
# ---- P3: 기업 코드 연결(개발 공개 2010–2024) 연결 vs 비연결 ----
kp = pd.read_parquet(B + r'\공모전데이터\작업데이터\kr_patents.parquet', columns=['출원번호', '공개일자', '세세분류코드', '우선권국가', '출원인명'])
kp.columns = ['appl_no', 'p', 'ksic', 'prio', 'an']; kp['ksic'] = kp.ksic.astype(str)
kp = kp[(kp.p.fillna('').str.len() == 8) & (kp.p < '20250101') & (kp.p >= '20100101')].copy()
assert kp.p.max() < '20250101'
docs = pd.read_parquet(C2 + r'\docs.parquet', columns=['cd', 'appl_no'])
anyc = set(docs.appl_no); corp = set(docs[docs.cd.str[0] == '1'].appl_no)
kp['linked'] = kp.appl_no.isin(anyc); kp['corp'] = kp.appl_no.isin(corp)
kp['py'] = kp.p.str[:4].astype(int); kp['fprio'] = kp.prio.fillna('').str.len().gt(0) & ~kp.prio.fillna('').str.contains('KR')
kp['corp_name'] = kp.an.fillna('').str.contains('주식회사|\(주\)|㈜|유한회사|corporation|co\.|ltd|inc', case=False, regex=True)
def smd(a, b):
    a = a.astype(float); b = b.astype(float); s = np.sqrt((a.var() + b.var()) / 2)
    return round(float((a.mean() - b.mean()) / s), 4) if s > 0 else 0.0
L1, L0 = kp[kp.linked], kp[~kp.linked]
rows = [dict(candidate='C03-C', link='KR 공개문서 ↔ 출원인코드(any)', n_linked=len(L1), n_unlinked=len(L0), link_rate=round(len(L1) / len(kp), 4),
             smd_pub_year=smd(L1.py, L0.py), smd_foreign_prio=smd(L1.fprio, L0.fprio), smd_corp_name=smd(L1.corp_name, L0.corp_name),
             tvd_ksic=round(float(0.5 * (L1.ksic.value_counts(normalize=True).sub(L0.ksic.value_counts(normalize=True), fill_value=0).abs().sum())), 4))]
# 법인 상호 문서 중 국내법인 코드 연결 여부
K = kp[kp.corp_name & ~kp.fprio]
K1, K0 = K[K.corp], K[~K.corp]
rows.append(dict(candidate='C03-C', link='법인 상호·국내 우선 문서 ↔ 국내법인 코드(1)', n_linked=len(K1), n_unlinked=len(K0), link_rate=round(len(K1) / max(len(K), 1), 4),
                 smd_pub_year=smd(K1.py, K0.py), smd_foreign_prio=None, smd_corp_name=None,
                 tvd_ksic=round(float(0.5 * (K1.ksic.value_counts(normalize=True).sub(K0.ksic.value_counts(normalize=True), fill_value=0).abs().sum())), 4)))
# 연도별 연결률
out['link_rate_by_pub_year'] = kp.groupby('py').linked.mean().round(4).to_dict()
out['corp_link_rate_by_pub_year'] = K.groupby('py').corp.mean().round(4).to_dict()
# 양성 가중 칸(KSIC×원점) 공존율: 양성 칸에 비연결 문서도 있는가(원점 창 기준)
P = pd.read_parquet(C + r'\C_dev_pos.parquet')
co = []
for t0, g in P.groupby('t0'):
    w = kp[(kp.p >= f'{t0+1}0101') & (kp.p <= f'{t0+1}0630')]
    cells = set(w[w.linked].ksic) & set(w[~w.linked].ksic)
    co.append((len(g), int(g.k.isin(cells).sum())))
out['pos_weighted_cell_coexist'] = round(sum(b for a, b in co) / sum(a for a, b in co), 4)
for r in rows: r['pos_weighted_cell_coexist'] = out['pos_weighted_cell_coexist']
pd.DataFrame(rows).to_csv(C + r'\s5_C_p3.csv', index=False, encoding='utf-8-sig')
out['p3_rows'] = rows; out['elapsed'] = round(time.time() - T0, 1)
json.dump(out, open(C + r'\s5_C_p3p4.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, default=str))
