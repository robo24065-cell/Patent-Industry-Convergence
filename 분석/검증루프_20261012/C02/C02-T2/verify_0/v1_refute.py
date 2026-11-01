# C02-T2 verify_0: 반박 검증 — KR 2024 등록 결손 분해 주장 재계산 (읽기 전용, 원천 sqlite/parquet 직접)
import os, sys, json, sqlite3
import duckdb, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
W = os.path.join(ROOT, r"분석\검증루프_20261012\C02\C02-T2")
OUT = os.path.join(W, 'verify_0')
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
REG = os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite").replace("\\", "/")
BR = os.path.join(ROOT, r"수집파이프라인\business_rights.sqlite").replace("\\", "/")
res = {}
rc = sqlite3.connect(f'file:{REG}?mode=ro', uri=True)
r = pd.read_sql("""select p.rgst_no, substr(p.rgstDate,1,4) y, substr(p.applDate,1,4) ay, p.ksic,
                          a.applicantNatl natl_raw
                   from patent p left join (select rgst_no, applicantNatl, min(seq) s from applicant group by rgst_no) a on a.rgst_no=p.rgst_no
                   where substr(p.rgst_no,1,2)='10' and p.rgstDate>='2019'""", rc)
rc.close()
r['ay'] = pd.to_numeric(r.ay, errors='coerce')
r['natl'] = np.where(r.natl_raw.isna(), '결측', np.where(r.natl_raw == '대한민국', '내국', '외국'))
res['n_by_year'] = r.groupby('y').size().to_dict()

def dec(df, y0, y1, catfun):
    a = df[df.y == y0]; b = df[df.y == y1]
    ca = catfun(a, int(y0)).value_counts(); cb = catfun(b, int(y0)).value_counts()
    t = pd.DataFrame({'n0': ca, 'n1': cb}).fillna(0); t['delta'] = t.n1 - t.n0
    t['contrib'] = t.delta / t.delta.sum(); t['rate'] = t.n1 / t.n0 - 1
    return t

# 1) 원주장 재현: 고정 컷오프 ≤2019 (2023→2024)
fixed = lambda d, y0: pd.Series(np.where(d.ay <= 2019, 'old', 'new'), index=d.index)
t = dec(r, '2023', '2024', fixed); res['claim_repro_fixed2019'] = t.round(4).to_dict('index')
# 2) 기준선: 같은 구조(컷오프 = y0−4)로 정상 연도 쌍
bench = {}
for y0 in ['2019', '2020', '2021', '2022', '2023']:
    y1 = str(int(y0) + 1)
    f = lambda d, yy: pd.Series(np.where(d.ay <= yy - 4, 'old', 'new'), index=d.index)
    tt = dec(r, y0, y1, f)
    bench[f'{y0}->{y1}'] = dict(old_rate=float(tt.loc['old', 'rate']), new_rate=float(tt.loc['new', 'rate']),
                                old_delta=float(tt.loc['old', 'delta']), new_delta=float(tt.loc['new', 'delta']),
                                total_delta=float(tt.delta.sum()))
res['bench_moving_cutoff_y0minus4'] = bench
# 3) 같은 경과연수 대역(등록연도−출원연도 ≥5 vs ≤4)으로 분해 — 컷오프 이동 인공물 제거
age = lambda d, yy: pd.Series(np.where((d.y.astype(int) - d.ay) >= 5, 'age>=5', 'age<=4'), index=d.index)
ageb = {}
for y0 in ['2019', '2020', '2021', '2022', '2023']:
    y1 = str(int(y0) + 1); tt = dec(r, y0, y1, age)
    ageb[f'{y0}->{y1}'] = tt.round(4).to_dict('index')
res['age_band_decomp'] = ageb
# 4) 반사실: 2022→23 의 old/new 변화율을 2023 에 적용했을 때의 '정상' 2024, 그 대비 초과감소의 old 기여
b = bench['2022->2023']; t0 = res['claim_repro_fixed2019']
exp_old = t0['old']['n0'] * (1 + b['old_rate']); exp_new = t0['new']['n0'] * (1 + b['new_rate'])
ex_old = t0['old']['n1'] - exp_old; ex_new = t0['new']['n1'] - exp_new
res['excess_vs_2022_23_pattern'] = dict(exp_old=exp_old, exp_new=exp_new, excess_old=ex_old, excess_new=ex_new,
                                        old_share_of_excess=ex_old / (ex_old + ex_new))
# 5) 내/외국 vs 공식(KOSIS A046 원파일)
k = json.load(open(os.path.join(W, r'download\kosis\data_TX_13801_A046.json'), encoding='utf-8'))
kd = pd.DataFrame(k); kd = kd[kd.C2_NM == '특허'][['C1_NM', 'PRD_DE', 'DT']]; kd['DT'] = pd.to_numeric(kd.DT)
off = kd.pivot_table(index='PRD_DE', columns='C1_NM', values='DT')
res['kosis_A046_patent'] = off.loc['2019':'2025'].to_dict()
nt = r[r.y.isin(['2022', '2023', '2024'])].groupby(['y', 'natl']).size().unstack()
res['registry_natl'] = nt.to_dict()
for nm in ['내국', '외국']:
    offnm = nm + '인'
    res[f'{nm}_ratio_prov_off'] = {y: float(nt.loc[y, nm] / off.loc[y, offnm]) for y in ['2022', '2023', '2024']}
    res[f'{nm}_rate_prov_23_24'] = float(nt.loc['2024', nm] / nt.loc['2023', nm] - 1)
    res[f'{nm}_rate_off_23_24'] = float(off.loc['2024', offnm] / off.loc['2023', offnm] - 1)
# 6) KSIC: 감소율 분포, 정상 연도(2022→23) 대비 분산
def krate(y0, y1):
    a = r[r.y == y0].ksic.value_counts(); bb = r[r.y == y1].ksic.value_counts()
    tt = pd.DataFrame({'a': a, 'b': bb}).fillna(0); tt['rate'] = tt.b / tt.a - 1; return tt
k34 = krate('2023', '2024'); k23 = krate('2022', '2023'); k12 = krate('2021', '2022')
tot34 = k34.b.sum() / k34.a.sum() - 1
# 이항 잡음 하 기대 분산: 각 KSIC 감소율이 전체율과 같다면 sd ≈ sqrt(p(1-p)/n) 근사 (n2024 / (n2023) 비 비율)
def overdisp(tt):
    p = tt.b.sum() / (tt.a.sum() + tt.b.sum())
    share = tt.b / (tt.a + tt.b); n = tt.a + tt.b
    chi = (((share - p) ** 2) / (p * (1 - p) / n)).sum(); return float(chi), int(len(tt) - 1)
res['ksic'] = dict(n_ksic=int(len(k34)), n_ksic_nonnull=int(k34.index.notna().sum()),
                   rate_min=float(k34.rate.min()), rate_max=float(k34.rate.max()), rate_median=float(k34.rate.median()),
                   n_lt_m10=int((k34.rate < -0.10).sum()), total_rate=float(tot34),
                   sd_rate_23_24=float(k34.rate.std()), sd_rate_22_23=float(k23.rate.std()), sd_rate_21_22=float(k12.rate.std()),
                   sd_rel_23_24=float((k34.rate - tot34).std()),
                   chi2_23_24=overdisp(k34), chi2_22_23=overdisp(k23), chi2_21_22=overdisp(k12),
                   corr_rate_2324_vs_2223=float(k34.rate.corr(k23.rate, method='spearman')))
# 7) BR 확인 누락분: 구출원 비중·해외우선권 비중 직접 재계산
con = duckdb.connect(); con.sql("set threads=2"); con.sql("set memory_limit='3GB'"); con.sql("install sqlite; load sqlite")
con.sql(f"""create temp table kr as select appl_no, any_value(rgst_no) rgst_no, any_value(appl_year) ay, any_value(ksic) ksic,
              max(case when prio_cc is not null and regexp_replace(replace(prio_cc,'KR',''),'\\|','','g')<>'' then 1 else 0 end) prio_for
            from read_parquet('{META}') where cc='KR' and substr(appl_no,1,2)='10' group by appl_no""")
con.sql(f"""create temp table br as select appl_no, min(rgst_date) rd from sqlite_scan('{BR}', 'right_list') where substr(rgst_no,1,2)='10' group by appl_no""")
ms = con.sql("""select left(b.rd,4) y, k.ay, k.ksic, k.prio_for from kr k join br b using(appl_no)
                where (k.rgst_no is null or k.rgst_no='') and b.rd<='20250822'""").df()
# 대조: 같은 BR 이 덮는 '제공 2024 등록' 의 구출원 비중(커버 편향 제거)
pres = con.sql("""select left(b.rd,4) y, k.ay, k.prio_for from kr k join br b using(appl_no)
                  where substr(k.rgst_no,1,2)='10' and left(b.rd,4) in ('2023','2024')""").df()
con.close()
m24 = ms[ms.y == '2024']
res['br_missing_2024'] = dict(n=int(len(m24)), share_old=float((m24.ay <= 2019).mean()), share_prio=float(m24.prio_for.mean()),
                              n_old=int((m24.ay <= 2019).sum()))
p24 = pres[pres.y == '2024']
res['br_present_2024_br_covered'] = dict(n=int(len(p24)), share_old=float((p24.ay <= 2019).mean()), share_prio=float(p24.prio_for.mean()))
res['present_2024_all_share_old'] = float((r[r.y == '2024'].ay <= 2019).mean())
res['present_2024_all_share_prio_claim'] = 6756 / 26766
# 결손 직접 추정의 구출원 기여: missing_old / missing_total
json.dump(res, open(os.path.join(OUT, 'v1_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
