# C02-T2 verify_0 v2: KSIC별 결손률을 BR 커버 안에서만(커버율 차이 중립) 재계산 + 잡음 대비 이질성
import os, sys, json
import duckdb, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C02\C02-T2\verify_0")
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
BR = os.path.join(ROOT, r"수집파이프라인\business_rights.sqlite").replace("\\", "/")
con = duckdb.connect(); con.sql("set threads=2"); con.sql("set memory_limit='3GB'"); con.sql("install sqlite; load sqlite")
con.sql(f"""create temp table kr as select appl_no, any_value(rgst_no) rgst_no, any_value(rgst_date) rgst_date, any_value(appl_year) ay, any_value(ksic) ksic
            from read_parquet('{META}') where cc='KR' and substr(appl_no,1,2)='10' group by appl_no""")
con.sql(f"""create temp table br as select appl_no, min(rgst_date) rd from sqlite_scan('{BR}', 'right_list') where substr(rgst_no,1,2)='10' group by appl_no""")
d = con.sql("""select k.ksic, left(b.rd,4) by_, (k.rgst_no is null or k.rgst_no='') miss, left(k.rgst_date,4) py, (b.appl_no is not null) inbr
               from kr k left join br b using(appl_no)
               where left(b.rd,4)='2024' or left(k.rgst_date,4)='2024'""").df()
con.close()
# 제공 2024 등록 중 BR 커버율 (KSIC별)
pres = d[(~d.miss) & (d.py == '2024')]
cov = pres.groupby('ksic').inbr.mean()
# BR 2024 등록 안에서 결손률 (커버 중립)
inb = d[d.inbr & (d.by_ == '2024')]
mr = inb.groupby('ksic').miss.agg(['mean', 'size', 'sum'])
p = inb.miss.mean()
chi = float((((mr['mean'] - p) ** 2) / (p * (1 - p) / mr['size'])).sum())
out = dict(overall_missing_rate_in_br=float(p), n_br2024=int(len(inb)),
           ksic_cov_min=float(cov.min()), ksic_cov_median=float(cov.median()), ksic_cov_max=float(cov.max()),
           ksic_missrate_in_br_min=float(mr['mean'].min()), ksic_missrate_in_br_median=float(mr['mean'].median()),
           ksic_missrate_in_br_max=float(mr['mean'].max()), chi2=chi, df=int(len(mr) - 1),
           n_ksic_missrate_dev_gt_5pp=int(((mr['mean'] - p).abs() > 0.05).sum()),
           corr_cov_vs_missrate=float(pd.concat([cov, mr['mean']], axis=1).corr(method='spearman').iloc[0, 1]))
mr.assign(cov=cov).to_csv(os.path.join(OUT, 'v2_ksic_missrate_in_br.csv'))
json.dump(out, open(os.path.join(OUT, 'v2_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
