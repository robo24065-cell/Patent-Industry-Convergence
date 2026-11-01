# s0: NPS 스냅샷 형식·범위 점검. 잠금 S1(2025-12, 2026-08)은 형식만(행 수, 컬럼, 코드 형식, n_emp 결측률·최소). n_emp 합·분위는 ≤2024-12만.
import sys, json, re
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
EXT = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext'
OUT = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C04\C04-T4\cache'
SN = ['201512', '201612', '201712', '201812', '201912', '202012', '202112', '202212', '202312', '202412', '202512', '202608']
LOCK_S1 = {'202512', '202608'}
rows = []
for s in SN:
    d = pd.read_parquet(EXT + rf'\firm_nps_wkpl_{s}.parquet')
    r = dict(snap=s, rows=len(d), cols='|'.join(d.columns), ym=';'.join(map(str, d.ym.value_counts().index[:3])))
    b = d.biz6.fillna('<NA>').astype(str)
    r['biz6_len'] = json.dumps(b.str.len().value_counts().head(5).to_dict())
    r['biz6_blank_or_000000'] = int(((b.str.strip() == '') | (b == '000000') | (b == '<NA>')).sum())
    r['biz6_nondigit'] = int((~b.str.fullmatch(r'\d{6}')).sum())
    r['form'] = json.dumps(d.form.astype(str).value_counts().to_dict())
    r['stat'] = json.dumps(d.stat.astype(str).value_counts().to_dict())
    ic = d.indcd.fillna('').astype(str)
    r['indcd_len'] = json.dumps(ic.str.len().value_counts().head(5).to_dict())
    r['indcd_blank'] = int((ic.str.strip() == '').sum()); r['indcd_000000'] = int((ic == '000000').sum())
    r['adpt_fmt'] = json.dumps(d.adpt_dt.fillna('').astype(str).str.len().value_counts().head(3).to_dict())
    r['n_emp_nan'] = round(float(d.n_emp.isna().mean()), 5)
    r['n_emp_min'] = float(d.n_emp.min())
    r['dup_key_rows'] = None
    if s not in LOCK_S1:
        r['n_emp_sum_all'] = int(d.n_emp.sum())
        r['n_emp_sum_active'] = int(d.loc[d.stat.astype(str) == '1', 'n_emp'].sum())
        r['n_emp_q'] = json.dumps({str(q): float(d.n_emp.quantile(q)) for q in (0.01, 0.1, 0.5, 0.9)})
        r['share_nemp_lt3'] = round(float((d.n_emp < 3).mean()), 4)
        r['rows_active'] = int((d.stat.astype(str) == '1').sum())
        r['rows_corp_active'] = int(((d.stat.astype(str) == '1') & (d.form.astype(str) == '1')).sum())
        r['adpt_max'] = str(d.adpt_dt.max())
        r['adpt_min'] = str(d.adpt_dt.min())
        k = d.biz6.fillna('') + '|' + d.name.fillna('')
        r['dup_key_rows'] = int(k.duplicated().sum())
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(OUT + r'\s0_nps_format.csv', index=False, encoding='utf-8-sig')
