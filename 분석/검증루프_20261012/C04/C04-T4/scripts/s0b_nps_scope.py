# s0b: NPS 스냅샷 범위·시점 점검(형식 수준). n_emp는 쓰지 않는다(잠금 S1 포함 전 스냅샷 공통).
# (1) 기준월 이후 적용일(adpt_dt) 행 수 = 스냅샷이 순수 시점 자료인지
# (2) 사업자번호 4–5자리(유형코드) 81–88 = 법인, 그 외 = 개인 → 스냅샷별 법인/개인 행 수, form 코드와의 교차
# (3) 2025-12·2026-08의 form=2 행이 2024-12 파일에 있었는지(이름+biz6 기준, 행 존재만)
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
EXT = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext'
OUT = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C04\C04-T4\cache'
SN = ['201512', '201612', '201712', '201812', '201912', '202012', '202112', '202212', '202312', '202412', '202512', '202608']
REF = {'201512': '2015-11-30', '201612': '2016-11-30', '201712': '2017-11-30', '201812': '2018-11-30', '201912': '2019-11-30',
       '202012': '2020-12-31', '202112': '2021-12-31', '202212': '2022-12-31', '202312': '2023-12-31', '202412': '2024-12-31',
       '202512': '2025-12-31', '202608': '2026-08-31'}
rows = []
keys24 = None
for s in SN:
    d = pd.read_parquet(EXT + rf'\firm_nps_wkpl_{s}.parquet', columns=['name', 'biz6', 'stat', 'form', 'adpt_dt', 'readpt_dt', 'rrg_dt'])
    tc = d.biz6.astype(str).str[3:5]
    corp = tc.isin([str(x) for x in range(81, 89)])
    r = dict(snap=s, rows=len(d), ref_end=REF[s])
    a = d.adpt_dt.fillna('').astype(str)
    r['adpt_after_ref'] = int((a > REF[s]).sum())
    r['adpt_after_ref_share'] = round(float((a > REF[s]).mean()), 5)
    r['adpt_after_ref_active'] = int(((a > REF[s]) & (d.stat.astype(str) == '1')).sum())
    rr = d.readpt_dt.fillna('').astype(str); r['readpt_after_ref'] = int((rr > REF[s]).sum())
    rg = d.rrg_dt.fillna('').astype(str); r['rrg_after_ref'] = int((rg > REF[s]).sum())
    r['typecode_corp_rows'] = int(corp.sum()); r['typecode_indiv_rows'] = int((~corp).sum())
    r['typecode_corp_share'] = round(float(corp.mean()), 4)
    r['form_x_type'] = json.dumps(pd.crosstab(d.form.astype(str), corp).to_dict())
    if s == '202412':
        keys24 = set(d.biz6.astype(str) + '|' + d.name.astype(str))
    if s in ('202512', '202608') and keys24 is not None:
        f2 = d[d.form.astype(str) == '2']
        k2 = f2.biz6.astype(str) + '|' + f2.name.astype(str)
        r['form2_rows'] = len(f2)
        r['form2_in_202412_file'] = round(float(k2.isin(keys24).mean()), 4)
        r['form2_adpt_le_2024'] = round(float((f2.adpt_dt.fillna('9999').astype(str) <= '2024-12-31').mean()), 4)
        r['form2_corp_suffix_name'] = round(float(f2.name.astype(str).str.contains(r'주식회사|\(주\)|㈜|유한회사|법인').mean()), 4)
        f1 = d[d.form.astype(str) == '1']; k1 = f1.biz6.astype(str) + '|' + f1.name.astype(str)
        r['form1_in_202412_file'] = round(float(k1.isin(keys24).mean()), 4)
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(OUT + r'\s0b_nps_scope.csv', index=False, encoding='utf-8-sig')
