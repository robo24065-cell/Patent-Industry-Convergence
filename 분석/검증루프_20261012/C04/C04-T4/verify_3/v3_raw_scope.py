# verify_3: 원본 CSV(data.go.kr 15083277)에서 직접 form 코드·유형코드 재계산. 잠금 S1(2025-12, 2026-08)은 n_emp를 읽지 않는다(행 수·코드만).
import sys, json, re
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
RAW = r'C:\Users\PC\AppData\Local\Temp\claude\C--Users-PC-Downloads------------------------\72c91564-e815-4ac3-bfb0-ac338e5a5b96\scratchpad\nps_raw'
OUT = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C04\C04-T4\verify_3'
SN = ['2015-12','2016-12','2017-12','2018-12','2019-12','2020-12','2021-12','2022-12','2023-12','2024-12','2025-12','2026-08']
LOCK = {'2025-12','2026-08'}
res = []; k24 = None; nemp24 = None
for s in SN:
    use = [1,2,3,12,14,15] + ([] if s in LOCK else [18])
    d = pd.read_csv(RAW + rf'\nps_{s}.csv', encoding='cp949', dtype=str, usecols=use, encoding_errors='replace')
    cols = ['name','biz6','stat','form','indnm','adpt'] + ([] if s in LOCK else ['n_emp'])
    d.columns = cols
    d['form'] = d.form.str.strip(); d['stat'] = d.stat.str.strip()
    tc = d.biz6.str.strip().str[3:5]
    corp = tc.isin([str(x) for x in range(81,89)])
    r = dict(snap=s, rows=len(d), form=json.dumps(d.form.value_counts().to_dict()),
             corp_share=round(float(corp.mean()),4),
             form2_indiv=int(((d.form=='2')&~corp).sum()), form2_corp=int(((d.form=='2')&corp).sum()),
             form1_indiv=int(((d.form=='1')&~corp).sum()))
    # 개인 유형코드 세부 분포(01-79 개인과세, 80 비영리 등, 89 법인 아닌 단체, 90-99 개인면세)
    r['indiv_tc_top'] = json.dumps(tc[~corp].value_counts().head(6).to_dict())
    if s == '2024-12':
        k24 = set(d.biz6.str.strip() + '|' + d.name.str.strip())
        d['n_emp'] = pd.to_numeric(d.n_emp, errors='coerce')
        r['nemp_active_sum'] = int(d.loc[d.stat=='1','n_emp'].sum())
        r['nemp_active_indiv_tc'] = int(d.loc[(d.stat=='1')&~corp,'n_emp'].sum())
    if s in LOCK:
        f2 = d[d.form=='2']
        key = f2.biz6.str.strip() + '|' + f2.name.str.strip()
        r['form2_in_2024'] = round(float(key.isin(k24).mean()),5)
        r['form2_biz6_in_2024'] = round(float(f2.biz6.str.strip().isin({k.split('|')[0] for k in k24}).mean()),4)
        r['form2_adpt_le2024'] = round(float((f2.adpt.fillna('9999').str.strip() <= '2024-12-31').mean()),4)
        r['form2_adpt_year_top'] = json.dumps(f2.adpt.str[:4].value_counts().head(5).to_dict())
        r['form2_active_rows'] = int((f2.stat=='1').sum())
        f1c = d[(d.form=='1')]
        r['form1_rows'] = len(f1c)
    res.append(r); print(r, flush=True)
pd.DataFrame(res).to_csv(OUT + r'\v3_raw_scope.csv', index=False, encoding='utf-8-sig')
