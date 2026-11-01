# s8: 캐시 → 산출 CSV(bias_checks.csv, essentiality_dev.csv) + P1 보조 수치.
import sys, json
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C04\C04-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from libA import *
u = pd.read_parquet(C + r'\A_units_dev.parquet')
p1 = dict(A_origin_pos=u.groupby('s0').yA.sum().astype(int).to_dict(), A_origin_sum=int(u.yA.sum()),
          A_nonoverlap=int(u[u.s0.isin(NONOVER_A)].yA.sum()), A_unique_firms=int(u[u.yA == 1].key.nunique()),
          C_origin_pos=u.groupby('s0').yC.sum().astype(int).to_dict(), C_nonoverlap=int(u[u.s0.isin(NONOVER_A)].yC.sum()),
          C_unique_firms=int(u[u.yC == 1].key.nunique()))
print(p1); json.dump(p1, open(C + r'\s8_p1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
b = pd.concat([pd.read_csv(C + r'\s2_bias_A.csv'), pd.read_csv(C + r'\s5_p2_markers.csv'), pd.read_csv(C + r'\s6_B_p3p5.csv').query("check=='P3'")], ignore_index=True)
f0 = pd.read_csv(C + r'\s0b_nps_scope.csv')
for _, r in f0.iterrows():
    b.loc[len(b)] = dict(candidate='공통(NPS)', check='P3/P4a 범위', metric=f"{r.snap} 법인 유형코드 행/개인 유형코드 행",
                         value=f"{r.typecode_corp_rows}/{r.typecode_indiv_rows}", note=f"기준월 후 적용일 {r.adpt_after_ref}행; form×유형 {r.form_x_type}")
b.to_csv(W + r'\bias_checks.csv', index=False, encoding='utf-8-sig')
e1 = pd.read_csv(C + r'\s3_p5_A_rows.csv'); e1.insert(0, 'candidate', 'A/C')
e2 = pd.read_csv(C + r'\s3_p5_A_boot.csv'); e2.insert(0, 'candidate', 'A/C'); e2['s0'] = 'BOOT(AUC_prov−AUC_pub)'
e3 = pd.DataFrame([dict(candidate='B', assign=k, s0='POOLED', **v) for k, v in json.load(open(C + r'\s6_B_p5.json', encoding='utf-8')).items()])
e3 = e3.rename(columns={'J_pub': 'pooled_jac'})
pd.concat([e1, e2, e3], ignore_index=True).to_csv(W + r'\essentiality_dev.csv', index=False, encoding='utf-8-sig')
print('ok')
