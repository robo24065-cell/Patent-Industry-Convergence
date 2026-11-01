# -*- coding: utf-8 -*-
"""C18-T1 verify_2 r2: 권리목록과 무관한 개인 공란 검정.
제공 kr_patents에 등록번호가 있는 등록의 월별 수를 출원인 유형별로 세고, 공식 월별 등록(연보 7.2)으로 나눈 비율이
공백 달에 정상 달(2019-01~2023-10) 대비 얼마나 떨어지는지 본다. 개인이 공란에서 빠져 있다면 개인 비율은 유지돼야 한다.
또 공백 달에 개인 출원 중 공란(등록번호 없음)이면서 공개일자가 같은 시기인 문서 수 대 정상 달 비교는 하지 않는다(미등록과 구분 불가)."""
import sys, pathlib, json
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
H = B/'분석/검증루프_20261012/C18/C18-T1'; OUT = H/'verify_2'
KP = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '등록번호', '등록일자', '출원인명', '출원인주소', '출원연도'])
KP.columns = ['appl_no', 'rgst_no', 'rd', 'nm', 'addr', 'ay']
for c in KP: KP[c] = KP[c].astype(str).str.strip()
SIDO = r'^(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충청|충북|충남|전라|전북|전남|경상|경북|경남|제주)'
CORP = r'주식회사|\(주\)|㈜|유한|법인|재단|조합|대학|연구원|연구소|공사|공단|센터|협회|병원|기술원|진흥원|INC|LTD|CORP|CO\.|GMBH|LLC|AG\b|S\.A|N\.V|B\.V|PLC|KABUSHIKI|KAISHA|컴퍼니|코포레이션|인코포레이티드|리미티드|게엠베하|가부시키'
dom = KP.addr.str.match(SIDO)
KP['atype'] = np.where(~dom, '외국', np.where(KP.nm.str.upper().str.contains(CORP, regex=True), '법인', '개인'))
k = KP[KP.rgst_no.str.match(r'^10\d{11}$') & KP.rd.str.match(r'^\d{8}$')].assign(ym=lambda d: d.rd.str[:6])
k = k[k.ym >= '201901']
T = k.groupby(['ym', 'atype']).size().unstack(fill_value=0)
OFF = pd.read_csv(H/'cache/s1_official_month.csv', dtype={'ym': str}).set_index('ym').O_pat
T = T.join(OFF, how='left')
G = [f'2024{m:02d}' for m in range(1, 9)] + [f'2025{m:02d}' for m in range(2, 8)]   # 2025-08 부분월 제외
ref = [i for i in T.index if '201901' <= i <= '202310' and not np.isnan(T.loc[i, 'O_pat'])]
o = {}
for a in ['개인', '법인', '외국']:
    r = T[a] / T.O_pat
    o[a] = {'ref_median_ratio': round(float(r.loc[ref].median()), 5), 'gap_median_ratio': round(float(r.loc[G].median()), 5),
            'drop_vs_ref': round(float(1 - r.loc[G].median() / r.loc[ref].median()), 4),
            'gap_sum_over_expected': round(float(T.loc[G, a].sum() / (r.loc[ref].median() * T.loc[G, 'O_pat'].sum())), 4),
            'ref_ratio_p05_p95': [round(float(r.loc[ref].quantile(.05)), 5), round(float(r.loc[ref].quantile(.95)), 5)]}
# 2024 2분기 공식 자체 감소는 분모에 이미 반영됨
T.to_csv(OUT/'r2_monthly_by_atype.csv', encoding='utf-8-sig')
json.dump(o, open(OUT/'r2_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(o, ensure_ascii=False, indent=1))
print(T.loc['202301':'202508'].to_string())
