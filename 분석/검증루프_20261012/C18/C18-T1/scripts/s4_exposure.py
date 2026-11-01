# -*- coding: utf-8 -*-
"""C18-T1 s4: 트랙 노출 — 공백 달(G) 누락 등록(권리목록 확인, 등록번호 공란)이 각 공식 트랙의 학습·최종 창에 몇 건 걸리는가.
원천 직접 계산(kr_patents, business_rights.right_list, registry_full). 읽기 전용. 쓰기: C18-T1/cache/s4_*."""
import sys, pathlib, json, sqlite3, time
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
t0 = time.time()
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
H = B/'분석/검증루프_20261012/C18/C18-T1'; C = H/'cache'
ro = lambda p: sqlite3.connect(pathlib.Path(p).as_uri() + '?mode=ro', uri=True)
END = '20250822'
T = pd.read_csv(H/'monthly_capture.csv', dtype={'ym': str}).set_index('ym'); G = set(T.index[T.gap])
KP = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '출원연도', '등록번호', '등록일자', '공개일자', '세세분류코드'])
KP.columns = ['appl_no', 'ay', 'rgst_no', 'rd', 'pub', 'ksic']
for c in ['appl_no', 'rgst_no', 'rd', 'pub']: KP[c] = KP[c].astype(str).str.strip()
KP['ay'] = pd.to_numeric(KP.ay, errors='coerce')
KP['blank'] = ~KP.rgst_no.str.match(r'^\d{13}$'); KP['pub_ok'] = KP.pub.str.match(r'^\d{8}$')
bc = ro(B/'수집파이프라인/business_rights.sqlite')
RL = pd.read_sql("select distinct rgst_no, rgst_date rd_rl, appl_no from right_list where substr(rgst_no,1,2)='10'", bc); bc.close()
RL = RL[RL.appl_no.isin(set(KP.appl_no)) & (RL.rd_rl <= END)].sort_values('rd_rl').drop_duplicates('appl_no')
rc = ro(B/'수집파이프라인/registry_full.sqlite'); rf = set(r[0] for r in rc.execute("select rgst_no from patent")); rc.close()
M = KP[KP.blank].merge(RL, on='appl_no', how='inner')
M['in_rf'] = M.rgst_no_y.isin(rf); M['ym'] = M.rd_rl.str[:6]; M['inG'] = M.ym.isin(G)
o = {'missing_docs_all': int(len(M)), 'missing_in_rf': int(M.in_rf.sum()), 'missing_inG': int(M.inG.sum()),
     'missing_min_rd': M.rd_rl.min(), 'missing_ym_hist': M.ym.value_counts().sort_index().to_dict(),
     'missing_pub_blank': int((~M.pub_ok).sum()), 'missing_pub_blank_inG': int(((~M.pub_ok) & M.inG).sum())}
def cnt(mask, note=''):
    s = M[mask]; return dict(docs=int(len(s)), docs_inG=int(s.inG.sum()), docs_pub_blank=int((~s.pub_ok).sum()), docs_pub_blank_inG=int(((~s.pub_ok) & s.inG).sum()))
E = {}
# T1: 학습 원점 2012–15(레이블 cap 2019-12-31, 피처 컷 2019-12-31), 최종 출원 2020–23·등록 ≤ 2025-08-22
E['T1_train'] = cnt(M.rd_rl <= '20191231')
E['T1_final'] = cnt(M.ay.between(2020, 2023)); E['T1_final']['denom_kp_appl_2020_23'] = int(KP.ay.between(2020, 2023).sum())
E['T1_final']['denom_kp_reg_2020_23'] = int((KP.ay.between(2020, 2023) & ~KP.blank).sum())
# T3: 공개연도 = least(pub, rgst). 최종 레이블 KR 공개 2022–24, 피처 ≤2021. 공개일이 없고 등록일이 공란이면 문서 자체가 사라짐
E['T3_train'] = cnt((~M.pub_ok) & (M.rd_rl <= '20211231'))
E['T3_final'] = cnt((~M.pub_ok) & M.rd_rl.between('20220101', '20241231'))
kp_pubyear = np.where(KP.pub_ok, KP.pub.str[:4], np.where(~KP.blank & KP.rd.str.match(r'^\d{8}$'), KP.rd.str[:4], ''))
E['T3_final']['denom_kp_pubyear_2022_24'] = int(pd.Series(kp_pubyear).isin(['2022', '2023', '2024']).sum())
# T4: 기준일 2022-01-01 존속 등록특허, 결과 2022–24 불납. 모집단 등록 < 2022 → 공백(2023-11~) 무관
E['T4_final'] = cnt(M.rd_rl < '20220101')
# N4: t0=2023 피처(≤2023-12-31), 결과 국세청 폐업 2024–25(등록원부 무관)
E['N4_final_feat'] = cnt(M.rd_rl <= '20231231')
# N5: t0=2022 피처(rgd ≤ 2022-12-31), 결과 ETSI(등록원부 무관)
E['N5_final'] = cnt(M.rd_rl <= '20221231')
# N6: t0=2021, 결과 첫 출원 2022–23(known = min(pub, rgst)). 공개 없음+등록 공란 → known 결측
E['N6_final'] = cnt(M.ay.between(2022, 2023) & ~M.pub_ok); E['N6_final']['denom_kp_appl_2022_23'] = int(KP.ay.between(2022, 2023).sum())
# N7: t0=2022 모집단(등록 ≥1, ≤2022-12-31), 결과 NTIS 2023–24
E['N7_final'] = cnt(M.rd_rl <= '20221231')
# N8: t0=2022 존속 등록(≤2022-12-31), 결과 심판 예고등록 2023–24(모집단 특허의 사건)
E['N8_final'] = cnt(M.rd_rl <= '20221231')
# N9: 과제 t=2022, 결과 출원 2022–23(NTIS 성과 연결, 출원연도 기준)
E['N9_final'] = cnt(M.ay.between(2022, 2023)); E['N9_final']['denom_kp_appl_2022_23'] = int(KP.ay.between(2022, 2023).sum())
o['exposure'] = E
# 등록번호 공란 전체(권리목록 미확인 포함) — 출원연도·공개 유무별
o['kp_blank_by_ay'] = KP[KP.blank].groupby('ay').size().loc[2015:2024].to_dict()
o['kp_blank_pubblank_by_ay'] = KP[KP.blank & ~KP.pub_ok].groupby('ay').size().loc[2015:2024].to_dict()
M[['appl_no', 'ay', 'rgst_no_y', 'rd_rl', 'ym', 'inG', 'pub_ok', 'ksic', 'in_rf']].rename(columns={'rgst_no_y': 'rgst_no'}).to_parquet(C/'s4_missing_docs.parquet', index=False)
o['secs'] = round(time.time() - t0)
json.dump(o, open(C/'s4_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(E, ensure_ascii=False)); print('done', o['secs'])
