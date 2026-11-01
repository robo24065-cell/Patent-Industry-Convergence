# 1단계 원천 캐시: 범위안 특허(원출원인 코드) / 권리목록 특허 / 법인 목록
import pandas as pd, numpy as np, time
from common import *
t0 = time.time()
# 1) 제공범위(in-scope) KR 등록특허
kr = pd.read_parquet(KR_REG, columns=['출원번호','출원일','출원연도','등록번호','등록일자','발명명칭','메인IPC','세세분류코드'])
kr.columns = ['appl_no','appl_date','ay','rgst_no','rgst_date','title','mipc','ksic']
kr['rgst_no'] = kr.rgst_no.astype(str); kr['ksic'] = kr.ksic.astype(str)
kr['appl_date'] = kr.appl_date.astype(str); kr['rgst_date'] = kr.rgst_date.astype(str)
kr['ay'] = kr.ay.astype(int)
kr.to_parquet(CACHE/'ins_pat.parquet', index=False)
print('in-scope patents', len(kr), kr.rgst_no.nunique(), 'ksic', kr.ksic.nunique(), 'rgst_date max', kr.rgst_date.max(), time.time()-t0, flush=True)
# 2) 원출원인(applicantCd) — registry_full.applicant (권리이전과 무관한 원출원인)
app = pd.read_sql('select rgst_no, seq, applicantCd, applicantName, applicantNatl from applicant', RG())
app.to_parquet(CACHE/'ins_applicant.parquet', index=False)
print('applicant rows', len(app), 'patents', app.rgst_no.nunique(), time.time()-t0, flush=True)
# 3) 법인 목록
aq = pd.read_sql('select applicant_cd, applicant_name, n_in_fusion, total_count, status from applicant_query', BR())
aq.to_parquet(CACHE/'firms.parquet', index=False)
# 4) 권리목록 특허(rgst_no '10'), JSON 필드 추출
q = """select applicant_cd, rgst_no, rgst_date, appl_no, appl_date, title, corp_rgstno,
 json_extract(raw,'$.pertDiv') pert, json_extract(raw,'$.indvCorpTpcd') tp,
 json_extract(raw,'$.applicantInfo') ainfo, json_extract(raw,'$.rightHolderInfo') hinfo,
 json_extract(raw,'$.rgstStatus') st
 from right_list where substr(rgst_no,1,2)='10'"""
parts = []
for ch in pd.read_sql(q, BR(), chunksize=300000):
    parts.append(ch); print('chunk', len(ch), time.time()-t0, flush=True)
rl = pd.concat(parts, ignore_index=True)
rl.to_parquet(CACHE/'rl_pat.parquet', index=False)
print('right_list patents rows', len(rl), 'unique rgst', rl.rgst_no.nunique(), time.time()-t0)
