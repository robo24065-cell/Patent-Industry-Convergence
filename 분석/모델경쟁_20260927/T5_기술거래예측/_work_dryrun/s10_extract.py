# 원본 sqlite(읽기전용)에서 필요한 테이블을 추출해 cache/*.parquet 로 저장.
# 🔒 홀드아웃 보호: 모든 사건(권리자 변동·등록사건·납부·실시권)은 날짜 < 2019-01-01 만 추출한다.
import sqlite3,sys,time,pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
DB=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
con=sqlite3.connect('file:'+DB+'?mode=ro',uri=True)
import os
CUT=os.environ.get('T5_CUT','2019-01-01').replace('-','')
t=time.time()
def save(name,sql):
    df=pd.read_sql(sql,con); df.to_parquet(f'cache/{name}.parquet',index=False); print(name,df.shape,round(time.time()-t,1),flush=True)
save('pat',"select rgst_no,ksic,applDate,applNo,applTpcd,claimCount,cndrtExptnDate,cpcCd,ipcCd,lastDspst,opnDate,pubDate,rfoexYn,rgstDate,title,engTitle from patent")
save('own',f"select rgst_no,seq,ownerName,ownerNatl,ownerRgstCsDate d,ownerRgstCsName rs,ownerRgstCsReason rr from owner where ownerRgstCsDate < '{CUT}'")
save('rgt',f"select rgst_no,seq,rgstCsDate d,rgstCsName nm,rgstCsReason rs from right where rgstCsDate < '{CUT}'")
save('pay',f"select rgst_no,seq,lastAnnl,statAnnl,payAmount,payDate d from pay where payDate < '{CUT}'")
save('lic_e',f"select rgst_no,elsLcnsStartDate d,elsLcnsEndDate e,elsLcnsPsnNm nm,elsLcnsRgstCsNm cs from elsLcnsList where elsLcnsStartDate is not null and elsLcnsStartDate < '{CUT}'")
save('lic_n',f"select rgst_no,noelsLcnsStartDate d,noelsLcnsEndDate e,noelsLcnsPsnNm nm,noelsLcnsRgstCsNm cs from noelsLcnsList where noelsLcnsStartDate is not null and noelsLcnsStartDate < '{CUT}'")
save('app',"select rgst_no,seq,applicantCd,applicantName,applicantNatl,rpstrYn from applicant")
save('inv',"select rgst_no,count(*) n_inv,sum(inventorNatl<>'대한민국') n_inv_for,group_concat(inventorCd,'|') inv_cds from inventor where inventorCd is not null group by rgst_no")
save('cof',"select rgst_no,count(*) n_prio,group_concat(cofprNatl,'|') prio_natl from cofpr where cofprNatl is not null group by rgst_no")
save('agt',"select rgst_no,max(applAgentName is not null) has_agent from applAgent group by rgst_no")
