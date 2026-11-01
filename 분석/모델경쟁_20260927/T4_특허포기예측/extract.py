# 원본 registry_full.sqlite(읽기전용) -> 트랙 폴더 cache/*.parquet
import sqlite3, time, pandas as pd, os
DB=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'cache')
con=sqlite3.connect(f"file:{DB}?mode=ro",uri=True)
Q={
 'patent':"select rgst_no,ksic,applDate,applNo,applTpcd,claimCount,cndrtExptnDate,cpcCd,ipcCd,lastDspst,opnDate,pubDate,rfoexDate,rfoexYn,rgstDate,title,engTitle from patent",
 'pay':"select rgst_no,seq,statAnnl,lastAnnl,payAmount,payDate from pay",
 'right':"select rgst_no,seq,rgstCsDate,rgstCsName,rgstCsReason from right",
 'owner':"select rgst_no,seq,ownerName,ownerNatl,ownerRgstCsDate,ownerRgstCsName,ownerRgstCsReason from owner",
 'applicant':"select rgst_no,seq,applicantCd,applicantName,applicantNatl,rpstrYn from applicant",
 'inventor_agg':"select rgst_no,count(inventorName) n_inv, sum(case when inventorNatl is not null and inventorNatl<>'대한민국' then 1 else 0 end) n_inv_foreign from inventor group by rgst_no",
 'cofpr':"select rgst_no,seq,cofprApplno,cofprDate,cofprNatl from cofpr where cofprNatl is not null or cofprDate is not null",
 'els':"select rgst_no,seq,elsLcnsStartDate,elsLcnsEndDate,elsLcnsRgstCsNm from elsLcnsList where elsLcnsPsnNm is not null",
 'noels':"select rgst_no,seq,noelsLcnsStartDate,noelsLcnsEndDate,noelsLcnsRgstCsNm from noelsLcnsList where noelsLcnsPsnNm is not null",
 'agent':"select rgst_no,count(applAgentName) n_agent, min(applAgentName) agent1 from applAgent group by rgst_no",
}
for k,s in Q.items():
    t=time.time()
    df=pd.read_sql_query(s,con)
    df.to_parquet(os.path.join(OUT,k+'.parquet'),index=False)
    print(k,len(df),round(time.time()-t,1),flush=True)
