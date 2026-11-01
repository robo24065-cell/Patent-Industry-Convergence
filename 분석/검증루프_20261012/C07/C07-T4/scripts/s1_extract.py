# registry_full(ro) → cache parquet. 사건 표는 날짜 < 2025-01-01만(잠금 창 2025 사건은 추출하지 않음).
import sqlite3, sys, time
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T4\scripts')
from lib7 import *
con = sqlite3.connect('file:' + DB + '?mode=ro', uri=True)
t = time.time()
def save(name, sql, dcol=None):
    df = pd.read_sql(sql, con)
    if dcol:
        assert (df[dcol].dropna() < EVENT_CUT).all()
    df.to_parquet(C + rf'\{name}.parquet', index=False); print(name, df.shape, round(time.time() - t, 1), flush=True)
save('pat', "select rgst_no, ksic, applDate, applNo, claimCount, cndrtExptnDate, ipcCd, opnDate, pubDate, rgstDate from patent")
save('own', f"select rgst_no, seq, ownerName, ownerNatl, ownerRgstCsDate d, ownerRgstCsName nm, ownerRgstCsReason rs from owner where ownerRgstCsDate < '{EVENT_CUT}'", 'd')
save('rgt', f"select rgst_no, seq, rgstCsDate d, rgstCsName nm from right where rgstCsDate < '{EVENT_CUT}'", 'd')
save('pay', f"select rgst_no, lastAnnl, payDate d from pay where payDate < '{EVENT_CUT}'", 'd')
save('lic', f"select rgst_no, elsLcnsStartDate d, 1 ex from elsLcnsList where elsLcnsStartDate < '{EVENT_CUT}' union all select rgst_no, noelsLcnsStartDate d, 0 ex from noelsLcnsList where noelsLcnsStartDate < '{EVENT_CUT}'", 'd')
save('app', "select rgst_no, seq, applicantCd, applicantName, applicantNatl from applicant")
save('inv', "select rgst_no, count(*) n_inv, max(inventorNatl<>'대한민국') inv_for from inventor group by rgst_no")
save('cof', "select rgst_no, count(*) n_prio, max(cofprNatl<>'대한민국') for_prio from cofpr where cofprNatl is not null group by rgst_no")
save('agt', "select rgst_no, max(applAgentName is not null) has_agent from applAgent group by rgst_no")
