# OECD SDMX 공개 API(키 불필요) — 특허 통계 DF_PATENTS_WIPO (WIPO 35 기술분야 x 국가 x 연도)
import requests, io, time, pandas as pd
H={"Accept":"application/vnd.sdmx.data+csv;version=1.0.0"}
areas="KOR+JPN+USA+CHN+DEU+EU27_2020+W+OECD+TWN+GBR+FRA"
out=[]
for flow in ["DF_PATENTS_WIPO"]:
    for auth in ["9P50_2","9P50_3","6F0","US50","9P50_1"]:
        url=f"https://sdmx.oecd.org/public/rest/data/OECD.STI.PIE,DSD_PATENTS@{flow},1.0/{auth}.A....{areas}......?startPeriod=1995"
        for k in range(4):
            r=requests.get(url,headers=H,timeout=300)
            if r.status_code==200: break
            print(auth,r.status_code); time.sleep(10*(k+1))
        d=pd.read_csv(io.StringIO(r.text)); print(flow,auth,len(d),d.TIME_PERIOD.min(),d.TIME_PERIOD.max()); out.append(d)
        time.sleep(3)
d=pd.concat(out)
d=d.drop(columns=[c for c in d.columns if d[c].nunique()<=1 and c not in('FREQ',)],errors='ignore')
d.to_parquet("quality_oecd_patents_wipo35.parquet",index=False); print(d.shape, d.columns.tolist())
