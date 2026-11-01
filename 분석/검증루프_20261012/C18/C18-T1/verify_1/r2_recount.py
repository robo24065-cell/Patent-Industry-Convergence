# -*- coding: utf-8 -*-
# 독립 재계산: sqlite(mode=ro)+pyarrow+dict/numpy. pandas 미사용.
import sys, pathlib, sqlite3, json, time, math
sys.dont_write_bytecode=True
import numpy as np, pyarrow.parquet as pq
from collections import defaultdict, Counter
t0=time.time()
B=pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
OUT=B/'분석/검증루프_20261012/C18/C18-T1/verify_1'
ro=lambda p: sqlite3.connect(pathlib.Path(p).as_uri()+'?mode=ro', uri=True)
# official from xlsx (already verified printout) — re-read via openpyxl
import openpyxl
X=B/'분석/검증루프_20261012/C13/C13-T1/download/moip_unzip'
O={}
for ed,f,yr in [('f_312_2','013_7.기타 출원등록_r1.xlsx',2019),('f_328_2','013_7-기타 출원등록.xlsx',2020),('f_360_1','013_7-기타 출원등록.xlsx',2021),('f_380_2','013_7.기타 출원등록v2.xlsx',2022),('f_400_2','013_7.기타 출원등록.xlsx',2023),('f_416_2','013_7.기타 출원등록.xlsx',2024),('f_437_1','013_[2025년 통계표] 2025 지식재산통계연보(26년 발행)__7.기타 출원등록.xlsx',2025)]:
    wb=openpyxl.load_workbook(X/ed/f, read_only=True, data_only=True)
    ws=[wb[s] for s in wb.sheetnames if s.strip().startswith('7. 2.')][0]
    for r in ws.iter_rows(values_only=True):
        if r and isinstance(r[0],str):
            s=r[0].strip()
            if s.endswith('월') and s[:-1].isdigit():
                vals=[c for c in r[1:] if isinstance(c,(int,float))]
                O['%d%02d'%(yr,int(s[:-1]))]=int(vals[0])
res={'official_months':len(O)}
# RF
rc=ro(B/'수집파이프라인/registry_full.sqlite')
RF={}
for rg,rd in rc.execute("select rgst_no, rgstDate from patent where substr(rgst_no,1,2)='10'"):
    RF[rg]=rd
rc.close()
res['RF_n']=len(RF)
# kr_patents appl set
t=pq.read_table(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호'])
P=set(str(x).strip() for x in t.column(0).to_pylist()); del t
res['P_n']=len(P)
# right_list
bc=ro(B/'수집파이프라인/business_rights.sqlite')
RLin={}; seqmax=defaultdict(int); allseq={}
for rg,rd,ap in bc.execute("select rgst_no, rgst_date, appl_no from right_list where substr(rgst_no,1,2)='10'"):
    if rd and len(rd)>=8 and rd[:8].isdigit():
        allseq[rg]=rd[:6]
    if ap in P and rg not in RLin: RLin[rg]=rd
bc.close()
for rg,rd in RF.items():
    if rd and rd[:8].isdigit(): allseq.setdefault(rg,rd[:6])
for rg,ym in allseq.items():
    s=int(rg[2:9])
    if s>seqmax[ym]: seqmax[ym]=s
yms=sorted(seqmax)
ostar={yms[i]:seqmax[yms[i]]-seqmax[yms[i-1]] for i in range(1,len(yms))}
# monthly counts
rfm=Counter(rd[:6] for rd in RF.values() if rd)
union={}
for rg,rd in RF.items(): union[rg]=rd[:6]
for rg,rd in RLin.items():
    if rg not in union and rd: union[rg]=rd[:6]
um=Counter(v for v in union.values())
miss=Counter(rd[:6] for rg,rd in RLin.items() if rg not in RF and rd and rd[:8]<='20250822')
res['missing_total_le20250822']=sum(miss.values())
mon={}
for ym in sorted(O):
    if ym>'202508': continue
    mon[ym]=dict(O=O[ym],ostar=ostar.get(ym),RF=rfm.get(ym,0),U=um.get(ym,0),miss=miss.get(ym,0),
                 R=rfm.get(ym,0)/O[ym], UoverO=um.get(ym,0)/O[ym])
ref=[mon[m]['R'] for m in mon if '201901'<=m<='202310']
Rref=float(np.median(ref)); res['Rref_median']=Rref; res['Rref_n']=len(ref)
base=np.mean([O[m] for m in ['202305','202306','202307','202308','202309','202310']])
res['base_2023_05_10']=base
for m in ['202404','202405','202406']:
    res['drop_'+m]=dict(O=O[m],vs_base=O[m]/base-1,yoy=O[m]/O[str(int(m[:4])-1)+m[4:]]-1,ostar=ostar[m],
                        vs_2023mean=O[m]/np.mean([O['2023%02d'%k] for k in range(1,13)])-1,
                        vs_pre6_2023_07_12=O[m]/np.mean([O[x] for x in ['202307','202308','202309','202310','202311','202312']])-1)
# annual
def yr(y,months=12):
    ms=['%d%02d'%(y,k) for k in range(1,months+1)]
    return dict(O=sum(O[m] for m in ms), RF=sum(rfm.get(m,0) for m in ms), U=sum(um.get(m,0) for m in ms),
                ostar=sum(ostar.get(m,0) for m in ms))
A={y:yr(y) for y in range(2019,2025)}; A['2025_1_7']=yr(2025,7)
for k,v in A.items(): v['R']=v['RF']/v['O']; v['UoverO']=v['U']/v['O']; v['ostar_over_O']=v['ostar']/v['O']-1
res['annual']=A
raw=A[2024]['RF']/A[2023]['RF']-1; off=A[2024]['O']/A[2023]['O']-1; rat=A[2024]['R']/A[2023]['R']-1
res['decomp']=dict(raw=raw,off=off,ratio=rat,share_off=math.log(1+off)/math.log(1+raw),share_ratio=math.log(1+rat)/math.log(1+raw))
res['ratio_2024_vs_mean1923']=A[2024]['R']/np.mean([A[y]['R'] for y in range(2019,2024)])-1
G24=['2024%02d'%k for k in range(1,9)]
res['deficit_2024_total']=Rref*A[2024]['O']-A[2024]['RF']
res['deficit_2024_gap']=sum(Rref*O[m]-rfm.get(m,0) for m in G24)
res['deficit_2024_nongap']=sum(Rref*O[m]-rfm.get(m,0) for m in ['2024%02d'%k for k in range(9,13)])
res['missing_found_2024_gap']=sum(miss.get(m,0) for m in G24)
res['monthly']={m:mon[m] for m in mon if m>='202301'}
res['secs']=time.time()-t0
json.dump(res, open(OUT/'r2_out.json','w',encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k:v for k,v in res.items() if k!='monthly'}, ensure_ascii=False, indent=1, default=float))
