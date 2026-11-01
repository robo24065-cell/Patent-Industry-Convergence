# verify_3 v2: 이전 사건의 대상 특허 등록연도 구성(잠금 창 재고 동결 효과) — 읽기전용
import sqlite3, pandas as pd
p=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite'
out=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C01\C01-T4\verify_3\v2_stock.txt'
con=sqlite3.connect('file:'+p+'?mode=ro',uri=True)
A="('양도','일부양도','지분양도','신탁계약에 의한 양도')"
d=pd.read_sql(f"select o.rgst_no, o.ownerRgstCsDate d, o.ownerName nm, p.rgstDate rg from owner o join patent p on p.rgst_no=o.rgst_no where o.ownerRgstCsName in {A} and o.ownerRgstCsDate>='2012'",con)
L=[]
L.append('patent rgstDate range '+str(pd.read_sql("select min(rgstDate),max(rgstDate) from patent",con).values.tolist()))
L.append('patent rgst year counts (all) '+str(pd.read_sql("select substr(rgstDate,1,4) y,count(*) from patent group by 1 order by 1",con).values.tolist()))
L.append('patent rgst monthly 2025 '+str(pd.read_sql("select substr(rgstDate,1,6) y,count(*) from patent where rgstDate>='2024' group by 1 order by 1",con).values.tolist()))
d['dy']=d.d.str[:4].astype(int); d['ry']=d.rg.str[:4].astype(int)
d['corp']=d.nm.str.contains('주식회사|\(주\)|㈜',regex=True)
# 19개월 창별: 창 시작 이후(=창 안) 등록된 특허 대상 이전 비율, 창 시작 전 12개월 내 등록 비율
for t0 in range(2012,2025):
    s=f"{t0+1}0101"; e=f"{t0+2}0731"
    w=d[(d.d>=s)&(d.d<=e)]
    if len(w)==0: continue
    inwin=(w.rg>=s).mean(); after_cut=(w.rg>'20250831').mean()
    L.append(f"t0={t0} 창 {s}-{e}: 허용사유 이전 {len(w)} (법인 {int(w.corp.sum())}), 창 안 등록 특허 비율 {inwin:.3f}, 등록 2025-09 이후 {after_cut:.3f}")
# 이전 시점의 특허 나이(등록 후 경과연수) 분포
d['age']=d.dy-d.ry
L.append('age dist (2015-2023 transfers) '+str(d[(d.dy>=2015)&(d.dy<=2023)].age.clip(upper=10).value_counts(normalize=True).sort_index().round(3).to_dict()))
# 상반기 비교 Jan-Jul
for y in [2022,2023,2024,2025,2026]:
    w=d[(d.d>=f"{y}0101")&(d.d<=f"{y}0731")]
    L.append(f"{y} Jan-Jul 허용사유 {len(w)}, 대상특허 등록<= {y-1}년말 기준: {int((w.rg<f'{y}0101').sum())}, 창안등록 {int((w.rg>=f'{y}0101').sum())}")
open(out,'w',encoding='utf-8').write('\n'.join(L))
