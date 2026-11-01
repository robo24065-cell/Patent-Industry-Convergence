import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
EXT=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext'
d=pd.read_parquet(EXT+r'\firm_nps_wkpl_202312.parquet',columns=['name','biz6','stat','n_emp','ldong'])
d=d[d.stat=='1']
for pat in ['^기아','현대제철','삼성중공업','엘지디스플레이','삼성에스디아이','농심','씨제이제일제당','오뚜기','한국타이어','현대위아']:
    x=d[d.name.astype(str).str.contains(pat)].sort_values('n_emp',ascending=False).head(8)
    print('==',pat); print(x.to_string())
