import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
EXT=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext'
d=pd.read_parquet(EXT+r'\firm_nps_wkpl_202312.parquet')
print(d.shape); print(d.dtypes)
print(d.stat.value_counts().head()); print(d.form.value_counts().head())
for pat in ['삼성전자','현대자동차','엘지전자','에스케이하이닉스','포스코']:
    x=d[d.name.astype(str).str.contains(pat)][['name','biz6','stat','form','adpt_dt','n_emp','ldong']]
    print('==',pat,len(x)); print(x.sort_values('n_emp',ascending=False).head(12).to_string())
