# 50 KSIC <-> ECOS 물가 품목 수작업 연결표 (품목명·KSIC 해설 예시 기준)
# quality: direct(품목이 산업 핵심제품) / proxy(인접 품목·상위분류) / none(대응 품목 없음: 가공서비스업·신산업)
import pandas as pd, pathlib
OUT = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext")
def rng(p, a, b):
    return [f'{p}{i:02d}' for i in range(a, b + 1)]
M = {  # basic = PPI 기본분류 소분류(6자리, 연속성 높음)
'15220': dict(ppi=['30221401','30221402'], imp=['30221401','30221402'], exp=['30221401'], basic=['302214'], q='proxy'),
'20202': dict(ppi=rng('305211',1,12), imp=['30521101','30521102','30521103','30521104'], exp=rng('305211',1,10), basic=['305211'], q='direct'),
'20411': dict(ppi=rng('305611',1,5), imp=['30561101'], exp=['30561101'], basic=['305611'], q='direct'),
'20423': dict(ppi=rng('305622',1,7), imp=['30562201'], exp=['30562201'], basic=['305622'], q='direct'),
'20491': dict(ppi=['30563201','30563303'], imp=['30563301','30563201'], exp=[], basic=['305632'], q='direct'),
'21220': dict(ppi=['30541105'], imp=[], exp=[], basic=[], q='direct'),
'23992': dict(ppi=['30624301'], imp=[], exp=['30624301'], basic=['306243'], q='direct'),
'24191': dict(ppi=rng('307141',1,5), imp=['30714101'], exp=rng('307141',1,4), basic=['307141'], q='direct'),
'24219': dict(ppi=['30721401','30721402','30721501'], imp=['30721401','30721402','30721501','30721502'], exp=['30721401','30721402','30721501'], basic=['307214','307215'], q='direct'),
'25914': dict(ppi=['30812201'], imp=[], exp=[], basic=['308122'], q='direct'),
'25922': dict(ppi=[], imp=[], exp=[], basic=[], q='none'),
'25924': dict(ppi=[], imp=[], exp=[], basic=[], q='none'),
'25932': dict(ppi=rng('308142',1,4), imp=[], exp=['30814201'], basic=['308142'], q='direct'),
'26111': dict(ppi=['30911201','30911202'], imp=['30911201','30911202'], exp=['30911201','30911202'], basic=['309112'], q='direct'),
'26112': dict(ppi=['30911203'], imp=['30911203'], exp=['30911203'], basic=['309112'], q='direct'),
'26121': dict(ppi=['30911102'], imp=['30911103'], exp=['30911102'], basic=['309111'], q='direct'),
'26129': dict(ppi=['30911101'], imp=['30911101','30911102','30911104','30911106'], exp=['30911101'], basic=['309111'], q='direct'),
'26211': dict(ppi=['30921102','30921103','30921104'], imp=['30921102'], exp=['30921102'], basic=['309211'], q='direct'),
'26212': dict(ppi=['30921201'], imp=[], exp=['30921201'], basic=['309212'], q='direct'),
'26223': dict(ppi=['30931102','30931103'], imp=['30931101'], exp=['30931101'], basic=['309311'], q='direct'),
'26224': dict(ppi=[], imp=['30931101'], exp=['30931101'], basic=['309311'], q='proxy'),
'26291': dict(ppi=['30932101'], imp=['30932101'], exp=['30932101'], basic=['309321'], q='direct'),
'26322': dict(ppi=['30941301'], imp=[], exp=['30941301'], basic=['309413'], q='direct'),
'26323': dict(ppi=[], imp=['30941301'], exp=['30941302'], basic=['309413'], q='direct'),
'26422': dict(ppi=['30951201'], imp=['30951201'], exp=['30951201'], basic=['309512'], q='direct'),
'26429': dict(ppi=[], imp=['30951301'], exp=['30951301'], basic=['309513'], q='proxy'),
'26519': dict(ppi=['30952201'], imp=[], exp=['30952201'], basic=['309522'], q='direct'),
'26529': dict(ppi=['30952302','30952303'], imp=['30952301','30952302'], exp=[], basic=['309523'], q='direct'),
'27199': dict(ppi=['30961103','30961104','30961106','30961107'], imp=['30961101','30961103','30961105'], exp=['30961103','30961201'], basic=['309611'], q='direct'),
'27211': dict(ppi=['30961201','30961202'], imp=['30961201'], exp=[], basic=[], q='direct'),
'27212': dict(ppi=['30961203','30961204'], imp=[], exp=[], basic=['309612'], q='direct'),
'27301': dict(ppi=['30962201','30962202'], imp=['30962201','30962202','30962203'], exp=['30962201'], basic=['309622'], q='direct'),
'28111': dict(ppi=['31011101','31011102','31011103'], imp=['31011101','31011102'], exp=['31011101','31011102'], basic=['310111'], q='direct'),
'28122': dict(ppi=['31012305'], imp=['31012304'], exp=['31012303'], basic=['310123'], q='direct'),
'28511': dict(ppi=['31015101','31015102','31015202','31015203','31015204','31015205'], imp=['31015201'], exp=['31015101','31015201'], basic=['310151','310152'], q='direct'),
'28902': dict(ppi=[], imp=['31016301','30624501'], exp=[], basic=['310163'], q='proxy'),
'29131': dict(ppi=['31112101','31112106'], imp=['31112101'], exp=['31112101'], basic=['311121'], q='direct'),
'29132': dict(ppi=['31112102','31112103','31112104','31112107'], imp=['31112102'], exp=['31112102'], basic=['311121'], q='direct'),
'29133': dict(ppi=rng('311131',1,6), imp=['31113101'], exp=['31113101'], basic=['311131'], q='direct'),
'29150': dict(ppi=['31116201'], imp=[], exp=[], basic=['311162'], q='proxy'),
'29162': dict(ppi=['31114102'], imp=[], exp=[], basic=['311141'], q='direct'),
'29210': dict(ppi=['31121101','31121102','31121103'], imp=[], exp=['31121101'], basic=['311211'], q='direct'),
'29229': dict(ppi=[], imp=[], exp=[], basic=['311221'], q='proxy'),
'29241': dict(ppi=['31121201','31121203','31121204','31121205'], imp=[], exp=['31121201','31121202','31121203'], basic=['311212'], q='direct'),
'29280': dict(ppi=['31125301'], imp=['31125301'], exp=[], basic=['311253'], q='direct'),
'30393': dict(ppi=['31213211'], imp=[], exp=[], basic=['312132'], q='direct'),
'31202': dict(ppi=[], imp=[], exp=[], basic=[], q='none'),
'31312': dict(ppi=[], imp=[], exp=[], basic=[], q='none'),
'33301': dict(ppi=['31312201'], imp=[], exp=[], basic=['313122'], q='direct'),
'33309': dict(ppi=[], imp=['31312201'], exp=[], basic=['313122'], q='proxy'),
}
SER = {'ppi': 'ppi_item', 'imp': 'imp_item', 'exp': 'exp_item', 'basic': 'ppi_basic'}
rows = []
for k, v in M.items():
    for s in SER:
        for c in v[s]:
            rows.append(dict(ksic=k, series=SER[s], item_code=c + 'AA', quality=v['q']))
m = pd.DataFrame(rows)
d = pd.read_parquet(OUT / 'demand_ecos_price_item_annual_1995_2021.parquet')
names = d.drop_duplicates(['series', 'item_code'])[['series', 'item_code', 'item_name']]
m = m.merge(names, on=['series', 'item_code'], how='left')
miss = m[m.item_name.isna()]
print('매칭 실패', len(miss)); print(miss)
m.to_csv(OUT / 'demand_ksic_ecos_price_map.csv', index=False, encoding='utf-8-sig')
print(m.groupby('series').ksic.nunique(), m.ksic.nunique())
