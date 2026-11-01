import json,sys
p=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\T3_기술셀부상예측\R3\final_spec_r3.json"
s=json.load(open(p,encoding='utf-8'))
sys.stdout.reconfigure(encoding='utf-8')
print({k:v for k,v in s.items() if k!='cfg'})
for c in s['cfg']['components']: print(c['type'],len(c['features']),c['features'])
print({k:v for k,v in s['cfg'].items() if k!='components'})
