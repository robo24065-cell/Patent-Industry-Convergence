import json, os
R2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a=json.load(open(os.path.join(R2,'final_spec_r2.json'),encoding='utf-8'));b=json.load(open(os.path.join(R2,'..','final_spec.json'),encoding='utf-8'))
print({k:v for k,v in a.items() if k!='spec'})
s=a['spec']
print('baseline same', s['baseline']==b['baseline'], s['baseline'])
print('ens', s['ensemble'], b['ensemble'])
print('train', s['train'])
for c in s['components']:
  print(c, s['components'][c]==b['components'].get(c), s['components'][c]['num'])
print('diff keys', [k for k in s if k!='ensemble' and s.get(k)!=b.get(k)])
