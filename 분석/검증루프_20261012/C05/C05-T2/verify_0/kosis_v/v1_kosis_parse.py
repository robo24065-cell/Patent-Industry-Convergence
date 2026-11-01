import json,sys,io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
d=sys.argv[1]
raw=json.load(open(d+r'\download\kosis_DT_133N_9815A_raw_capture.json',encoding='utf-8'))
for i,x in enumerate(raw): print(i,x.get('type'),len(x.get('text','')), x.get('text','')[-300:].replace('\n','|'))
t,end=json.JSONDecoder().raw_decode(raw[0]['text'])
lines=t.split('\n')
print('lines',len(lines))
rows=[l.split('\t') for l in lines]
for c in rows[:6]: print(len(c),c[:4])
for c in rows[2:]: print(c[0],c[1],len(c))
json.dump(rows,open(sys.argv[2],'w',encoding='utf-8'),ensure_ascii=False)
