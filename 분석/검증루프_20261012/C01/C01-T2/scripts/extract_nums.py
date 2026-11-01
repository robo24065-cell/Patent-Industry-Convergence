import re, sys, csv
sys.stdout.reconfigure(encoding='utf-8')
B = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/"
md = open(B+"보고서_20261012/manuscript_v2_from_response.md", encoding='utf-8').read().split('\n')
R = [(7,22,'요약'),(201,290,'3.2-3.5'),(380,396,'4.2'),(411,418,'4.4')]
NUM = re.compile(r'[−\-+]?\d[\d,]*(?:\.\d+)?%?p?')
out=[]
for a,b,sec in R:
    for i in range(a,b+1):
        l = md[i-1]
        if l.strip().startswith('[그림'): continue
        for m in NUM.finditer(l):
            s=max(0,m.start()-25); out.append((sec,i,m.group(),l[s:m.end()+10].replace('\n',' ')))
with open(B+"검증루프_20261012/C01/C01-T2/regex_numbers_raw.csv",'w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f); w.writerow(['절','행','토큰','문맥']); w.writerows(out)
print(len(out))
for r in out: print(r[1], r[2], '|', r[3])
