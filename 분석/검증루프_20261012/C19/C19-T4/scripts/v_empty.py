import csv,os
D=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B=os.path.dirname(os.path.dirname(D))
o=open(os.path.join(D,'cache','v_empty.txt'),'w',encoding='utf-8')
L=open(os.path.join(B,'ERRATA.txt'),encoding='utf-8').read().split('\n')
for r in csv.DictReader(open(os.path.join(B,'C18','C18-T3','errata_status_v18.csv'),encoding='utf-8-sig')):
  try: ln=int(r['ERRATA줄']); h=L[ln-1]
  except: ln=None; h=''
  if not h.strip() or r['ID'] not in h:
    hits=[i+1 for i,t in enumerate(L) if r['ID'] in t]
    o.write(f"{r['ID']} ERRATA줄={r['ERRATA줄']!r} 범위={r['범위']} hits={hits[:8]} first={L[hits[0]-1][:150] if hits else ''}\n")
