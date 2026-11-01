import csv,os
D=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
o=open(os.path.join(D,'cache','v_grades.txt'),'w',encoding='utf-8')
for r in csv.DictReader(open(os.path.join(D,'cache','orig_grades.csv'),encoding='utf-8-sig')):
  if r['원등급'] or not r['헤더']: o.write(f"{r['ID']} | {r['원등급']} | {r['헤더'][:130]}\n")
