import csv,re
E=open(r'..\..\ERRATA.txt',encoding='utf-8').read().split('\n')
r=list(csv.DictReader(open('errata_status_v18.csv',encoding='utf-8-sig')))
for x in r:
    if x['상태'] in ('미반영','부분') and x['분류'] in ('선택','facts'):
        s=x['ERRATA줄'].split(',')[0]
        head=E[int(s)-1][:110] if s.isdigit() else '(no line)'
        print(x['ID'],x['범위'],x['상태'],x['분류'],x['글자증감'],'|',head)
