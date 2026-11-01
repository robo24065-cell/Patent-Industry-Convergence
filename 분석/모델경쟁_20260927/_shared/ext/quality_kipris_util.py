# KIPRIS Plus 키 로더 (키 값은 출력·저장하지 않음)
import re
KF=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\API키_기록.txt'
def kipris_key():
    t=open(KF,encoding='utf-8').read()
    sec=t.split('[KIPRIS Plus]')[1]
    m=re.search(r'키\s*=\s*(\S+)',sec); return m.group(1)
