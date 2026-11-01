import sys
base = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C18\C18-T3"
s = open(base + r"\scripts\s6_facts.py", encoding='utf-8').read()
old = 'OUT = L + r"\\C18\\C18-T3"; WK = OUT + r"\\facts_work"'
assert old in s
s = s.replace(old, 'OUT = L + r"\\C18\\C18-T3\\verify_2\\rerun"; WK = OUT + r"\\facts_work"')
# 다른 곳에서 원래 작업폴더를 쓰는지 점검
assert s.count('C18-T3"') == 0 or True
open(base + r"\verify_2\rerun\s6_rerun.py", 'w', encoding='utf-8').write(s)
print([l for l in s.split('\n') if 'C18' in l])
