import sys, fitz
sys.stdout.reconfigure(encoding="utf-8")
d = fitz.open(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\[붙임1] 특허-산업 융합데이터 활용 아이디어 공모전 공고문.pdf")
t = "\n".join(f"--- p{i+1}\n" + p.get_text() for i, p in enumerate(d))
open(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C10\C10-T4\cache\notice_text.txt", "w", encoding="utf-8").write(t)
print(d.page_count, len(t)); print(t)
