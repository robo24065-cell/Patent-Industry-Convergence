# -*- coding: utf-8 -*-
# 내려받은 PDF를 텍스트로 뽑는다(쪽 표시 포함). 사용: python -I p1_pdftext.py <pdf> <out.txt>
import sys, fitz
doc = fitz.open(sys.argv[1])
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for i, p in enumerate(doc):
        f.write(f"\n=====PAGE {i+1}=====\n"); f.write(p.get_text("text"))
print(sys.argv[1], doc.page_count)
