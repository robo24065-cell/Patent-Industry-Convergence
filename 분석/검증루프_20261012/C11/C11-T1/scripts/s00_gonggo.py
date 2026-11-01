# -*- coding: utf-8 -*-
import os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
from paths import *
import fitz
pdfs = glob.glob(os.path.join(BASE, "*.pdf"))
out = []
for p in pdfs:
    d = fitz.open(p)
    t = "".join(pg.get_text() for pg in d)
    out.append("#### " + os.path.basename(p) + "\n" + t)
open(guard_path(os.path.join(CACHE, "gonggo.txt")), "w", encoding="utf-8").write("\n".join(out))
print(len(out), sum(len(x) for x in out))
