# -*- coding: utf-8 -*-
# 내려받은 PDF/HTML에서 텍스트 추출(읽기만), 키워드 주변 출력
import sys, re
sys.stdout.reconfigure(encoding="utf-8")
path, kw = sys.argv[1], sys.argv[2]
win = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
if path.lower().endswith(".pdf"):
    from pypdf import PdfReader
    t = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
else:
    raw = open(path, "rb").read()
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            t = raw.decode(enc); break
        except Exception:
            continue
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t); t = re.sub(r"[ \t\r\f\v]+", " ", t); t = re.sub(r"\n\s*\n+", "\n", t)
idx = [m.start() for m in re.finditer(re.escape(kw), t)]
print("len", len(t), "hits", len(idx))
seen = -10 ** 9
for i in idx:
    if i - seen < win:
        continue
    print("-----", i); print(t[max(0, i - win // 3):i + win]); seen = i
