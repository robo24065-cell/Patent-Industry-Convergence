# -*- coding: utf-8 -*-
# zip 안의 표 파일(xls/xlsx/pdf)만 새 빈 폴더에 푼다. 이름은 cp949 복원, 경로 탈출 차단, 실행 파일 무시.
import sys, os, zipfile
sys.stdout.reconfigure(encoding="utf-8")
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=False)
z = zipfile.ZipFile(src)
for i, info in enumerate(z.infolist()):
    if info.is_dir(): continue
    n = info.filename
    if not (info.flag_bits & 0x800):
        try: n = n.encode("cp437").decode("cp949")
        except Exception: pass
    base = os.path.basename(n.replace("\\", "/"))
    ext = os.path.splitext(base)[1].lower()
    if ext not in (".xls", ".xlsx", ".pdf", ".csv"):
        print("SKIP", n); continue
    sub = os.path.dirname(n.replace("\\", "/")).replace("/", "__").replace("..", "_")
    out = os.path.join(dst, f"{i:03d}_" + (sub + "__" if sub else "") + base)
    with open(out, "wb") as f: f.write(z.read(info))
    print(i, n, info.file_size)
