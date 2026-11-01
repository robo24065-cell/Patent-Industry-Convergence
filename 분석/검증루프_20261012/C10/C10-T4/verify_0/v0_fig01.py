# verify_0: E-C10T4-01(그림 1 '48–54%') 독립 확인 — pandas/fitz 미사용, csv·zipfile·hashlib·PIL만
import sys, os, csv, zipfile, hashlib, json, re
sys.stdout.reconfigure(encoding="utf-8")
A = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
R = os.path.join(A, "보고서_20261012"); V = os.path.join(A, r"검증루프_20261012\C10\C10-T4\verify_0")
out = {}
# (1) 원천 재계산: c1_label_agreement.csv 출원2015-19_발명 해외 free5
vals = {}
with open(os.path.join(A, r"모델경쟁_20260927\_shared_report\fusion_advantage\c5_R10\c1_label_agreement.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["window"] == "출원2015-19_발명" and r["cc"] in ("US", "JP", "CN", "EP"):
            vals[r["cc"]] = float(r["free5"]) * 100
out["foreign_free5_pct"] = {k: round(v, 3) for k, v in vals.items()}
out["range_int_round_half_up"] = [int(min(vals.values()) + 0.5), int(max(vals.values()) + 0.5)]
# (2) 그림 1 PNG: figs 파일 = docx 내장 = 렌더 스크립트 문구
png = os.path.join(R, r"figs\fig01_framework.png")
h_fig = hashlib.sha1(open(png, "rb").read()).hexdigest()
z = zipfile.ZipFile(os.path.join(R, "보고서_초안_v11.docx"))
media = {n: hashlib.sha1(z.read(n)).hexdigest() for n in z.namelist() if n.startswith("word/media/")}
out["fig01_sha1"] = h_fig; out["fig01_in_v11_docx"] = [n for n, h in media.items() if h == h_fig]
src = open(os.path.join(R, r"figs\make_diagrams.py"), encoding="utf-8").read().splitlines()
out["make_diagrams_lines"] = [(i + 1, l.strip()) for i, l in enumerate(src) if "48–5" in l or "48-5" in l]
out["fig01_png_mtime"] = __import__("time").strftime("%Y-%m-%d %H:%M", __import__("time").localtime(os.path.getmtime(png)))
out["manuscript_mtime"] = __import__("time").strftime("%Y-%m-%d %H:%M", __import__("time").localtime(os.path.getmtime(os.path.join(R, "manuscript_v3.md"))))
ms = open(os.path.join(R, "manuscript_v3.md"), encoding="utf-8").read().splitlines()
out["manuscript_hits"] = {p: [i + 1 for i, l in enumerate(ms) if p in l] for p in ["48–54", "48–53", "48.2–53.5"]}
# (3) 시각 확인용 잘라내기
from PIL import Image
im = Image.open(png); w, hh = im.size
im.crop((int(w * 0.34), int(hh * 0.64), int(w * 0.66), int(hh * 0.99))).save(os.path.join(V, "fig01_control_crop.png"))
out["crop"] = "verify_0\fig01_control_crop.png"
json.dump(out, open(os.path.join(V, "v0_fig01_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
