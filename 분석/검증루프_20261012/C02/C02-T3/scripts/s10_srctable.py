# 출처표(부록) 점검: (1) '쓰인 절' 열이 실제 원고 절과 맞는지(X2), (2) 1–2장·3.1·3.6 관련 행 값의 원천 대조(일부 자동)
import sys, re, json, sqlite3
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
B = P + r"\분석"; MC = B + r"\모델경쟁_20260927"; OUT = B + r"\검증루프_20261012\C02\C02-T3"
MS = open(B + r"\보고서_20261012\manuscript_v3.md", encoding="utf-8").read().splitlines()
end_body = next(i for i, l in enumerate(MS) if l.startswith("# [부록"))
sec_of = []; cur = "표지·요약"
for i, l in enumerate(MS[:end_body]):
    m = re.match(r"^#{1,3} (\d+(?:\.\d+)*)[. ]", l)
    if m: cur = m.group(1)
    elif l.startswith("## 데이터 출처"): cur = "데이터 출처"
    elif l.startswith("## 참고문헌"): cur = "참고문헌"
    sec_of.append(cur)
def text_of(sec):
    if sec in ("표지·요약", "요약"): sec = "표지·요약"
    if sec.startswith("그림"):
        n = re.findall(r"\d+", sec)[0]
        return "\n".join(l for l in MS[:end_body] if re.match(rf"^그림 {n}\.", l) or f"(그림 {n}" in l)
    return "\n".join(l for l, s in zip(MS[:end_body], sec_of) if s == sec or s.startswith(sec + "."))
def variants(tok):
    t = tok.replace(",", "")
    out = {tok}
    try:
        d = Decimal(t)
        for nd in range(0, 4):
            r = d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
            s = f"{r:,}" if abs(r) >= 1000 else str(r)
            out.add(s); out.add(s.replace("-", "−"))
            if abs(d) < 1:  # 비율 → %
                p = (d * 100).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP); out.add(str(p))
    except Exception:
        pass
    return {v for v in out if len(v.replace(".", "").replace(",", "").replace("−", "").replace("-", "")) >= 2}
st = next(i for i, l in enumerate(MS) if l.startswith("| id | 쓰인 절"))
rows = []
for i in range(st + 2, len(MS)):
    l = MS[i]
    if not l.startswith("|"): break
    c = [x.strip() for x in l.strip("|").split("|")]
    rid, secs, val, src = c[0], c[1], c[2], c[3]
    toks = [t for t in re.findall(r"[−+-]?\d[\d,]*\.\d+|\d{1,3}(?:,\d{3})+|\b\d{2,}\b", val) if not re.fullmatch(r"(19|20)\d\d", t.replace("−", ""))]
    for s in [x.strip() for x in secs.split(",")]:
        if s in ("표지·요약",): s2 = "표지·요약"
        else: s2 = s
        txt = text_of(s2)
        hit = [t for t in toks if any(v in txt for v in variants(t.replace("−", "-")))]
        rows.append(dict(출처표_줄=i + 1, id=rid, 쓰인_절=s, 숫자토큰=" ".join(toks[:8]), 절에서_발견=" ".join(hit[:6]), 판정=("확인" if hit else ("숫자 없음(검토)" if toks else "숫자 토큰 없음"))))
df = pd.DataFrame(rows)
df.to_csv(OUT + r"\x2_section_check.csv", index=False, encoding="utf-8-sig")
print(df.판정.value_counts().to_string())
print(df[df.판정 != "확인"][["출처표_줄", "id", "쓰인_절", "숫자토큰"]].to_string())
# (2) 원천 대조 보강: 등록원부 소멸·회복, C-AGREE 5개 행
c = sqlite3.connect("file:" + (P + r"\수집파이프라인\registry_full.sqlite").replace("\\", "/") + "?mode=ro", uri=True)
R = {"right_events": c.execute("select rgstCsName, count(*), count(distinct rgst_no) from right where rgstCsName like '%소멸%' or rgstCsName like '%회복%' group by 1 order by 2 desc limit 12").fetchall()}
c.close()
c1 = pd.read_csv(MC + r"\_shared_report\fusion_advantage\c5_R10\c1_label_agreement.csv", encoding="utf-8-sig")
c1w = c1[c1.window.str.startswith("출원2015")].set_index("cc")
R["c_agree"] = {cc: dict(n=int(c1w.loc[cc, "n"]), group=round(c1w.loc[cc, "cw_group"] * 100, 1), cw5=round(c1w.loc[cc, "cw5"] * 100, 1), txt=round(c1w.loc[cc, "txt5"] * 100, 1), free=round(c1w.loc[cc, "free5"] * 100, 1), pkr=round(c1w.loc[cc, "pkr5"] * 100, 1)) for cc in ["KR", "US", "JP", "CN", "EP"]}
json.dump(R, open(OUT + r"\s10_srctable_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
for i in range(st + 2, len(MS)):
    if MS[i].startswith("| C-AGREE-") or MS[i].startswith("| DATA-REG-EVENTS") or MS[i].startswith("| MAP-SYNC") or MS[i].startswith("| LINK-NPS |") or MS[i].startswith("| DATA-REGISTRY"):
        print(i + 1, MS[i][:260])
