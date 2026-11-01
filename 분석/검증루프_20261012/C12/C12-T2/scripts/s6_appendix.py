# C12-T2 s6: 부록 출처표 미처리 522토큰(C09-T2 coverage_map, v10 줄) → v13 줄 대응 + 원천 파일 원값 반올림 대조
# 원천 파일은 읽기만. 출력: appendix_tokens_raw.csv (작업 폴더)
import sys, os, re, csv, glob, difflib, json
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
AN = ROOT + r"\분석"
MC = AN + r"\모델경쟁_20260927"
C09 = AN + r"\검증루프_20261012\C09\C09-T2"
W = AN + r"\검증루프_20261012\C12\C12-T2"
v10 = open(C09 + r"\cache\manuscript_now_88d77045.md", encoding="utf-8").read().split("\n")
v13 = open(AN + r"\보고서_20261012\manuscript_v3.md", encoding="utf-8").read().split("\n")
cm = pd.read_csv(C09 + r"\coverage_map.csv", encoding="utf-8-sig", dtype=str)
u = cm[(cm.scope == "R") & cm.excluded.isna() & cm.recompute_status.str.contains("최하", na=False)].copy()
assert len(u) == 522, len(u)
# v11->v12, v12->v13 변경 행(v13 줄) = A 단계에서 감사
chg = json.load(open(W + r"\cache\s1_diff.json", encoding="utf-8"))
changed_v13 = {}
for o in chg:
    if o["step"] == "v12->v13": changed_v13[o["line_new"]] = "v12->v13"
for o in chg:
    if o["step"] == "v11->v12" and o["line_new"] not in changed_v13: changed_v13[o["line_new"]] = "v11->v12"

def cells(l):
    return [x.strip() for x in l.strip().strip("|").split("|")]
def fp(l):
    return re.sub(r"[\d\s.,%+−\-–~]", "", l)
id13 = {}
for i, l in enumerate(v13):
    if l.startswith("|"):
        c = cells(l)
        if c and c[0] and not set(c[0]) <= set("-: "): id13.setdefault(c[0], i + 1)
def map_line(ln10):
    l = v10[ln10 - 1]
    if l.startswith("|"):
        c0 = cells(l)[0]
        if c0 in id13: return id13[c0], "행ID"
    # 문장 지문(숫자 제거) 최근접
    f = fp(l); best, br = None, 0
    lo = max(0, ln10 - 40); hi = min(len(v13), ln10 + 40)
    for j in range(lo, hi):
        r = difflib.SequenceMatcher(None, f, fp(v13[j]), autojunk=False).ratio()
        if r > br: best, br = j + 1, r
    return (best, "지문%.2f" % br) if br >= 0.85 else (None, "대응 없음%.2f" % br)

def resolve(srccell):
    out = []
    for part in re.split(r";\s*", srccell):
        p = part.strip()
        if not p: continue
        p = p.replace("(MC)", MC).replace("(분석)", AN)
        if p.startswith("분석\\"): p = ROOT + "\\" + p
        elif p.startswith("수집파이프라인\\"): p = ROOT + "\\" + p
        elif p.startswith("보고서_20261012\\"): p = AN + "\\" + p
        elif not re.match(r"^[A-Za-z]:", p):
            # 앞 경로의 폴더에 붙는 파일명(예: 'res_te...' 축약)
            if out: p = os.path.join(os.path.dirname(out[-1]), p)
        p = re.split(r"\s+(?:\(|§|표|fig|합계)", p)[0].strip()
        p = re.sub(r"\s*\(.*$", "", p)
        out.append(p)
    return out
NUMRE = re.compile(r"(?<![\w.])[-−+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?")
NUMRE2 = re.compile(r"(?<![\d.])[-−+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")  # 쉼표 묶음 없이(CSV 칸 구분 쉼표 대비)
_cache = {}
def src_numbers(path):
    if path in _cache: return _cache[path]
    vals = None
    if os.path.isfile(path) and not path.endswith(".sqlite") and os.path.getsize(path) < 60e6:
        try:
            t = open(path, encoding="utf-8-sig", errors="replace").read()
            vals = []
            for m in list(NUMRE.finditer(t)) + list(NUMRE2.finditer(t)):
                s = m.group(0).replace("−", "-").replace(",", "")
                try: vals.append(float(s))
                except: pass
        except Exception:
            vals = None
    _cache[path] = vals
    return vals
def tokval(tok):
    s = tok.replace("−", "-").replace(",", "").replace("+", "")
    d = len(s.split(".")[1]) if "." in s else 0
    return float(s), d
def rnd(x, d):
    if not (abs(x) < 1e15): return float("nan")
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
def match(tok, vals):
    v, d = tokval(tok); av = abs(v)
    hits = []
    for scale in (1, 100, 0.01, 1000):
        for s in vals:
            if rnd(abs(s) * scale, d) == round(av, d):
                hits.append((scale, s))
                if len(hits) > 3: break
        if hits: return hits[0]
    return None
rows = []
for _, r in u.iterrows():
    ln10 = int(r.line); tok = r.token
    l10 = v10[ln10 - 1]
    ln13, how = map_line(ln10)
    l13 = v13[ln13 - 1] if ln13 else ""
    still = (tok in l13) if ln13 else False
    c10 = cells(l10) if l10.startswith("|") else []
    rid = c10[0] if c10 else l10[:40]
    srccell = c10[-1] if len(c10) >= 4 else ""
    paths = resolve(srccell) if srccell else []
    v, d = tokval(tok)
    weak = (d == 0 and abs(v) <= 10) or tok in ("100", "50", "1,000")
    status, mt, mv, note = "", "", "", ""
    if ln13 in changed_v13 and not still:
        status = "변경됨(v11–v13, A단계 감사)"
    elif ln13 in changed_v13:
        status = "변경 행(A단계)·토큰 유지"
    exist = [p for p in paths if os.path.exists(p)]
    numsrc = [p for p in exist if os.path.isfile(p) and not p.endswith(".sqlite")]
    hit = None
    for p in numsrc:
        vals = src_numbers(p)
        if vals:
            hit = match(tok, vals)
            if hit: mt = "원값 반올림 일치(x%s)" % hit[0]; mv = hit[1]; break
    if not paths: verdict = "원천 경로 없음(메모 문장)"
    elif not exist: verdict = "원천 파일 없음"
    elif not numsrc: verdict = "원천=DB(별도 질의)"
    elif hit and weak: verdict = "일치(약함: 작은 정수)"
    elif hit: verdict = "일치"
    else: verdict = "불일치(수동 확인)"
    rows.append(dict(v10_line=ln10, v13_line=ln13 or "", map=how, row_id=rid[:40], token=tok, unit=r.unit or "", c09_cls=r.cls,
                     a_stage=status, token_in_v13=still, n_src=len(paths), src_exist=len(exist), verdict=verdict, match=mt,
                     src_value=mv, src=(exist[0] if exist else (paths[0] if paths else ""))[-110:], context=(r.context or "")[:90]))
df = pd.DataFrame(rows)
df.to_csv(W + r"\cache\appendix_tokens_raw.csv", index=False, encoding="utf-8-sig")
print(df.verdict.value_counts().to_string())
print(df.a_stage.value_counts().to_string())
print(df["map"].str[:2].value_counts().to_string())
print(df[df.verdict.str.startswith("불일치")][["v10_line", "v13_line", "row_id", "token", "src"]].to_string())
print(df[df.verdict.str.contains("없음|DB")][["v10_line", "row_id", "token", "verdict", "src"]].to_string())
