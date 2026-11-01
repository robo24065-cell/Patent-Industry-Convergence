# -*- coding: utf-8 -*-
r"""C07-T1 독립 검증자(verify_0). 본 작업 스크립트(s1~s5)를 import하지 않고 다른 코드 경로로 반박을 시도한다.
(1) 공식 대조 무작위 30칸: 작업 파일 csv를 csv 모듈로, KOSIS JSON을 json 모듈로 직접 읽어 비교(pandas 미사용).
(2) 4개 사례 성장률을 '작업 파일'이 아니라 '공식 JSON'에서 직접 계산(B0·V1a).
(3) V1a SP-FE r을 OLS(그룹×창 더미) 잔차 상관으로, SP-VAR 그룹 안 분산 비중을 ANOVA 식으로 재계산, 다른 시드(777)로 순열 p.
(4) '파일 안에 9→10 단절 없음' 반박 시도: 9차 원표 2014 값 = 10차 소급표 2014 값인지(1:1 코드), 2016·2017 값의 LST_CHN_DE.
출력: verify_0\v0_result.json"""
import os, sys, csv, json, math, random
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
H = os.path.join(B, r"분석\검증루프_20261012\C07\C07-T1")
F = os.path.join(B, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv")
res = {}
# (1)
file = {}
for r in csv.DictReader(open(F, encoding="utf-8-sig")):
    file[(r["기준"], r["세세분류코드"], r["항목코드"], int(r["연도"]))] = (r["값"], r["마스킹"])
off = {}
for basis, fn in (("10차", r"download\kosis10\DT_1FS1001_2007_2020.json"), ("11차", r"download\kosis11\DT_1FS1101_2020_2024.json")):
    for r in json.load(open(os.path.join(H, fn), encoding="utf-8")):
        off[(basis, r["C2"][1:] if r["C2"].startswith("C") and len(r["C2"]) == 6 else r["C2"], r["ITM_ID"], int(r["PRD_DE"]))] = r["DT"]
keys = sorted(k for k, (v, m) in file.items() if m == "0" and v != "")
random.seed(20261013)
pick = random.sample(keys, 30)
chk = []
for k in pick:
    fv = float(file[k][0]); ov = off.get(k)
    ok = ov is not None and ov not in ("X", "-", "") and abs(fv - float(ov)) <= 0.005 * abs(float(ov))
    chk.append([*map(str, k), file[k][0], ov, ok])
res["random30"] = {"n_ok": sum(c[-1] for c in chk), "cells": chk}
# 전 칸(독립 경로)도
allok = [abs(float(v) - float(off[k])) <= 0.005 * abs(float(off[k])) for k, (v, m) in file.items() if m == "0" and v != "" and k in off and off[k] not in ("X", "-", "")]
res["all_cells_independent"] = {"n": len(allok), "ok": int(sum(allok))}
# (2) 공식 JSON에서 사례 성장률
def oval(code, it, y):
    v = off.get(("10차", code, it, y)); return float(v)
def g(code, it, a, b, sp):
    return 100 * (math.log(np.mean([oval(code, it, y) for y in b])) - math.log(np.mean([oval(code, it, y) for y in a]))) / sp
cases = {}
for c in ("26211", "26212", "26111", "29280"):
    cases[c] = {lab: {"B0": round(g(c, it, [2008, 2009], [2016, 2017], 8), 3), "V1a": round(g(c, it, [2008, 2009], [2015, 2016], 7), 3)}
                for it, lab in (("T04", "출하"), ("T02", "종사자"), ("T01", "사업체"))}
res["cases_from_official"] = cases
res["sign_flips_V1a"] = [(c, lab) for c, d in cases.items() for lab, x in d.items() if np.sign(x["B0"]) != np.sign(x["V1a"])]
# (3) V1a FE r(OLS 잔차) + SP-VAR
import pandas as pd  # 패널 읽기에만
sys.path.insert(0, os.path.join(B, r"분석\모델경쟁_20260927\_shared_report\label_validity"))
sys.dont_write_bytecode = True
from lv_common import ksic_to_group
P = pd.read_parquet(os.path.join(B, r"분석\모델경쟁_20260927\_shared_report\fusion_advantage\split34\panel_ksic_cc_year.parquet"))
kr = {}
for k, y, n in P[P.cc == "KR"][["ksic", "year", "n"]].itertuples(index=False):
    kr[(k, int(y))] = kr.get((k, int(y)), 0) + n
K50 = sorted(P.ksic.unique()); grp = {k: ksic_to_group(k) for k in K50}
cnt = {}
for k in K50: cnt[grp[k]] = cnt.get(grp[k], 0) + 1
K34 = [k for k in K50 if cnt[grp[k]] > 1]
lmk = lambda k, ys: math.log(np.mean([kr.get((k, y), 0) for y in ys]) + 1)
lmx = lambda k, it, ys: math.log(np.mean([oval(k, it, y) for y in ys]))
def fe_r(it, W):
    rows = []
    for wi, (a, b, s) in enumerate(W):
        for k in K34:
            try:
                x = (lmx(k, it, b) - lmx(k, it, a)) / s
            except Exception:
                continue
            rows.append((f"{grp[k]}_{wi}", (lmk(k, b) - lmk(k, a)) / s, x))
    cells = sorted({r[0] for r in rows}); D = np.array([[1.0 if r[0] == c else 0.0 for c in cells] for r in rows])
    p = np.array([r[1] for r in rows]); x = np.array([r[2] for r in rows])
    rp = p - D @ np.linalg.lstsq(D, p, rcond=None)[0]; rx = x - D @ np.linalg.lstsq(D, x, rcond=None)[0]
    r = float(np.corrcoef(rp, rx)[0, 1])
    rng = np.random.default_rng(777); idx = {c: [i for i, rr in enumerate(rows) if rr[0] == c] for c in cells}; sims = []
    for _ in range(5000):
        pp = p.copy()
        for c, ix in idx.items():
            pp[ix] = p[rng.permutation(ix)]
        e = pp - D @ np.linalg.lstsq(D, pp, rcond=None)[0]
        sims.append(np.corrcoef(e, rx)[0, 1])
    return round(r, 4), round(float(np.mean(np.array(sims) >= r - 1e-12)), 4), len(rows)
W0 = [([2008, 2009], [2012, 2013], 4.0), ([2012, 2013], [2016, 2017], 4.0)]
W1 = [([2008, 2009], [2012, 2013], 4.0), ([2012, 2013], [2015, 2016], 3.0)]
res["FE"] = {f"{lab}_{nm}": fe_r(it, W) for it, lab in (("T01", "사업체"), ("T02", "종사자"), ("T04", "출하")) for nm, W in (("B0", W0), ("V1a", W1))}
def var_share(it, endw, sp, seed=777):
    v = []; gg = []
    for k in K34:
        try:
            v.append((lmx(k, it, endw) - lmx(k, it, [2008, 2009])) / sp); gg.append(grp[k])
        except Exception:
            pass
    v = np.array(v); gg = np.array(gg)
    def ws(labels):
        tot = ((v - v.mean()) ** 2).sum(); w = sum(((v[labels == u] - v[labels == u].mean()) ** 2).sum() for u in np.unique(labels)); return w / tot
    obs = ws(gg); rng = np.random.default_rng(seed); sims = np.array([ws(rng.permutation(gg)) for _ in range(5000)])
    return round(float(obs), 4), round(float(sims.mean()), 4), round(float((sims <= obs + 1e-12).mean()), 4), len(v)
res["VAR"] = {f"{lab}_{nm}": var_share(it, e, s) for it, lab in (("T01", "사업체"), ("T04", "출하")) for nm, e, s in (("B0", [2016, 2017], 8.0), ("V1a", [2015, 2016], 7.0))}
# (4)
o9 = {}
for r in json.load(open(os.path.join(H, r"download\kosis9\DT_1F160622_2005_2014.json"), encoding="utf-8")):
    o9[(r["C2"].lstrip("C"), r["ITM_ID"], int(r["PRD_DE"]))] = r["DT"]
m = list(csv.DictReader(open(os.path.join(H, "ksic9_10_map_check.csv"), encoding="utf-8-sig")))
one = [x["ksic10"] for x in m if x["type_9to10"] == "1:1"]
eq = [(c, it) for c in one for it in ("T01", "T02", "T04") if (c, it, 2014) in o9 and o9[(c, it, 2014)] not in ("X", "-") and
      off.get(("10차", c, it, 2014)) not in (None, "X", "-") and abs(float(o9[(c, it, 2014)]) - float(off[("10차", c, it, 2014)])) <= 0.005 * float(off[("10차", c, it, 2014)])]
res["9th_eq_10th_2014_one2one"] = {"n_one2one_codes": len(one), "n_cells_eq": len(eq), "n_cells_total": 3 * len(one)}
lst = {}
for r in json.load(open(os.path.join(H, r"download\kosis10\DT_1FS1001_2007_2020.json"), encoding="utf-8")):
    if r["PRD_DE"] in ("2014", "2015", "2016", "2017"):
        lst.setdefault(r["PRD_DE"], set()).add(r["LST_CHN_DE"])
res["LST_CHN_DE_by_year"] = {k: sorted(v) for k, v in lst.items()}
json.dump(res, open(os.path.join(H, r"verify_0\v0_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k != "random30"}, ensure_ascii=False, indent=1, default=str)); print("random30 ok", res["random30"]["n_ok"])
