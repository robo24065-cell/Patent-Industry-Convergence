# -*- coding: utf-8 -*-
"""C13-T1 verify_0: 포착률과 S1을 별도 코드로 다시 계산하고 반박을 시도한다(s3/s4 코드 import 없음).
- N_reg·제공 존속 수: sqlite3 SQL로 직접 집계(판다스 연산 대신 SQL).
- 공식 값: 연보 엑셀을 openpyxl로 다시 읽는다(판다스 파서와 다른 경로). 2010·2014는 PDF 판독값(당사자계 합과 일치 확인됨)을 상수로.
- AUC: sklearn roc_auc_score(sample_weight).
- 반박: v1 분모 F2(직전 15년 등록 합)·F3(해당 연도 등록), v2 공식 연도 ±1 이동, v3 중복 제거, v4 특허 단독, v5 특허별 예고 집중."""
import os, sys, json, sqlite3, glob, re
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import openpyxl, xlrd
V = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V)
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
REG = os.path.join(BASE, r"수집파이프라인\registry_full.sqlite")
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
INV, SCO = "무효(심판)예고등록", "권리범위(심판)예고등록"
R = {}
# ---------------------------------------------------------------- 1) N_reg (SQL)
q = f"""select substr(rgstCsDate,1,4) y, rgstCsName k, substr(rgst_no,1,2) j, count(*) n, count(distinct rgst_no||rgstCsDate) nd
        from right where rgstCsName in ('{INV}','{SCO}') group by 1,2,3"""
NR = {}
for y, k, j, n_, nd in c.execute(q):
    NR[(int(y), "무효" if k == INV else "권리범위", "특허" if j == "10" else "실용")] = (n_, nd)
# ---------------------------------------------------------------- 2) 제공 존속(SQL)
c.execute("create temp table ext as select rgst_no, min(cast(rgstCsDate as integer)) d from right where rgstCsName in ('소멸등록','권리의 말소등록','말소등록') group by 1")
ALIVE = {}; REGS = {}
for y in range(2006, 2026):
    E = y * 10000 + 1231
    for j, pre in [("특허", "10"), ("실용", "20")]:
        n_ = c.execute(f"""select count(*) from patent p left join ext e on p.rgst_no=e.rgst_no
             where substr(p.rgst_no,1,2)='{pre}' and cast(p.rgstDate as integer)<= {E} and cast(p.cndrtExptnDate as integer) > {E}
             and (e.d is null or e.d > {E})""").fetchone()[0]
        ALIVE[(y, j)] = n_
        REGS[(y, j)] = c.execute(f"select count(*) from patent where substr(rgst_no,1,2)='{pre}' and substr(rgstDate,1,4)='{y}'").fetchone()[0]
R["alive_2022_total"] = ALIVE[(2022, "특허")] + ALIVE[(2022, "실용")]
# ---------------------------------------------------------------- 3) 공식 값(openpyxl/xlrd 직접)
U = os.path.join(W, "download", "moip_unzip")
def rows_of(f, sheetpred):
    out = []
    if f.endswith(".xls"):
        bk = xlrd.open_workbook(f)
        for sh in bk.sheets():
            if sheetpred(sh.name):
                out.append((sh.name, [[sh.cell_value(i, j) for j in range(sh.ncols)] for i in range(sh.nrows)]))
    else:
        bk = openpyxl.load_workbook(f, read_only=True, data_only=True)
        for nm in bk.sheetnames:
            if sheetpred(nm):
                out.append((nm, [list(r) for r in bk[nm].iter_rows(values_only=True)]))
    return out
OFF = {}
PUBY = {"f_267_2": 2017, "f_283_2": 2018, "f_312_2": 2019, "f_328_2": 2020, "f_360_1": 2021, "f_380_2": 2022, "f_400_2": 2023, "f_416_2": 2024, "f_437_1": 2025}
for d in sorted(glob.glob(os.path.join(U, "*"))):
    tag = os.path.basename(d)
    for f in glob.glob(os.path.join(d, "*")):
        if "심판" not in os.path.basename(f):
            continue
        for nm, rows in rows_of(f, lambda s: "종류별 청구" in s or s.strip() == "t05"):
            head = " ".join(str(r[0]) for r in rows[:4] if r)
            m = re.search(r"in (\d{4})", head); yr = int(m.group(1)) if m else PUBY.get(tag)
            sec = None
            for r in rows:
                if not r or len(r) < 3: continue
                c0 = str(r[0] or "").replace("\n", "").replace(" ", ""); c1 = str(r[1] or "").replace(" ", "")
                if c0: sec = "무효" if c0.startswith("무효") else ("권리범위" if c0.startswith("권리범위") else "x")
                if sec in ("무효", "권리범위") and (c1.startswith("특허") or c1.startswith("실용")):
                    v = r[2]
                    v = float(str(v).replace(",", "")) if v not in (None, "", "-") else 0.0
                    OFF[(yr, sec, "특허" if c1.startswith("특허") else "실용")] = v
for k, v in {(2010, "무효", "특허"): 651, (2010, "무효", "실용"): 120, (2010, "권리범위", "특허"): 418, (2010, "권리범위", "실용"): 132,
             (2014, "무효", "특허"): 687, (2014, "무효", "실용"): 64, (2014, "권리범위", "특허"): 385, (2014, "권리범위", "실용"): 64}.items():
    OFF[k] = float(v)
R["off_2022_scope_raw"] = [OFF[(2022, "권리범위", "특허")], OFF[(2022, "권리범위", "실용")]]
OFF[(2022, "권리범위", "특허")] = 674 - OFF[(2022, "무효", "특허")]; OFF[(2022, "권리범위", "실용")] = 25 - OFF[(2022, "무효", "실용")]
# 존속·등록 공식(2025판 총괄 1.5/1.7)
f17 = glob.glob(os.path.join(U, "f_437_1", "001_*"))[0]
INF = {}; OREG = {}
for nm, rows in rows_of(f17, lambda s: "존속" in s or "연도별 등록" in s):
    for r in rows:
        try: y = int(float(r[0]))
        except Exception: continue
        if 1990 < y < 2030:
            tgt = INF if "존속" in nm else OREG
            tgt[(y, "특허")] = float(r[1]); tgt[(y, "실용")] = float(r[2])
# ---------------------------------------------------------------- 4) r 계산(변형)
def share(y, j, how):
    if how == "F1": return ALIVE[(y, j)] / INF[(y, j)]
    if how == "F2":
        a = sum(REGS.get((t, j), 0) for t in range(y - 14, y + 1)); b = sum(OREG.get((t, j), 0) for t in range(y - 14, y + 1)); return a / b
    if how == "F3": return REGS[(y, j)] / OREG[(y, j)]
def r_window(years, ks, js=("특허", "실용"), how="F1", shift=0, dedup=False):
    x = e = 0.0
    for y in years:
        for k in ks:
            for j in js:
                nn = NR.get((y, k, j), (0, 0)); x += nn[1] if dedup else nn[0]
                e += OFF.get((y + shift, k, j), np.nan) * share(y, j, how)
    return x / e
T = list(range(2008, 2023)); F = [2023, 2024]
VAR = {}
for lab, kw in [("base_F1", {}), ("v1_F2", {"how": "F2"}), ("v1_F3", {"how": "F3"}), ("v2_off_y-1", {"shift": -1}), ("v2_off_y+1", {"shift": 1}),
                ("v3_dedup", {"dedup": True}), ("v4_patent_only", {"js": ("특허",)})]:
    Tw = T if kw.get("shift", 0) == 0 else [y for y in T if 2006 <= y + kw["shift"] <= 2025]
    out = {}
    for kl, ks in [("both", ["무효", "권리범위"]), ("inv", ["무효"]), ("scope", ["권리범위"])]:
        rT = r_window(Tw, ks, **kw); rF = r_window(F, ks, **kw)
        out[kl] = dict(r_T=rT, r_F=rF, ratio=rF / rT)
    ri = r_window(list(range(2008, 2025)), ["무효"], **{k: v for k, v in kw.items()}) if kw.get("shift", 0) == 0 else out["inv"]["r_T"]
    rs = r_window(list(range(2008, 2025)), ["권리범위"], **{k: v for k, v in kw.items()}) if kw.get("shift", 0) == 0 else out["scope"]["r_T"]
    out["J1_shift"] = abs(out["both"]["ratio"] - 1) > 0.25
    out["J2_incomplete"] = out["both"]["r_T"] < 0.5 or out["both"]["r_F"] < 0.5
    out["inv_over_scope"] = ri / rs; out["J3_type"] = ri / rs > 2 or ri / rs < 0.5
    VAR[lab] = out
R["variants"] = VAR
R["yearly_r_both_F1"] = {y: r_window([y], ["무효", "권리범위"]) for y in range(2006, 2026)}
# ---------------------------------------------------------------- 5) v5 특허별 예고 집중(같은 해 같은 특허 다수 예고 = 대량 청구)
conc = {}
for y in (2015, 2016, 2022, 2023, 2024):
    rows = c.execute(f"""select rgstCsName, max(cnt), sum(cnt), count(*) from (select rgst_no, rgstCsName, count(*) cnt from right
            where rgstCsName in ('{INV}','{SCO}') and substr(rgstCsDate,1,4)='{y}' group by 1,2) group by 1""").fetchall()
    conc[y] = {("무효" if a == INV else "권리범위"): dict(max_per_patent=b, notices=s_, patents=n_) for a, b, s_, n_ in rows}
R["v5_concentration"] = conc
# ---------------------------------------------------------------- 6) S1 재계산(sklearn)
import pandas as pd
from sklearn.metrics import roc_auc_score
H = os.path.join(BASE, r"분석\모델경쟁_20260927\N8_특허분쟁표적레이더")
P = pd.read_parquet(os.path.join(H, "final_preds_r6.parquet"), columns=["rgst_no", "y", "p_champ", "p_M0"])
fe = dict(c.execute(f"""select rgst_no, min(rgstCsDate) from right where rgstCsName in ('{INV}','{SCO}')
        and rgstCsDate between '20230101' and '20241231' group by 1""").fetchall())
y = P.y.values
fy = np.array([int(fe[r][:4]) if (r in fe) else 0 for r in P.rgst_no])
assert ((fy > 0) == (y == 1)).all()
r23, r24 = R["yearly_r_both_F1"][2023], R["yearly_r_both_F1"][2024]
w = np.ones(len(P)); w[fy == 2023] = 1 / r23; w[fy == 2024] = 1 / r24
w[y == 1] = w[y == 1] / w[y == 1].mean()
def sk(wt):
    a = roc_auc_score(y, P.p_champ, sample_weight=wt); b = roc_auc_score(y, P.p_M0, sample_weight=wt); return a, b, (a - b) / (1 - b)
R["S0_sklearn"] = sk(None); R["S1_sklearn"] = sk(w)
# 반박: S1을 v2(공식 연도 +1 이동) r로
r23b = r_window([2023], ["무효", "권리범위"], shift=1); r24b = r_window([2024], ["무효", "권리범위"], shift=1)
w2 = np.ones(len(P)); w2[fy == 2023] = 1 / r23b; w2[fy == 2024] = 1 / r24b; w2[y == 1] = w2[y == 1] / w2[y == 1].mean()
R["S1_shift+1_sklearn"] = sk(w2); R["r23_r24_shift+1"] = [r23b, r24b]
json.dump(R, open(os.path.join(V, "v0_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=lambda o: o if not isinstance(o, (np.floating, np.integer, np.bool_)) else o.item())
print(json.dumps({k: v for k, v in R.items() if k != "yearly_r_both_F1"}, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print({k: round(v, 3) for k, v in R["yearly_r_both_F1"].items()})
