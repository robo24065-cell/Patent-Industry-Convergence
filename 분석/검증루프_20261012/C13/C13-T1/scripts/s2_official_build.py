# -*- coding: utf-8 -*-
"""C13-T1 s2: 공식 심판·존속 통계 정리와 (a)·(b) 점검.
입력(읽기만): download/moip_unzip/*(연보 엑셀), cache/official_trials_raw.csv(s1), PDF 판독값(2010·2014, 아래 MANUAL).
출력: cache/official_trials.csv(연도×종류×권리 최종값과 출처), cache/interpartes_by_edition.csv, cache/inforce_by_edition.csv,
      cache/pharma_field.csv, cache/s2_checks.json"""
import sys, os, glob, re, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = os.path.join(W, "download", "moip_unzip")
PUB = {"f_115_2": 2007, "f_139_2": 2008, "f_141_2": 2009, "f_156_2": 2010, "f_176_1": 2012, "f_186_3": 2013, "f_205_1": 2014,
       "idx_2015": 2016, "f_250_1": 2017, "f_267_2": 2018, "f_283_2": 2019, "f_312_2": 2020, "f_328_2": 2021, "f_360_1": 2022,
       "f_380_2": 2023, "f_400_2": 2024, "f_416_2": 2025, "f_437_1": 2026}   # 폴더 → 발행 연도
CHK = {}

def num(v):
    try:
        return float(str(v).replace(",", ""))
    except Exception:
        return np.nan

def multi_year(sheet_pred, section_pred, out_name):
    """여러 해 표에서 section(첫 열) 안의 특허·실용 행을 연도별로 뽑는다."""
    rows = []
    for d in sorted(glob.glob(os.path.join(U, "*"))):
        tag = os.path.basename(d)
        for f in glob.glob(os.path.join(d, "*")):
            b = os.path.basename(f)
            if not sheet_pred[0](b):
                continue
            xl = pd.ExcelFile(f)
            for sh in xl.sheet_names:
                df = xl.parse(sh, header=None)
                if df.shape[1] < 3:
                    continue
                head = " ".join(str(x) for x in df.iloc[:3, 0].tolist())
                if not sheet_pred[1](head):
                    continue
                # 연도 머리행
                yrow = None
                for i in range(min(10, len(df))):
                    ys = [num(x) for x in df.iloc[i, 2:].tolist()]
                    if sum(1 for y in ys if 1990 < (y if y == y else 0) < 2030) >= 3:
                        yrow = i; break
                if yrow is None:
                    continue
                years = {j: int(num(df.iat[yrow, j])) for j in range(2, df.shape[1]) if 1990 < (num(df.iat[yrow, j]) if num(df.iat[yrow, j]) == num(df.iat[yrow, j]) else 0) < 2030}
                sec = ""
                for i in range(yrow + 1, len(df)):
                    c0 = str(df.iat[i, 0]) if pd.notna(df.iat[i, 0]) else ""
                    c1 = str(df.iat[i, 1]) if pd.notna(df.iat[i, 1]) else ""
                    if c0.strip():
                        sec = c0.replace("\n", " ")
                    if not section_pred(sec, c0, c1):
                        continue
                    right = "특허" if c1.replace(" ", "").startswith("특") else ("실용" if c1.replace(" ", "").startswith("실용") else None)
                    if right is None:
                        continue
                    for j, y in years.items():
                        rows.append(dict(edition=PUB[tag], src=tag, sheet=sh.strip(), year=y, right=right, value=num(df.iat[i, j]),
                                         section=sec[:20]))
    R = pd.DataFrame(rows).drop_duplicates(["edition", "year", "right"], keep="first")
    R.to_csv(os.path.join(W, "cache", out_name), index=False, encoding="utf-8-sig")
    return R

# ---------------------------------------------------------------- 1) 당사자계 여러 해(특허·실용)
IP = multi_year((lambda b: "심판" in b, lambda h: ("심판청구 현황" in h or "심판 종류별 심판청구" in h or "5.1. 심판청구" in h or "Requests for trials by year" in h or "types of trial" in h)),
                lambda sec, c0, c1: sec.strip().startswith("당사자") or sec.strip().startswith("Inter"), "interpartes_by_edition.csv")
ipv = IP.pivot_table(index=["year", "right"], columns="edition", values="value")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("당사자계 청구(특허·실용) 판별:"); print(ipv.loc[ipv.index.get_level_values(0) >= 2005].to_string())

# ---------------------------------------------------------------- 2) 존속 권리(특허·실용) 여러 해
# 1.7 표(연도가 행)는 별도 처리
rows = []
for d in sorted(glob.glob(os.path.join(U, "*"))):
    tag = os.path.basename(d)
    for f in glob.glob(os.path.join(d, "*")):
        if "총괄" not in os.path.basename(f):
            continue
        xl = pd.ExcelFile(f)
        for sh in xl.sheet_names:
            if "존속" not in sh:
                continue
            df = xl.parse(sh, header=None)
            for i in range(len(df)):
                y = num(df.iat[i, 0])
                if 1990 < (y if y == y else 0) < 2030:
                    rows.append(dict(edition=PUB[tag], src=tag, year=int(y), right="특허", value=num(df.iat[i, 1])))
                    rows.append(dict(edition=PUB[tag], src=tag, year=int(y), right="실용", value=num(df.iat[i, 2])))
INF2 = pd.DataFrame(rows)
# 등록 시트의 '소계' 행 값(내외국인 합)만 남기기: multi_year는 내국인·외국인·소계 행을 모두 특허/실용으로 볼 수 있으므로 1.7 표를 주로 쓴다
INF2.to_csv(os.path.join(W, "cache", "inforce_17_by_edition.csv"), index=False, encoding="utf-8-sig")
infv = INF2.pivot_table(index=["year", "right"], columns="edition", values="value")
print("존속 권리(1.7 표):"); print(infv.loc[infv.index.get_level_values(0) >= 2005].to_string())

# 옛 판(존속권리 건수 시트: 소계 행) — 특허/실용 '소계'
rows = []
for d in sorted(glob.glob(os.path.join(U, "*"))):
    tag = os.path.basename(d)
    for f in glob.glob(os.path.join(d, "*")):
        if "등록" not in os.path.basename(f):
            continue
        xl = pd.ExcelFile(f)
        for sh in xl.sheet_names:
            if "존속" not in sh:
                continue
            df = xl.parse(sh, header=None)
            if df.shape[1] < 3:
                continue
            head = " ".join(str(x) for x in df.iloc[:3, 0].tolist())
            if not ("in force" in head and ("존속권리 건수" in head or "내·외국인별 존속권리" in head or "Intellectual property rights in force" in head)):
                continue
            yrow = next((i for i in range(6) if sum(1 for x in df.iloc[i, 2:] if 1990 < (num(x) if num(x) == num(x) else 0) < 2030) >= 3), None)
            if yrow is None:
                continue
            years = {j: int(num(df.iat[yrow, j])) for j in range(2, df.shape[1]) if 1990 < (num(df.iat[yrow, j]) if num(df.iat[yrow, j]) == num(df.iat[yrow, j]) else 0) < 2030}
            right = None
            for i in range(yrow + 1, len(df)):
                c0 = str(df.iat[i, 0]).replace(" ", "") if pd.notna(df.iat[i, 0]) else ""
                c1 = str(df.iat[i, 1]).replace(" ", "") if pd.notna(df.iat[i, 1]) else ""
                if c0.startswith("특허"): right = "특허"
                elif c0.startswith("실용"): right = "실용"
                elif c0: right = None
                if right and c1.startswith("소계"):
                    for j, y in years.items():
                        rows.append(dict(edition=PUB[tag], src=tag, year=y, right=right, value=num(df.iat[i, j])))
INF3 = pd.DataFrame(rows).drop_duplicates(["edition", "year", "right"])
INF3.to_csv(os.path.join(W, "cache", "inforce_reg_by_edition.csv"), index=False, encoding="utf-8-sig")
infall = pd.concat([INF2.assign(tab="1.7"), INF3.assign(tab="등록")])
rev = infall.groupby(["year", "right"]).value.agg(["min", "max", "count"])
rev["rel_range"] = (rev["max"] - rev["min"]) / rev["max"]
CHK["inforce_revision_max_rel_range_2005_2025"] = float(rev.loc[rev.index.get_level_values(0) >= 2005, "rel_range"].max())
CHK["inforce_revision_rows_gt_0.5pct"] = rev[(rev.rel_range > 0.005) & (rev.index.get_level_values(0) >= 2005)].reset_index().to_dict("records")
INFF = INF2[INF2.edition == 2026].set_index(["year", "right"]).value  # 2025판(2026 발행) 최종
CHK["inforce_final_edition"] = "2025 지식재산통계연보(2026 발행) 1.7 연도별 존속권리"

# ---------------------------------------------------------------- 3) 종류별 청구(무효·권리범위) 최종값
R = pd.read_csv(os.path.join(W, "cache", "official_trials_raw.csv"))
R = R[R.trial.isin(["무효", "권리범위", "정정"])]
O = R.groupby(["year", "trial", "right"]).agg(request=("request", "sum"), src=("src", "first"), n=("request", "size")).reset_index()
assert O.n.max() == 1, O[O.n > 1]
O["basis"] = "연보 엑셀 5.2.2/7./5. 심판종류별 청구 및 처리 '청구'"
MANUAL = [  # PDF 판독(쪽 번호는 PDF 쪽). 2010: 지식재산통계연보(2011년 발행)-정정본 p146, 2014: 지식재산통계연보(2014)_pdf1 p145
    (2010, "무효", "특허", 651, "f_170_2 p146"), (2010, "무효", "실용", 120, "f_170_2 p146"),
    (2010, "권리범위", "특허", 418, "f_170_2 p146"), (2010, "권리범위", "실용", 132, "f_170_2 p146"),
    (2010, "정정", "특허", 95, "f_170_2 p147"), (2010, "정정", "실용", 5, "f_170_2 p147"),
    (2014, "무효", "특허", 687, "f_222_2 p145"), (2014, "무효", "실용", 64, "f_222_2 p145"),
    (2014, "권리범위", "특허", 385, "f_222_2 p145"), (2014, "권리범위", "실용", 64, "f_222_2 p145")]
for y, t, r, v, s in MANUAL:
    O = pd.concat([O, pd.DataFrame([dict(year=y, trial=t, right=r, request=float(v), src=s, n=1, basis="연보 PDF 판독(좌표 행)")])])
O = O.sort_values(["year", "trial", "right"]).reset_index(drop=True)

# 당사자계 합 대조: 당사자계(최신판 값) vs 무효+권리범위
ipl = IP.sort_values("edition").groupby(["year", "right"]).value.last()
ipf = IP.sort_values("edition").groupby(["year", "right"]).value.first()
chk = []
for (y, r), g in O[O.trial.isin(["무효", "권리범위"])].groupby(["year", "right"]):
    s = g.set_index("trial").request
    tot = ipl.get((y, r), np.nan)
    chk.append(dict(year=y, right=r, inv=s.get("무효"), scope=s.get("권리범위"), interpartes_latest=tot,
                    interpartes_first=ipf.get((y, r), np.nan), resid=tot - s.sum() if tot == tot else np.nan))
CK = pd.DataFrame(chk)
print("당사자계 − (무효+권리범위):"); print(CK.to_string())
CHK["interpartes_identity"] = CK.to_dict("records")
# 2022 권리범위 값 점검: 2016과 같은 값(632/47)인지
o16 = O[(O.year == 2016) & (O.trial == "권리범위")].set_index("right").request.to_dict()
o22 = O[(O.year == 2022) & (O.trial == "권리범위")].set_index("right").request.to_dict()
CHK["scope_2022_equals_2016"] = (o16 == o22)
if o16 == o22:
    for r in ["특허", "실용"]:
        inv22 = float(O[(O.year == 2022) & (O.trial == "무효") & (O.right == r)].request.iloc[0])
        tot22 = float(ipl[(2022, r)])
        new = tot22 - inv22
        CHK[f"scope_2022_{r}_corrected"] = dict(xlsx=o22[r], interpartes=tot22, invalid=inv22, corrected=new)
        O.loc[(O.year == 2022) & (O.trial == "권리범위") & (O.right == r), ["request", "basis"]] = [new, "당사자계 합(2023–2026판) − 무효 (2022판 엑셀 청구 열 632/47은 2016년 값 복제 오류로 판단)"]
O.to_csv(os.path.join(W, "cache", "official_trials.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 4) WIPO 의약품 분야 심판청구(전 종류, 특허) — 대표성 보조
rows = []
for d in sorted(glob.glob(os.path.join(U, "*"))):
    tag = os.path.basename(d)
    for f in glob.glob(os.path.join(d, "*")):
        if "심판" not in os.path.basename(f):
            continue
        xl = pd.ExcelFile(f)
        for sh in xl.sheet_names:
            if not ("WIPO" in sh or "기술분야" in sh):
                continue
            df = xl.parse(sh, header=None)
            head = " ".join(str(x) for x in df.iloc[:3, 0].tolist())
            m = re.search(r"in (\d{4})", head)
            year = int(m.group(1)) if m else PUB[tag] - 1
            tot = ph = np.nan
            for i in range(len(df)):
                rowtxt = " ".join(str(x) for x in df.iloc[i, :2].tolist())
                vals = [num(x) for x in df.iloc[i, 2:].tolist()]
                vals = [v for v in vals if v == v]
                if not vals:
                    continue
                if ("의약품" in rowtxt or "Pharmaceutical" in rowtxt) and ph != ph:
                    ph = vals
                if ("총계" in rowtxt or "Grand" in rowtxt or ("계" in rowtxt.split()[:1] and "Total" in rowtxt)):
                    tot = vals
            if ph == ph:
                rows.append(dict(edition=PUB[tag], year=year, sheet=sh.strip(), pharma_vals=json.dumps(ph), total_vals=json.dumps(tot) if tot == tot else ""))
PF = pd.DataFrame(rows)
PF.to_csv(os.path.join(W, "cache", "pharma_field.csv"), index=False, encoding="utf-8-sig")
print(PF.to_string()[:3000])
json.dump(CHK, open(os.path.join(W, "cache", "s2_checks.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
piv = O.pivot_table(index="year", columns=["trial", "right"], values="request")
print(piv.to_string())
print({k: v for k, v in CHK.items() if k != "interpartes_identity"})
