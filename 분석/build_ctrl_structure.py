# -*- coding: utf-8 -*-
"""
통제변수 구축 — 산업구조 파생지표 (KOSIS 광업제조업조사)
목적: 심사 반론 "PADI 는 그냥 자본집약도 낮고 영세한 산업을 지목하는 것 아니냐" 에 대응.
      산업의 구조적 속성(자본집약도/부가가치율/사업체규모/노동생산성/노동분배율)을
      통제해도 β_PADI 가 살아남는지 확인한다.

[설계 원칙]
 1) 시간분할: 종속변수 dd_late 는 2017년 이후 값 → 통제변수는 반드시 2016년 이하만 사용.
    predictive_validity.py 의 SPLIT_YEAR=2017, 전반기=2008~2016 과 동일 기준.
 2) 해상도: KOSIS 원자료 '산업별' 코드는 알파벳1 + 숫자5 (예: C10111) = KSIC 세세분류 5자리.
    predictive_agg.csv 의 ksic(5자리)와 1:1 매칭 가능 → 해상도 함정(2자리) 회피.
 3) 전국(시도별=전국) 값만 사용.
"""
import os, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
SRC = os.path.join(BASE, "공모전데이터", "외부데이터", "산업분류별")
FUS = os.path.join(BASE, "분석", "fusion")
OUT = os.path.join(FUS, "ctrl_structure.csv")
SPLIT_YEAR = 2017  # 전반기 = 2008~2016

log = lambda m: print(m, flush=True)


# ── 1. 원자료 로드 (long 형태로 정규화) ─────────────────────────
def load_all():
    """12개 파일 전부 읽어 (ksic, year, item, value) long 테이블 생성.
    전국(시도별=전국) 행만 남긴다. 시도 코드 '00' = 전국."""
    frames = []
    for p in sorted(glob.glob(os.path.join(SRC, "*.csv"))):
        df = pd.read_csv(p, encoding="cp949", dtype=str)
        df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
        # 전국만: DT_1FS1101 계열은 시도명이 '전국' 그대로, 코드는 '00'
        df = df[df["[A]시도별"].astype(str).str.strip() == "00"]
        if df.empty:
            continue
        ycols = [c for c in df.columns if "년" in str(c)]
        m = df.melt(id_vars=["[B]산업별", "산업별", "항목"],
                    value_vars=ycols, var_name="ycol", value_name="value")
        m["year"] = m["ycol"].str.extract(r"(\d{4})").astype(int)
        m["src"] = "DT_1FS1101" if "1FS1101" in os.path.basename(p) else "DT_1FS1001"
        m["file"] = os.path.basename(p)
        frames.append(m)
    d = pd.concat(frames, ignore_index=True)

    # 코드 정규화: 'C10111' → 알파벳 섹션 'C', 5자리 'ksic'
    code = d["[B]산업별"].astype(str).str.strip()
    d["section"] = code.str[0]
    d["ksic"] = code.str[1:]
    d = d[d["ksic"].str.len() == 5]  # 총계('0') 등 제외

    # 값 정리: 'X'(비공개), '-', 빈칸 → NaN
    d["value"] = pd.to_numeric(d["value"], errors="coerce")

    # 항목명 정규화 ('사업체수[개]' → '사업체수')
    d["item"] = d["항목"].astype(str).str.replace(r"\[.*\]", "", regex=True).str.strip()
    return d[["ksic", "section", "산업별", "year", "item", "value", "src", "file"]]


raw = load_all()
log(f"[로드] long rows={len(raw):,} | 연도={sorted(raw.year.unique())}")
log(f"[로드] 항목={sorted(raw.item.unique())}")
log(f"[로드] KSIC5 고유={raw.ksic.nunique()} | 섹션={sorted(raw.section.unique())}")

# 계열 중복 확인 (2020년은 두 계열 모두 존재)
dup = raw.groupby(["ksic", "year", "item"]).src.nunique()
log(f"[중복] 두 계열 동시 존재 셀 수 = {int((dup > 1).sum()):,} (2020년 겹침)")


# ── 2. 전반기(≤2016) 평균 패널 ──────────────────────────────────
early = raw[raw.year < SPLIT_YEAR].copy()
log(f"[전반기] 연도={sorted(early.year.unique())} rows={len(early):,}")

# 같은 (ksic, year, item) 이 중복되면 평균 (전반기는 DT_1FS1001 단일 계열이라 사실상 없음)
panel = (early.groupby(["ksic", "year", "item"], as_index=False)
              .value.mean()
              .pivot(index=["ksic", "year"], columns="item", values="value")
              .reset_index())
panel.columns.name = None

ITEMS = ["사업체수", "종사자수", "급여액", "출하액", "주요생산비", "부가가치", "유형자산 연말잔액"]
missing = [c for c in ITEMS if c not in panel.columns]
if missing:
    raise RuntimeError(f"항목 누락: {missing}")

log(f"[패널] ksic×year = {len(panel):,} | ksic={panel.ksic.nunique()}")
for c in ITEMS:
    log(f"   {c:16s} 결측 {panel[c].isna().sum():5,} / {len(panel):,}")


# ── 3. 연도별 비율 산출 → 연도평균 (ratio-of-means 가 아닌 mean-of-ratios) ──
# 주: 비율을 연도별로 먼저 만든 뒤 평균내면 규모 큰 연도가 과대대표되지 않는다.
#     분모 0 또는 결측이면 그 연도는 자동 제외(NaN).
p = panel.copy()
den = lambda s: s.where(s > 0)  # 0/음수 분모 방어

p["str_capital"] = p["유형자산 연말잔액"] / den(p["출하액"])          # 고정자산부담
p["str_capital_pw"] = p["유형자산 연말잔액"] / den(p["종사자수"])      # 종사자당 유형자산(백만원/명)
p["str_va_ratio"] = p["부가가치"] / den(p["출하액"])                   # 부가가치율
p["str_estab_size"] = p["종사자수"] / den(p["사업체수"])               # 평균 사업체 규모(명)
p["str_labor_prod"] = p["부가가치"] / den(p["종사자수"])               # 종사자당 부가가치(백만원/명)
p["str_wage_share"] = p["급여액"] / den(p["부가가치"])                 # 노동분배율 근사

STRS = ["str_capital", "str_capital_pw", "str_va_ratio", "str_estab_size",
        "str_labor_prod", "str_wage_share"]

ctrl = p.groupby("ksic")[STRS].mean().reset_index()
# 각 지표가 몇 개 연도 관측에 기반했는지
nyr = p.groupby("ksic")[STRS].apply(lambda d: d.notna().sum())
nyr.columns = [c + "_nyr" for c in nyr.columns]
ctrl = ctrl.merge(nyr.reset_index(), on="ksic", how="left")
ctrl["n_years"] = p.groupby("ksic").year.nunique().values
ctrl["period"] = f"{int(early.year.min())}-{int(early.year.max())}"

# 파생 파생: 로그 (분포 왜도 완화용, 회귀에서 선택 사용)
for c in ["str_capital", "str_capital_pw", "str_estab_size", "str_labor_prod"]:
    ctrl["ln_" + c] = np.log(ctrl[c].where(ctrl[c] > 0))

ctrl["k3"] = ctrl.ksic.str[:3]
ctrl["mid"] = ctrl.ksic.str[:2]

os.makedirs(FUS, exist_ok=True)
ctrl.to_csv(OUT, index=False, encoding="utf-8-sig")
log(f"\n[저장] {OUT}  ({len(ctrl):,}행 × {ctrl.shape[1]}열)")


# ── 4. predictive_agg.csv 와 merge 검증 ─────────────────────────
d = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
mg = d.merge(ctrl[["ksic"] + STRS + [c + "_nyr" for c in STRS]], on="ksic", how="left")
log(f"\n[merge] predictive_agg n={len(d)} → 매칭 {mg.str_va_ratio.notna().sum()}/{len(d)}")
for c in STRS:
    log(f"   {c:18s} 유효 {mg[c].notna().sum():2d}/39 | "
        f"mean={mg[c].mean():10.4f} sd={mg[c].std():10.4f}")
miss = mg.loc[mg.str_va_ratio.isna(), "ksic"].tolist()
if miss:
    log(f"   결측 ksic: {miss}")

mg.to_csv(os.path.join(FUS, "with_structure.csv"), index=False, encoding="utf-8-sig")
log(f"[저장] {os.path.join(FUS, 'with_structure.csv')}")
