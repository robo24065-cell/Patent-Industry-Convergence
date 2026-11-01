# -*- coding: utf-8 -*-
"""
PADI v3 — 지표군 확장 (설계안 v2 §2.2 미구현 지표 완성 + 검정력 강화)
사전 등록 (2026-08-12, 결과 보기 전 확정):
  T1-1 AAI  : 포기 특허 평균 연차의 기대 대비 편차 (음수=젊은 특허까지 버림=위험)
  T1-2 PADI_SME : 법인/개인 층화 지수
  T1-3 PADI_val : 청구항 수 가중 (중요 특허 포기 여부)
  T2-1 월별화 (분기 → 월, 관측치 3배)
가설: AAI는 '수준'이 아닌 '질' 신호이므로 산업 정체성과 직교 → 산업내 시계열 신호를 가질 수 있다
"""
import sqlite3, datetime, os, re
import numpy as np
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
SNAP_DIR = os.path.join(BASE, "공모전데이터", "작업데이터")
OUT = os.path.join(BASE, "분석", "v3")
os.makedirs(OUT, exist_ok=True)

CUTOFF = datetime.date(2025, 6, 30)
COLLECT = datetime.date(2026, 8, 12)
BUFFER = 300
CMIN, CMAX = 2005, 2021
PSTART = 2008
BANDS = [(2005, 2009), (2010, 2013), (2014, 2017), (2018, 2021)]
MINR = 30
log = lambda m: print(m, flush=True)


def to_date(s):
    return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def add_years(d, n):
    try:
        return d.replace(year=d.year + n)
    except ValueError:
        return d.replace(year=d.year + n, day=28)


def band_of(y):
    for lo, hi in BANDS:
        if lo <= y <= hi:
            return f"{lo}-{hi}"
    return None


# ── 법인 사전 (PADI_SME용) ──────────────────────────────────────
def load_corp():
    p = os.path.join(BASE, "공모전데이터", "외부데이터", "Corporate_20260720", "CORP_APPLICANT.txt")
    norm = lambda s: re.sub(r"주식회사|\(주\)|\(유\)|유한회사|\s+", "", s).strip()
    names = set()
    with open(p, encoding="utf-8") as f:
        next(f)
        for ln in f:
            parts = ln.rstrip("\n").split("\u00b6")
            if len(parts) >= 2 and parts[1].strip():
                names.add(norm(parts[1]))
    return names, norm


# ── 이벤트 + 특허 속성 ──────────────────────────────────────────
snap = max([os.path.join(SNAP_DIR, f) for f in os.listdir(SNAP_DIR)
            if f.startswith("registry_snapshot")], key=os.path.getmtime)
log(f"스냅샷: {os.path.basename(snap)}")
con = sqlite3.connect(snap)
rows = con.execute("""SELECT rgst_no, ksic, appl_no, reg_date, exp_date, last_annl, claim_count
    FROM registry WHERE status='OK' AND source LIKE 'datago%'
      AND reg_date IS NOT NULL AND exp_date IS NOT NULL AND last_annl IS NOT NULL""").fetchall()
con.close()

# 출원인 (제공데이터에서)
import duckdb
d = duckdb.connect()
app = d.execute(f"""SELECT trim(CAST(등록번호 AS VARCHAR)) rgst, 출원인명
    FROM read_parquet('{os.path.join(SNAP_DIR, "kr_registered.parquet").replace(chr(92), "/")}')""").fetchall()
appmap = {r[0]: (r[1] or "") for r in app}
corp_names, norm = load_corp()
log(f"법인사전 {len(corp_names):,} / 출원인 매핑 {len(appmap):,}")

recs = []
for rgst, ksic, appl, rd, ed, la, cc in rows:
    reg, exp = to_date(rd), to_date(ed)
    if exp <= reg or not (CMIN <= reg.year <= CMAX):
        continue
    typ = "UM" if rgst.startswith("2") else "PAT"
    M = round((exp - reg).days / 365.25)
    la = int(la)
    cov = add_years(reg, la)
    if cov >= exp:
        ev, age = "expire", None
    elif (COLLECT - cov).days > BUFFER:
        ev, age = "abandon", la + 1
    else:
        ev, age = "alive", None
    near = (ev == "abandon" and age is not None and age > M - 2)
    an = appmap.get(rgst, "")
    is_corp = norm(re.split(r"[|;,]", an)[0]) in corp_names if an else False
    recs.append((rgst, ksic, typ, reg, exp, M, la, ev, age, near,
                 band_of(reg.year), int(cc or 0), is_corp))

df = pd.DataFrame(recs, columns=["rgst", "ksic", "type", "reg", "exp", "M", "la",
                                 "event", "age", "near", "band", "claims", "corp"])
log(f"이벤트 {len(df):,}건 | 법인 {df.corp.mean()*100:.1f}% | 청구항 중앙값 {df.claims.median():.0f}")

# ── 노출 레코드 (월 단위) ───────────────────────────────────────
def month_index(dt):
    return (dt.year - PSTART) * 12 + dt.month - 1


exp_rows = []
for r in df.itertuples(index=False):
    w = max(int(r.claims), 1)
    for a in range(2, min(int(r.M), 20) + 1):
        due = add_years(r.reg, a - 1)
        if due > CUTOFF or due.year < PSTART:
            continue
        if r.la < a - 1:
            continue
        ev = 1 if (r.event == "abandon" and r.age == a and not r.near) else 0
        exp_rows.append((r.ksic, r.type, r.band, a, month_index(due), due.year,
                         ev, w, int(r.corp)))
expo = pd.DataFrame(exp_rows, columns=["ksic", "type", "band", "age", "mi", "year",
                                       "event", "w", "corp"])
expo["qi"] = expo.mi // 3
log(f"노출 레코드 {len(expo):,} (월 {expo.mi.nunique()}개)")

# ── 기저위험률 (코호트밴드 조건부) ──────────────────────────────
def hazard(src, weight=None):
    if weight is None:
        g = src.groupby(["type", "band", "age"]).agg(O=("event", "sum"), R=("event", "size"))
        p = src.groupby(["type", "age"]).agg(O=("event", "sum"), R=("event", "size"))
    else:
        src = src.assign(_o=src.event * src[weight], _r=src[weight])
        g = src.groupby(["type", "band", "age"]).agg(O=("_o", "sum"), R=("_r", "sum"))
        p = src.groupby(["type", "age"]).agg(O=("_o", "sum"), R=("_r", "sum"))
    g = g[g.R >= MINR]; p = p[p.R >= MINR]
    return (g.O / g.R).to_dict(), (p.O / p.R).to_dict()


hm, hp = hazard(expo)
expo["h"] = [hm.get((t, b, a), hp.get((t, a), np.nan)) for t, b, a in zip(expo.type, expo.band, expo.age)]
hmw, hpw = hazard(expo, "w")
expo["hw"] = [hmw.get((t, b, a), hpw.get((t, a), np.nan)) for t, b, a in zip(expo.type, expo.band, expo.age)]
expo = expo.dropna(subset=["h"])

# 기간효과 δ (월)
dlt = expo.groupby("mi").apply(lambda g: g.event.sum() / g.h.sum(), include_groups=False)
dlt = dlt.rolling(12, center=True, min_periods=1).mean().rename("delta")
expo = expo.merge(dlt, on="mi", how="left")
expo["E"] = expo.h * expo.delta
expo["Ew"] = expo.hw * expo.delta * expo.w
expo["Ow"] = expo.event * expo.w
# 기대 연차 (AAI 기준선): 위험집합에서 기대되는 포기 연차
expo["age_E"] = expo.age * expo.E

# ── 산업×월 집계 → 지표군 ──────────────────────────────────────
def build(unit_col, win_m=12):
    g = expo.groupby([unit_col, "mi"]).agg(
        O=("event", "sum"), E=("E", "sum"), R=("event", "size"),
        Ow=("Ow", "sum"), Ew=("Ew", "sum"),
        ageO=("age", lambda s: np.nan), aE=("age_E", "sum")).reset_index()
    # 관측 포기의 연차 합 (별도 계산)
    ev = expo[expo.event == 1].groupby([unit_col, "mi"]).age.agg(["sum", "count"]).reset_index()
    ev.columns = [unit_col, "mi", "ageO_sum", "ageO_n"]
    g = g.merge(ev, on=[unit_col, "mi"], how="left").fillna({"ageO_sum": 0, "ageO_n": 0})
    # 법인 층화
    sme = expo[expo.corp == 0].groupby([unit_col, "mi"]).agg(O_s=("event", "sum"), E_s=("E", "sum")).reset_index()
    big = expo[expo.corp == 1].groupby([unit_col, "mi"]).agg(O_b=("event", "sum"), E_b=("E", "sum")).reset_index()
    g = g.merge(sme, on=[unit_col, "mi"], how="left").merge(big, on=[unit_col, "mi"], how="left")

    out = []
    for u, gg in g.groupby(unit_col):
        gg = gg.sort_values("mi").set_index("mi")
        gg = gg.reindex(range(int(g.mi.min()), int(g.mi.max()) + 1)).fillna(0.0)
        gg[unit_col] = u
        for c in ["O", "E", "R", "Ow", "Ew", "ageO_sum", "ageO_n", "aE", "O_s", "E_s", "O_b", "E_b"]:
            gg[c + "_r"] = gg[c].rolling(win_m).sum()
        gg["padi"] = gg.O_r / gg.E_r.replace(0, np.nan)
        gg["padi_val"] = gg.Ow_r / gg.Ew_r.replace(0, np.nan)          # T1-3
        gg["padi_sme"] = gg.O_s_r / gg.E_s_r.replace(0, np.nan)         # T1-2
        gg["padi_big"] = gg.O_b_r / gg.E_b_r.replace(0, np.nan)
        # T1-1 AAI: 관측 포기 평균연차 − 기대 포기 평균연차 (음수 = 예상보다 젊은 특허 포기)
        obs_age = gg.ageO_sum_r / gg.ageO_n_r.replace(0, np.nan)
        exp_age = gg.aE_r / gg.E_r.replace(0, np.nan)
        gg["aai"] = -(obs_age - exp_age)                                # 부호 반전: 높을수록 위험
        out.append(gg.reset_index().rename(columns={"index": "mi"}))
    p = pd.concat(out, ignore_index=True)
    p["qi"] = p.mi // 3
    return p


for unit, name in (("ksic", "k5"), (None, "k3")):
    if unit is None:
        expo["k3"] = expo.ksic.str[:3]
        p = build("k3")
        p = p.rename(columns={"k3": "unit"})
    else:
        p = build("ksic").rename(columns={"ksic": "unit"})
    p.to_parquet(os.path.join(OUT, f"padi_v3_{name}.parquet"), index=False)
    log(f"  저장 padi_v3_{name}: {p.unit.nunique()}단위 × {p.mi.nunique()}월")

log("\n[지표 분포 점검]")
p = pd.read_parquet(os.path.join(OUT, "padi_v3_k3.parquet"))
for c in ["padi", "padi_val", "padi_sme", "padi_big", "aai"]:
    s = p[c].replace([np.inf, -np.inf], np.nan).dropna()
    if len(s):
        log(f"  {c:10s} n={len(s):>5,} p5={s.quantile(.05):+.3f} p50={s.median():+.3f} p95={s.quantile(.95):+.3f}")
log(f"저장 → {OUT}")
