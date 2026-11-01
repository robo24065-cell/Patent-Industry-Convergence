# -*- coding: utf-8 -*-
"""
모형 강화 1 — 시간분할 예측타당도 + 제3축 탐색
사전 등록 (2026-08-12):
  [P] 예측타당도: 전반기(2008~2016) PADI·경쟁압력 → 후반기(2017~2026) 침체 심도
      현재 결과는 동일 기간 설명이라 순환 위험. 시간을 가르면 진짜 예측 검증이 된다.
      가설: β<0 유지 (전반기 포기율 높은 산업이 후반기에도 깊게 침체)
  [X] 제3축 후보 (융합데이터에서 산출, 전부 사전 명시):
      X1 기술 다각화도 — 산업 내 IPC4 분포의 정규화 섀넌 엔트로피
      X2 출원인 집중도 — 상위 출원인 점유율(HHI 근사)
      X3 한국 상대지위 — KR / (US+JP+EP) 출원비
      X4 실용신안 의존도 — 실용신안 비중 (저품질 IP 의존)
"""
import sqlite3, datetime, os, warnings
import numpy as np
import pandas as pd
import duckdb
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
WORK = os.path.join(BASE, "공모전데이터", "작업데이터")
FUS = os.path.join(BASE, "분석", "fusion")
MACRO = os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표")
PSTART = 2008
SPLIT_YEAR = 2017                      # 전반기/후반기 경계
CUTOFF = datetime.date(2025, 6, 30)
COLLECT = datetime.date(2026, 8, 12)
BUFFER, CMIN, CMAX, MINR = 300, 2005, 2021, 30
BANDS = [(2005, 2009), (2010, 2013), (2014, 2017), (2018, 2021)]
rng = np.random.default_rng(17)
log = lambda m: print(m, flush=True)


def to_date(s): return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
def add_years(d, n):
    try: return d.replace(year=d.year + n)
    except ValueError: return d.replace(year=d.year + n, day=28)
def band_of(y):
    for lo, hi in BANDS:
        if lo <= y <= hi: return f"{lo}-{hi}"
    return None


# ── 이벤트 → 노출 ───────────────────────────────────────────────
snap = max([os.path.join(WORK, f) for f in os.listdir(WORK) if f.startswith("registry_snapshot")],
           key=os.path.getmtime)
con = sqlite3.connect(snap)
rows = con.execute("""SELECT rgst_no, ksic, reg_date, exp_date, last_annl FROM registry
  WHERE status='OK' AND source LIKE 'datago%' AND reg_date IS NOT NULL
    AND exp_date IS NOT NULL AND last_annl IS NOT NULL""").fetchall()
con.close()
attrs = pd.read_parquet(os.path.join(FUS, "kr_value_attrs.parquet"))
fam = dict(zip(attrs.rgst_no, attrs.has_family))

er = []
for rgst, ksic, rd, ed, la in rows:
    reg, exp = to_date(rd), to_date(ed)
    if exp <= reg or not (CMIN <= reg.year <= CMAX): continue
    typ = "UM" if rgst.startswith("2") else "PAT"
    M, la = round((exp - reg).days / 365.25), int(la)
    cov = add_years(reg, la)
    if cov >= exp: ev, age = "expire", None
    elif (COLLECT - cov).days > BUFFER: ev, age = "abandon", la + 1
    else: ev, age = "alive", None
    near = (ev == "abandon" and age is not None and age > M - 2)
    b = band_of(reg.year)
    f = int(fam.get(rgst, 0))
    for a in range(2, min(int(M), 20) + 1):
        due = add_years(reg, a - 1)
        if due > CUTOFF or due.year < PSTART or la < a - 1: continue
        e = 1 if (ev == "abandon" and age == a and not near) else 0
        er.append((ksic, typ, b, a, due.year, e, f))
expo = pd.DataFrame(er, columns=["ksic", "type", "band", "age", "year", "event", "fam"])
log(f"노출 {len(expo):,} | 전반기(~{SPLIT_YEAR-1}) {int((expo.year < SPLIT_YEAR).sum()):,} "
    f"후반기 {int((expo.year >= SPLIT_YEAR).sum()):,}")


def padi_by(sub):
    """기간 부분표본으로 h 추정 후 산업별 O/E 산출 (해당 기간 정보만 사용)"""
    g = sub.groupby(["type", "band", "age"]).agg(O=("event", "sum"), R=("event", "size"))
    p = sub.groupby(["type", "age"]).agg(O=("event", "sum"), R=("event", "size"))
    g, p = g[g.R >= MINR], p[p.R >= MINR]
    hm, hp = (g.O / g.R).to_dict(), (p.O / p.R).to_dict()
    s = sub.copy()
    s["h"] = [hm.get((t, b, a), hp.get((t, a), np.nan)) for t, b, a in zip(s.type, s.band, s.age)]
    s = s.dropna(subset=["h"])
    out = s.groupby("ksic").apply(lambda d: d.event.sum() / d.h.sum(), include_groups=False)
    out_lo = s[s.fam == 0].groupby("ksic").apply(lambda d: d.event.sum() / d.h.sum(), include_groups=False)
    return out.rename("padi"), out_lo.rename("padi_lo")


早 = expo[expo.year < SPLIT_YEAR]
p_early, plo_early = padi_by(早)
p_all, plo_all = padi_by(expo)

# ── 경기변수: 기간별 침체 심도 ──────────────────────────────────
k = pd.read_csv(os.path.join(MACRO, "kosis_광공업생산지수_월별.csv"), encoding="utf-8-sig", dtype=str)
k["val"] = pd.to_numeric(k.값, errors="coerce"); k = k.dropna(subset=["val"])
k["yr"] = k.연월.str[:4].astype(int)
k["mi"] = (k.yr - PSTART) * 12 + k.연월.str[4:6].astype(int) - 1
k["k3"] = k.산업코드.str[1:]
k = k[(k.k3.str.len() == 3) & (k.항목명.str.contains("생산"))]
kq = k.groupby(["k3", "mi", "yr"]).val.mean().reset_index().rename(columns={"val": "prod"}).sort_values(["k3", "mi"])


def maxdd(g):
    pr = g["prod"].values
    return float((pr / np.maximum.accumulate(pr) - 1).min() * 100)


dd_late = kq[kq.yr >= SPLIT_YEAR].groupby("k3").apply(maxdd, include_groups=False).rename("dd_late")
dd_early = kq[kq.yr < SPLIT_YEAR].groupby("k3").apply(maxdd, include_groups=False).rename("dd_early")
dd_all = kq.groupby("k3").apply(maxdd, include_groups=False).rename("dd_all")

# ── 제3축 후보 산출 ─────────────────────────────────────────────
log("\n제3축 후보 산출 중...")
d = duckdb.connect(); d.execute("SET threads TO 6")
src = os.path.join(BASE, "공모전데이터", "융합데이터_*", "*.csv").replace(os.sep, "/")
rd = ("read_csv('" + src + "', delim='\t', quote='\"', escape='\"', header=true, "
      "ignore_errors=true, sample_size=20000)")
# X1 기술 다각화 (IPC4 엔트로피, KR 전반기 출원 기준) + X4 실용신안 비중
x14 = d.execute(f"""
  SELECT trim(CAST("세세분류 코드" AS VARCHAR)) AS ksic,
         substr(trim("메인 ipc"),1,4) AS ipc4,
         CASE WHEN substr(trim(CAST("출원번호" AS VARCHAR)),1,2)='20' THEN 1 ELSE 0 END AS um,
         count(*) AS n
  FROM {rd}
  WHERE trim("국가코드")='KR' AND CAST("출원연도" AS INT) BETWEEN 2005 AND {SPLIT_YEAR-1}
    AND "메인 ipc" IS NOT NULL AND trim("메인 ipc")<>''
  GROUP BY 1,2,3""").fetchdf()
ent = []
for ks, g in x14.groupby("ksic"):
    tot = g.n.sum()
    pk = g.groupby("ipc4").n.sum() / tot
    K = len(pk)
    H = -(pk * np.log(pk)).sum() / np.log(K) if K > 1 else 0
    ent.append({"ksic": ks, "ipc_entropy": H, "n_ipc4": K, "um_share": g[g.um == 1].n.sum() / tot})
X1 = pd.DataFrame(ent)
# X2 출원인 집중도
x2 = d.execute(f"""
  SELECT ksic, max(c)*1.0/sum(c) AS top1_share, sum(c*c)*1.0/(sum(c)*sum(c)) AS hhi FROM (
    SELECT trim(CAST("세세분류 코드" AS VARCHAR)) AS ksic, "출원인명" AS app, count(*) AS c
    FROM {rd} WHERE trim("국가코드")='KR' AND CAST("출원연도" AS INT) BETWEEN 2005 AND {SPLIT_YEAR-1}
    GROUP BY 1,2) GROUP BY 1""").fetchdf()
log(f"  X1 엔트로피 {len(X1)}산업, X2 집중도 {len(x2)}산업")

comp = pd.read_csv(os.path.join(FUS, "competition_panel.csv"), encoding="utf-8-sig", dtype={"ksic": str})
comp_e = comp[(comp.yr >= 2005) & (comp.yr < SPLIT_YEAR)].groupby("ksic").agg(
    cn_share_e=("cn_share", "mean"), cn_growth_e=("cn_growth", "mean"),
    kr_rel_e=("kr_rel_adv", "mean")).reset_index()

# ── 결합 ────────────────────────────────────────────────────────
A = pd.DataFrame({"padi_e": p_early, "padi_lo_e": plo_early}).reset_index()
A = A.merge(pd.DataFrame({"padi_a": p_all, "padi_lo_a": plo_all}).reset_index(), on="ksic")
A["k3"] = A.ksic.str[:3]
A = A.merge(dd_late.reset_index(), on="k3").merge(dd_early.reset_index(), on="k3").merge(dd_all.reset_index(), on="k3")
A = A.merge(X1, on="ksic", how="left").merge(x2, on="ksic", how="left").merge(comp_e, on="ksic", how="left")
A["mid"] = A.ksic.str[:2]
A = A.dropna(subset=["padi_e", "dd_late"])
log(f"분석 표본 {len(A)}산업")


def boot(x, y, groups, n=3000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y)); x, y, g = x[ok], y[ok], np.asarray(groups)[ok]
    r0 = np.corrcoef(x, y)[0, 1]; bs = []
    uq = np.unique(g)
    for _ in range(n):
        pick = rng.choice(uq, len(uq), replace=True)
        i = np.concatenate([np.where(g == c)[0] for c in pick])
        if len(i) > 3 and np.std(x[i]) > 1e-9 and np.std(y[i]) > 1e-9:
            bs.append(np.corrcoef(x[i], y[i])[0, 1])
    return r0, np.percentile(bs, [2.5, 97.5])


log("\n" + "=" * 66)
log(f"[P] 예측타당도 — 전반기(~{SPLIT_YEAR-1}) 지표 → 후반기({SPLIT_YEAR}~) 침체 심도")
for x, lab in [("padi_e", "전반기 PADI"), ("padi_lo_e", "전반기 PADI_lo(국내전용)")]:
    r, ci = boot(A[x], A.dd_late, A.mid)
    rs = A[x].corr(A.dd_late, method="spearman")
    sig = "유의" if ci[1] < 0 else "비유의"
    log(f"  corr({lab:22s}, 후반기 최대낙폭) = {r:+.3f} 클러스터CI[{ci[0]:+.3f},{ci[1]:+.3f}] {sig} ρ={rs:+.3f}")
r, ci = boot(A.padi_e, A.dd_early, A.mid)
log(f"  (참고) corr(전반기 PADI, 전반기 낙폭) = {r:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}]")

log(f"\n  [P-2] 예측 회귀: 후반기 낙폭 ~ 전반기 PADI + 전반기 중국증가율")
reg = A.dropna(subset=["dd_late", "padi_e", "cn_growth_e"])
f = smf.ols("dd_late ~ padi_e + cn_growth_e", data=reg).fit(
    cov_type="cluster", cov_kwds={"groups": reg.mid})
log("    " + f.summary2().tables[1].round(3).to_string().replace("\n", "\n    "))
log(f"    R²={f.rsquared:.3f}  n={int(f.nobs)}")

log("\n" + "=" * 66)
log("[X] 제3축 후보 — 후반기 낙폭과의 단순 상관 (전부 보고)")
cands = [("ipc_entropy", "기술 다각화(IPC 엔트로피)"), ("n_ipc4", "IPC 분야 수"),
         ("um_share", "실용신안 비중"), ("top1_share", "1위 출원인 점유율"),
         ("hhi", "출원인 HHI"), ("kr_rel_e", "한국 상대지위"),
         ("cn_share_e", "중국 비중"), ("cn_growth_e", "중국 증가율")]
res = []
for c, lab in cands:
    if c not in A or A[c].notna().sum() < 20: continue
    r, ci = boot(A[c], A.dd_late, A.mid)
    sig = "유의" if (ci[0] > 0 or ci[1] < 0) else ""
    res.append({"축": lab, "r": r, "CI_lo": ci[0], "CI_hi": ci[1], "판정": sig})
    log(f"  {lab:24s} r={r:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] {sig}")

log("\n  [X-2] 유의 후보를 2축 모형에 추가 시 ΔR²")
base = smf.ols("dd_late ~ padi_e + cn_growth_e", data=reg).fit()
for c, lab in cands:
    if c in ("cn_share_e", "cn_growth_e") or c not in A: continue
    d2 = A.dropna(subset=["dd_late", "padi_e", "cn_growth_e", c])
    if len(d2) < 25: continue
    b2 = smf.ols("dd_late ~ padi_e + cn_growth_e", data=d2).fit()
    f2 = smf.ols(f"dd_late ~ padi_e + cn_growth_e + {c}", data=d2).fit(
        cov_type="cluster", cov_kwds={"groups": d2.mid})
    log(f"    +{lab:24s} ΔR²={f2.rsquared-b2.rsquared:+.3f} β={f2.params[c]:+8.2f} "
        f"t={f2.tvalues[c]:+5.2f} p={f2.pvalues[c]:.3f}")

A.to_csv(os.path.join(FUS, "predictive_agg.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(res).to_csv(os.path.join(FUS, "third_axis.csv"), index=False, encoding="utf-8-sig")
log(f"\n저장 → {FUS}")
