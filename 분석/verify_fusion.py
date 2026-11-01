# -*- coding: utf-8 -*-
"""
융합데이터 자산 활용 검정 — 사전 등록 (2026-08-12)
  [A] 가치 층화: PADI_hi(패밀리 보유) vs PADI_lo — 신호대잡음비 개선되는가
      진단: within 신뢰도비율 비교 (기존 0.545) + 횡단면 상관 강도 비교
  [B] 경쟁압력 축: CN 비중·증가율을 통제/상호작용 → 잔여 신호 선명화되는가
      + 신규 횡단면 가설: 침체 심도 ~ 중국 경쟁압력 + PADI (분리 가능한가)
  [C] IPC 확산지수: 산업 내 IPC4 중 포기율 상승 비중 (수준 아닌 폭)
"""
import sqlite3, datetime, os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
WORK = os.path.join(BASE, "공모전데이터", "작업데이터")
FUS = os.path.join(BASE, "분석", "fusion")
MACRO = os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표")
PSTART = 2008
CUTOFF = datetime.date(2025, 6, 30)
COLLECT = datetime.date(2026, 8, 12)
BUFFER, CMIN, CMAX, MINR = 300, 2005, 2021, 30
BANDS = [(2005, 2009), (2010, 2013), (2014, 2017), (2018, 2021)]
rng = np.random.default_rng(3)
log = lambda m: print(m, flush=True)


def to_date(s): return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
def add_years(d, n):
    try: return d.replace(year=d.year + n)
    except ValueError: return d.replace(year=d.year + n, day=28)
def band_of(y):
    for lo, hi in BANDS:
        if lo <= y <= hi: return f"{lo}-{hi}"
    return None


# ── 이벤트 + 가치속성 결합 ──────────────────────────────────────
snap = max([os.path.join(WORK, f) for f in os.listdir(WORK) if f.startswith("registry_snapshot")],
           key=os.path.getmtime)
con = sqlite3.connect(snap)
rows = con.execute("""SELECT rgst_no, ksic, reg_date, exp_date, last_annl FROM registry
  WHERE status='OK' AND source LIKE 'datago%' AND reg_date IS NOT NULL
    AND exp_date IS NOT NULL AND last_annl IS NOT NULL""").fetchall()
con.close()
attrs = pd.read_parquet(os.path.join(FUS, "kr_value_attrs.parquet"))
amap = dict(zip(attrs.rgst_no, attrs.has_family))
imap = dict(zip(attrs.rgst_no, attrs.ipc4))

recs = []
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
    recs.append((rgst, ksic, typ, reg, M, la, ev, age, near, band_of(reg.year),
                 int(amap.get(rgst, 0)), imap.get(rgst) or "NA"))
df = pd.DataFrame(recs, columns=["rgst", "ksic", "type", "reg", "M", "la", "event",
                                 "age", "near", "band", "fam", "ipc4"])
log(f"이벤트 {len(df):,}건 | 패밀리 보유 {df.fam.mean()*100:.1f}% | IPC4 {df.ipc4.nunique()}종")

# ── 노출 레코드 (월) ────────────────────────────────────────────
er = []
for r in df.itertuples(index=False):
    for a in range(2, min(int(r.M), 20) + 1):
        due = add_years(r.reg, a - 1)
        if due > CUTOFF or due.year < PSTART or r.la < a - 1: continue
        ev = 1 if (r.event == "abandon" and r.age == a and not r.near) else 0
        er.append((r.ksic, r.type, r.band, a, (due.year - PSTART) * 12 + due.month - 1,
                   due.year, ev, r.fam, r.ipc4))
expo = pd.DataFrame(er, columns=["ksic", "type", "band", "age", "mi", "year", "event", "fam", "ipc4"])
log(f"노출 {len(expo):,}")


def hz(src):
    g = src.groupby(["type", "band", "age"]).agg(O=("event", "sum"), R=("event", "size"))
    p = src.groupby(["type", "age"]).agg(O=("event", "sum"), R=("event", "size"))
    g, p = g[g.R >= MINR], p[p.R >= MINR]
    return (g.O / g.R).to_dict(), (p.O / p.R).to_dict()


hm, hp = hz(expo)
expo["h"] = [hm.get((t, b, a), hp.get((t, a), np.nan)) for t, b, a in zip(expo.type, expo.band, expo.age)]
expo = expo.dropna(subset=["h"])
dlt = expo.groupby("mi").apply(lambda g: g.event.sum() / g.h.sum(), include_groups=False)
dlt = dlt.rolling(12, center=True, min_periods=1).mean().rename("delta")
expo = expo.merge(dlt, on="mi", how="left")
expo["E"] = expo.h * expo.delta


def make_index(sub, win=12):
    g = sub.groupby(["ksic", "mi"]).agg(O=("event", "sum"), E=("E", "sum"), R=("event", "size")).reset_index()
    out = []
    for u, gg in g.groupby("ksic"):
        gg = gg.sort_values("mi").set_index("mi").reindex(
            range(int(g.mi.min()), int(g.mi.max()) + 1)).fillna(0.0)
        gg["ksic"] = u
        for c in ("O", "E", "R"):
            gg[c + "r"] = gg[c].rolling(win).sum()
        gg["padi"] = gg.Or / gg.Er.replace(0, np.nan)
        out.append(gg.reset_index().rename(columns={"index": "mi"}))
    return pd.concat(out, ignore_index=True)


log("\n" + "=" * 66)
log("[A] 가치 층화 — 패밀리 보유가 신호를 정제하는가")
idx = {}
for tag, sub in (("all", expo), ("hi", expo[expo.fam == 1]), ("lo", expo[expo.fam == 0])):
    p = make_index(sub)
    p = p.rename(columns={"padi": f"padi_{tag}"})
    idx[tag] = p[["ksic", "mi", f"padi_{tag}", "Er"]].rename(columns={"Er": f"Er_{tag}"})
    d = p.dropna(subset=[f"padi_{tag}", "Er"])
    d = d[d.Er > 0]
    wv = float(d.groupby("ksic")[f"padi_{tag}"].transform(lambda s: s - s.mean()).pow(2).mean())
    nv = float((d[f"padi_{tag}"] / d.Er).mean())
    rel = max(0.0, wv - nv) / wv if wv > 0 else np.nan
    log(f"  {tag:3s}: within분산 {wv:.4f} 잡음 {nv:.4f} → **신뢰도비율 {rel:.3f}**")

P = idx["all"].merge(idx["hi"], on=["ksic", "mi"]).merge(idx["lo"], on=["ksic", "mi"])

# ── 경기변수 ────────────────────────────────────────────────────
k = pd.read_csv(os.path.join(MACRO, "kosis_광공업생산지수_월별.csv"), encoding="utf-8-sig", dtype=str)
k["val"] = pd.to_numeric(k.값, errors="coerce"); k = k.dropna(subset=["val"])
k["mi"] = (k.연월.str[:4].astype(int) - PSTART) * 12 + k.연월.str[4:6].astype(int) - 1
k["k3"] = k.산업코드.str[1:]
k = k[(k.k3.str.len() == 3) & (k.항목명.str.contains("생산"))]
kq = k.groupby(["k3", "mi"]).val.mean().reset_index().rename(columns={"val": "prod"}).sort_values(["k3", "mi"])
kq["yoy"] = kq.groupby("k3")["prod"].pct_change(12) * 100

P["k3"] = P.ksic.str[:3]
M = P.merge(kq[["k3", "mi", "prod", "yoy"]], on=["k3", "mi"], how="inner")
M = M.replace([np.inf, -np.inf], np.nan).sort_values(["ksic", "mi"])
M["ti"] = M.mi.astype(int)

# 산업별 침체 심도
sev = []
for u, g in M.groupby("ksic"):
    g = g.sort_values("mi").dropna(subset=["prod"])
    pr = g["prod"].values
    dd = (pr / np.maximum.accumulate(pr) - 1) * 100
    sev.append({"ksic": u, "maxdd": dd.min()})
sev = pd.DataFrame(sev)

agg = M.groupby("ksic").agg(padi_all=("padi_all", "mean"), padi_hi=("padi_hi", "mean"),
                            padi_lo=("padi_lo", "mean")).reset_index().merge(sev, on="ksic")
comp = pd.read_csv(os.path.join(FUS, "competition_agg.csv"), encoding="utf-8-sig", dtype={"ksic": str})
agg = agg.merge(comp, on="ksic", how="left")
agg["mid"] = agg.ksic.str[:2]


def boot(x, y, groups=None, n=3000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    g = np.asarray(groups)[ok] if groups is not None else None
    r0 = np.corrcoef(x, y)[0, 1]; bs = []
    for _ in range(n):
        if g is None:
            i = rng.integers(0, len(x), len(x))
        else:
            uq = np.unique(g); pick = rng.choice(uq, len(uq), replace=True)
            i = np.concatenate([np.where(g == c)[0] for c in pick])
        if len(i) > 3 and np.std(x[i]) > 1e-9 and np.std(y[i]) > 1e-9:
            bs.append(np.corrcoef(x[i], y[i])[0, 1])
    return r0, np.percentile(bs, [2.5, 97.5])


log("\n  [A-2] 층별 횡단면 상관 (침체 심도=최대낙폭, 음수일수록 가설 부합)")
for tag in ("all", "hi", "lo"):
    r, ci = boot(agg[f"padi_{tag}"], agg.maxdd, groups=agg.mid)
    rs = agg[f"padi_{tag}"].corr(agg.maxdd, method="spearman")
    log(f"    padi_{tag:3s}: r={r:+.3f} 클러스터CI[{ci[0]:+.3f},{ci[1]:+.3f}] ρ={rs:+.3f}")

log("\n" + "=" * 66)
log("[B] 경쟁압력 축 — 융합데이터 5개국 자산")
log("  [B-1] 경쟁압력 자체의 횡단면 설명력")
for v, lab in [("cn_share", "중국 출원비중"), ("cn_growth", "중국 출원증가율"), ("kr_rel", "한국 상대지위")]:
    r, ci = boot(agg[v], agg.maxdd, groups=agg.mid)
    log(f"    corr({lab:12s}, 최대낙폭) = {r:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}]")

log("\n  [B-2] PADI와 경쟁압력은 서로 다른 것을 재는가")
r, ci = boot(agg.padi_all, agg.cn_share, groups=agg.mid)
log(f"    corr(PADI, 중국비중) = {r:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}]  (낮으면 독립 축)")

log("\n  [B-3] 다중회귀: 최대낙폭 ~ PADI + 경쟁압력")
reg = agg.dropna(subset=["maxdd", "padi_all", "cn_share", "cn_growth"])
f = smf.ols("maxdd ~ padi_all + cn_share + cn_growth", data=reg).fit()
log("    " + f.summary2().tables[1].round(3).to_string().replace("\n", "\n    "))
log(f"    R² = {f.rsquared:.3f}, n = {int(f.nobs)}")
f0 = smf.ols("maxdd ~ padi_all", data=reg).fit()
log(f"    PADI 단독 R² = {f0.rsquared:.3f} → 경쟁압력 추가 시 ΔR² = {f.rsquared - f0.rsquared:+.3f}")

log("\n  [B-4] 패널: 경쟁압력 통제 후 시계열 신호 (교란 제거 효과)")
cp = pd.read_csv(os.path.join(FUS, "competition_panel.csv"), encoding="utf-8-sig", dtype={"ksic": str})
cp["yr"] = cp.yr.astype(int)
M["yr"] = PSTART + M.mi // 12
MM = M.merge(cp[["ksic", "yr", "cn_share", "cn_growth"]], on=["ksic", "yr"], how="left")
res = []
for ind in ("padi_all", "padi_hi", "padi_lo"):
    for h in (3, 6, 12):
        d = MM.copy()
        d["y"] = d.groupby("ksic")["yoy"].shift(-h)
        d = d.dropna(subset=["y", ind, "cn_share"])
        if len(d) < 300: continue
        d["x"] = d.groupby("ksic")[ind].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
        for spec, form in (("기본", "y ~ x + C(ksic) + C(ti)"),
                           ("경쟁통제", "y ~ x + cn_share + cn_growth + C(ksic) + C(ti)")):
            try:
                fit = smf.ols(form, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.ksic})
                res.append({"ind": ind, "h": h, "spec": spec, "beta": fit.params["x"],
                            "t": fit.tvalues["x"], "p": fit.pvalues["x"]})
            except Exception:
                pass
r = pd.DataFrame(res)
log(r.round(4).to_string(index=False))
log(f"    p<0.05: {int((r.p < 0.05).sum())}/{len(r)}")

agg.to_csv(os.path.join(FUS, "fusion_agg.csv"), index=False, encoding="utf-8-sig")
r.to_csv(os.path.join(FUS, "fusion_panel.csv"), index=False, encoding="utf-8-sig")
log(f"\n저장 → {FUS}")
