# -*- coding: utf-8 -*-
"""
모형 강화 2 — 분할연도 민감도 + IPC 확산지수 + 취약성 지도
  [S] 분할연도 민감도: 2014~2019 각각으로 갈라도 예측타당도가 유지되는가
      (2017 하나만 보고하면 '분할연도를 골랐다'는 지적을 받는다)
  [D] IPC 확산지수: 산업 내 IPC4 중 포기율 상승 분야의 비중 — 수준이 아닌 폭
  [M] 2축 취약성 지도 산출물
"""
import sqlite3, datetime, os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
WORK = os.path.join(BASE, "공모전데이터", "작업데이터")
FUS = os.path.join(BASE, "분석", "fusion")
MACRO = os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표")
OUT = os.path.join(BASE, "분석", "v4")
os.makedirs(OUT, exist_ok=True)
PSTART = 2008
CUTOFF = datetime.date(2025, 6, 30)
COLLECT = datetime.date(2026, 8, 12)
BUFFER, CMIN, CMAX, MINR = 300, 2005, 2021, 30
BANDS = [(2005, 2009), (2010, 2013), (2014, 2017), (2018, 2021)]
rng = np.random.default_rng(31)
log = lambda m: print(m, flush=True)

NAMES = {"15220": "신발부품", "20202": "합성수지", "20411": "일반도료", "20423": "화장품",
         "20491": "감광재료", "21220": "한의약품", "23992": "연마재", "24191": "도금강재",
         "24219": "비철금속", "25914": "금속압형", "25922": "도금업", "25924": "절삭가공",
         "25932": "일반철물", "26112": "비메모리", "26223": "연성PCB", "26224": "실장기판",
         "26291": "전자축전기", "26322": "컴퓨터모니터", "26323": "컴퓨터프린터", "26422": "이동전화기",
         "26529": "기타음향", "27211": "레이더항행", "28122": "전기회로", "28511": "주방전기",
         "28902": "탄소절연", "29131": "액체펌프", "29132": "기체펌프", "29133": "탭밸브",
         "29150": "산업오븐", "29162": "승강기", "29210": "농업기계", "29229": "가공공작기계",
         "29241": "건설기계", "29280": "산업로봇", "30393": "자동차의자", "31202": "철도부품",
         "31312": "무인항공기", "33301": "체력단련", "33309": "기타운동"}


def to_date(s): return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
def add_years(d, n):
    try: return d.replace(year=d.year + n)
    except ValueError: return d.replace(year=d.year + n, day=28)
def band_of(y):
    for lo, hi in BANDS:
        if lo <= y <= hi: return f"{lo}-{hi}"
    return None


# ── 이벤트 → 노출 (IPC 포함) ────────────────────────────────────
snap = max([os.path.join(WORK, f) for f in os.listdir(WORK) if f.startswith("registry_snapshot")],
           key=os.path.getmtime)
con = sqlite3.connect(snap)
rows = con.execute("""SELECT rgst_no, ksic, reg_date, exp_date, last_annl FROM registry
  WHERE status='OK' AND source LIKE 'datago%' AND reg_date IS NOT NULL
    AND exp_date IS NOT NULL AND last_annl IS NOT NULL""").fetchall()
con.close()
attrs = pd.read_parquet(os.path.join(FUS, "kr_value_attrs.parquet"))
ipcm = dict(zip(attrs.rgst_no, attrs.ipc4))

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
    b, ip = band_of(reg.year), (ipcm.get(rgst) or "NA")
    for a in range(2, min(int(M), 20) + 1):
        due = add_years(reg, a - 1)
        if due > CUTOFF or due.year < PSTART or la < a - 1: continue
        e = 1 if (ev == "abandon" and age == a and not near) else 0
        er.append((ksic, typ, b, a, due.year, e, ip))
expo = pd.DataFrame(er, columns=["ksic", "type", "band", "age", "year", "event", "ipc4"])
log(f"노출 {len(expo):,} | IPC4 {expo.ipc4.nunique()}종")


def padi_of(sub, by="ksic"):
    g = sub.groupby(["type", "band", "age"]).agg(O=("event", "sum"), R=("event", "size"))
    p = sub.groupby(["type", "age"]).agg(O=("event", "sum"), R=("event", "size"))
    g, p = g[g.R >= MINR], p[p.R >= MINR]
    hm, hp = (g.O / g.R).to_dict(), (p.O / p.R).to_dict()
    s = sub.copy()
    s["h"] = [hm.get((t, bb, a), hp.get((t, a), np.nan)) for t, bb, a in zip(s.type, s.band, s.age)]
    s = s.dropna(subset=["h"])
    return s.groupby(by).apply(lambda d: d.event.sum() / d.h.sum() if d.h.sum() > 0 else np.nan,
                               include_groups=False), s


# ── 경기: 기간별 최대낙폭 ───────────────────────────────────────
k = pd.read_csv(os.path.join(MACRO, "kosis_광공업생산지수_월별.csv"), encoding="utf-8-sig", dtype=str)
k["val"] = pd.to_numeric(k.값, errors="coerce"); k = k.dropna(subset=["val"])
k["yr"] = k.연월.str[:4].astype(int)
k["mi"] = (k.yr - PSTART) * 12 + k.연월.str[4:6].astype(int) - 1
k["k3"] = k.산업코드.str[1:]
k = k[(k.k3.str.len() == 3) & (k.항목명.str.contains("생산"))]
kq = k.groupby(["k3", "mi", "yr"]).val.mean().reset_index().rename(columns={"val": "prod"}).sort_values(["k3", "mi"])


def dd_from(yr_min=None, yr_max=None):
    s = kq
    if yr_min is not None: s = s[s.yr >= yr_min]
    if yr_max is not None: s = s[s.yr <= yr_max]
    return s.groupby("k3").apply(
        lambda g: float((g["prod"].values / np.maximum.accumulate(g["prod"].values) - 1).min() * 100),
        include_groups=False)


comp = pd.read_csv(os.path.join(FUS, "competition_panel.csv"), encoding="utf-8-sig", dtype={"ksic": str})

log("\n" + "=" * 66)
log("[S] 분할연도 민감도 — 전반기 지표 → 후반기 낙폭")
res = []
for SY in (2014, 2015, 2016, 2017, 2018, 2019):
    pe, _ = padi_of(expo[expo.year < SY])
    ce = comp[(comp.yr >= 2005) & (comp.yr < SY)].groupby("ksic").cn_growth.mean()
    A = pd.DataFrame({"padi_e": pe}).reset_index()
    A["k3"] = A.ksic.str[:3]
    A = A.merge(dd_from(SY).rename("dd_late").reset_index(), on="k3")
    A = A.merge(ce.rename("cn_e").reset_index(), on="ksic", how="left")
    A["mid"] = A.ksic.str[:2]
    A = A.dropna(subset=["padi_e", "dd_late"])
    r = A.padi_e.corr(A.dd_late)
    rs = A.padi_e.corr(A.dd_late, method="spearman")
    d = A.dropna(subset=["cn_e"])
    f = smf.ols("dd_late ~ padi_e + cn_e", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.mid})
    res.append({"분할연도": SY, "n": len(A), "r": r, "ρ": rs,
                "β_PADI": f.params["padi_e"], "z": f.tvalues["padi_e"],
                "p": f.pvalues["padi_e"], "R²": f.rsquared})
S = pd.DataFrame(res)
log(S.round(3).to_string(index=False))
log(f"  >>> r 범위 [{S.r.min():+.3f}, {S.r.max():+.3f}], 전 분할에서 음수 유지: {bool((S.r < 0).all())}")
log(f"  >>> p<0.05 분할 수: {int((S.p < 0.05).sum())}/{len(S)}")
S.to_csv(os.path.join(OUT, "split_sensitivity.csv"), index=False, encoding="utf-8-sig")

log("\n" + "=" * 66)
log("[D] IPC 확산지수 — 산업 내 기술분야별 포기 확산 폭")
SY = 2017
early = expo[expo.year < SY]
# 산업×IPC4 단위 PADI (전반기)
sub = early.groupby(["ksic", "ipc4"]).filter(lambda g: len(g) >= 200)
pi, _ = padi_of(sub, by=["ksic", "ipc4"])
pi = pi.rename("padi_ipc").reset_index()
log(f"  산업×IPC4 셀 {len(pi):,}개 (셀당 노출 200+ 필터)")
diff = pi.groupby("ksic").agg(
    n_ipc=("padi_ipc", "size"),
    breadth=("padi_ipc", lambda s: (s > 1.0).mean()),      # 기대 초과 분야 비중
    disp=("padi_ipc", "std")).reset_index()
diff = diff[diff.n_ipc >= 3]
log(f"  IPC 3개 이상 산업 {len(diff)}개")
pe, _ = padi_of(early)
A = pd.DataFrame({"padi_e": pe}).reset_index().merge(diff, on="ksic")
A["k3"] = A.ksic.str[:3]; A["mid"] = A.ksic.str[:2]
A = A.merge(dd_from(SY).rename("dd_late").reset_index(), on="k3")
ce = comp[(comp.yr >= 2005) & (comp.yr < SY)].groupby("ksic").cn_growth.mean().rename("cn_e")
A = A.merge(ce.reset_index(), on="ksic", how="left").dropna(subset=["dd_late", "padi_e", "cn_e"])
log(f"  분석 표본 {len(A)}산업")
for v, lab in [("breadth", "확산 폭(기대초과 IPC 비중)"), ("disp", "분야간 편차")]:
    r = A[v].corr(A.dd_late)
    b = smf.ols("dd_late ~ padi_e + cn_e", data=A).fit()
    f = smf.ols(f"dd_late ~ padi_e + cn_e + {v}", data=A).fit(
        cov_type="cluster", cov_kwds={"groups": A.mid})
    log(f"  {lab:24s} 단순r={r:+.3f} | ΔR²={f.rsquared-b.rsquared:+.3f} "
        f"β={f.params[v]:+7.2f} t={f.tvalues[v]:+5.2f} p={f.pvalues[v]:.3f}")
A.to_csv(os.path.join(OUT, "diffusion.csv"), index=False, encoding="utf-8-sig")

log("\n" + "=" * 66)
log("[M] 취약성 지도 생성")
P = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "mid": str})
P = P.dropna(subset=["padi_e", "cn_growth_e", "dd_late"])
P["name"] = P.ksic.map(NAMES).fillna(P.ksic)
fig, ax = plt.subplots(figsize=(11, 8))
sc = ax.scatter(P.padi_e, P.cn_growth_e, s=np.abs(P.dd_late) * 6,
                c=P.dd_late, cmap="RdYlGn", edgecolors="k", linewidths=.6, alpha=.85)
ax.axvline(P.padi_e.median(), color="gray", ls="--", lw=.8)
ax.axhline(P.cn_growth_e.median(), color="gray", ls="--", lw=.8)
for _, r in P.iterrows():
    ax.annotate(r["name"], (r.padi_e, r.cn_growth_e), fontsize=7,
                xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("전반기 특허 조기포기율 (PADI) →  내부 기술자산 이탈")
ax.set_ylabel("전반기 중국 출원 증가율 (%) →  외부 경쟁 압력")
ax.set_title("산업 취약성 지도 — 색·크기 = 후반기(2017~) 최대낙폭", fontsize=13)
plt.colorbar(sc, ax=ax, label="최대낙폭 (%)")
ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "vulnerability_map.png"), dpi=140); plt.close(fig)
log(f"  저장: vulnerability_map.png")

P["risk_score"] = (P.padi_e.rank(pct=True) + P.cn_growth_e.rank(pct=True)) / 2
rank = P.sort_values("risk_score", ascending=False)[["ksic", "name", "padi_e", "cn_growth_e", "risk_score", "dd_late"]]
log("\n  [취약성 랭킹 상위 8]")
log(rank.head(8).round(3).to_string(index=False))
log("  [하위 5]")
log(rank.tail(5).round(3).to_string(index=False))
rank.to_csv(os.path.join(OUT, "vulnerability_rank.csv"), index=False, encoding="utf-8-sig")
log(f"\n저장 → {OUT}")
