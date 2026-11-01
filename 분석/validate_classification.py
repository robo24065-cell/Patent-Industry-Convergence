# -*- coding: utf-8 -*-
"""
융합데이터 KSIC 라벨의 구성타당도 검증 — 라벨 대 라벨 직접 대조
────────────────────────────────────────────────────────────────
설계:
  정답지  = 기업의 사업자등록 업종 (DART 기업개황 induty_code)
  피검증  = 융합데이터가 그 기업의 특허에 부여한 세세분류코드
  연결고리 = 법인등록번호 13자리 완전일치 (이름 유사도 아님 → 위양성 0)

왜 상관이 아니라 대조인가:
  "특허지표 ↔ 생산지표 상관"은 분류 정확도를 측정하지 못한다. 완벽한 분류기라도
  특허성향·시차·산업규모 때문에 상관이 낮게 나오기 때문이다(구성타당도 실패).
  라벨 대 라벨 비교는 그 문제가 없다.

정답지의 한계(반드시 병기):
  기업 주업종 ≠ 개별 특허의 기술귀속 산업. 다각화 기업일수록 불일치가 커지는 것은
  데이터 오류가 아니라 정상이다. 따라서 '전체 일치율'이 아니라
  **다각화 수준별 일치율 곡선**과 **단일업종 중소기업 부분표본**을 주 결과로 읽는다.
"""
import os, re, sqlite3, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
DART = os.path.join(BASE, "공모전데이터", "외부데이터", "DART")
OUT  = os.path.join(BASE, "분석", "fusion")
os.makedirs(OUT, exist_ok=True)
log = lambda m: print(m, flush=True)


def norm(s):
    s = str(s).strip()
    s = re.sub(r"\(주\)|주식회사|\(유\)|유한회사|\(재\)|재단법인|\(사\)|사단법인|\(학\)|학교법인|㈜", "", s)
    return re.sub(r"[\s\.\,\-\(\)]+", "", s).upper()


# ══════════════════════════════════════════════════════════════
log("[1] 데이터 적재")
con = sqlite3.connect(os.path.join(DART, "dart_company.sqlite"))
D = pd.read_sql("SELECT * FROM company WHERE induty_code<>''", con)
con.close()
D["jurir"] = D.jurir_no.str.replace("-", "", regex=False).str.strip()
D["nm"] = D.corp_name.map(norm)
log(f"  DART 업종확보 기업 {len(D):,}")

C = pd.read_parquet(os.path.join(DART, "corp_applicant.parquet"))
log(f"  특허 법인명부 {len(C):,}")

kr = pd.read_parquet(os.path.join(BASE, "공모전데이터", "작업데이터", "kr_registered.parquet"),
                     columns=["출원번호", "출원연도", "출원인명", "세세분류코드"])
kr["nm"] = kr["출원인명"].map(norm)
kr["ksic"] = kr["세세분류코드"].astype(str).str.strip()
kr = kr[kr.ksic.str.len() == 5]
log(f"  등록특허(세세분류 유효) {len(kr):,}")

# ══════════════════════════════════════════════════════════════
log("\n[2] 연결 — 법인등록번호 완전일치로 이름매칭 검증")
link = C[["nm", "jurir", "출원인명"]].merge(
    D[["nm", "jurir", "corp_name", "induty_code", "corp_cls", "est_dt"]],
    on="nm", how="inner", suffixes=("_pat", "_dart"))
link["id_ok"] = (link.jurir_pat == link.jurir_dart) & (link.jurir_pat.str.len() == 13)
n_all, n_ok = len(link), int(link.id_ok.sum())
log(f"  이름일치 쌍 {n_all:,} → 법인번호까지 일치 {n_ok:,} ({n_ok/n_all*100:.1f}%)")
log(f"  ※ 나머지 {n_all-n_ok:,}쌍은 동명이인(상호 중복) → 폐기. 위양성 제거.")
L = link[link.id_ok].drop_duplicates("nm")[["nm", "corp_name", "induty_code", "corp_cls", "est_dt"]]
log(f"  최종 검증대상 기업 {len(L):,}개")

# ══════════════════════════════════════════════════════════════
log("\n[3] 특허 단위 대조표 구성")
M = kr.merge(L, on="nm", how="inner")
log(f"  대조 가능 특허 {len(M):,}건 (전체 등록특허의 {len(M)/len(kr)*100:.1f}%)")

# 기업 특성
prof = M.groupby("nm").agg(n_pat=("출원번호", "size"),
                           n_ksic=("ksic", "nunique")).reset_index()
prof["diversification"] = prof.n_ksic / prof.n_pat.clip(upper=1e9)
M = M.merge(prof, on="nm", how="left")
log(f"  기업당 특허 중앙값 {prof.n_pat.median():.0f}, 보유 세세분류 중앙값 {prof.n_ksic.median():.0f}")

# ══════════════════════════════════════════════════════════════
log("\n[4] 일치율 — 공통 해상도까지 잘라서 비교")


def agree(a, b, k):
    """앞 k자리 일치 여부. 정답지가 k자리보다 짧으면 비교 불가(NaN)."""
    aa, bb = a.str[:k], b.str[:k]
    valid = (a.str.len() >= k) & (b.str.len() >= k)
    return np.where(valid, (aa == bb), np.nan)


for k, lab in ((2, "대분류내 중분류 2자리"), (3, "소분류 3자리"), (5, "세세분류 5자리")):
    M[f"ag{k}"] = agree(M.ksic, M.induty_code, k)

res = {}
log("  [특허 가중]")
for k in (2, 3, 5):
    v = pd.Series(M[f"ag{k}"]).dropna()
    res[f"patent_w_{k}"] = (float(v.mean()), int(len(v)))
    log(f"    {k}자리 일치율 {v.mean()*100:5.1f}%  (비교가능 {len(v):,}건)")

log("  [기업 가중 — 대기업 편중 제거]")
CW = M.groupby("nm")[["ag2", "ag3", "ag5"]].mean().reset_index()
for k in (2, 3, 5):
    v = CW[f"ag{k}"].dropna()
    res[f"corp_w_{k}"] = (float(v.mean()), int(len(v)))
    log(f"    {k}자리 기업평균 일치율 {v.mean()*100:5.1f}%  (기업 {len(v):,}개)")

# ══════════════════════════════════════════════════════════════
log("\n[5] Cohen's kappa — 우연일치 보정")


def kappa(a, b):
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    po = (a == b).mean()
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    pe = sum(pa.get(c, 0) * pb.get(c, 0) for c in set(pa.index) | set(pb.index))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan, po, pe


for k in (2, 3, 5):
    sub = M[(M.ksic.str.len() >= k) & (M.induty_code.str.len() >= k)]
    kp, po, pe = kappa(sub.ksic.str[:k], sub.induty_code.str[:k])
    res[f"kappa_{k}"] = float(kp)
    interp = ("거의완전" if kp > .8 else "상당" if kp > .6 else "보통" if kp > .4
              else "약함" if kp > .2 else "미미")
    log(f"  {k}자리 κ={kp:.3f} ({interp})  관측일치 {po*100:.1f}% 우연기대 {pe*100:.1f}%")

# ══════════════════════════════════════════════════════════════
log("\n[6] ★ 다각화 수준별 일치율 — 정답지 한계를 통제한 핵심 결과")
prof2 = M.groupby("nm").agg(n_pat=("출원번호", "size"), n_ksic=("ksic", "nunique"),
                            ag3=("ag3", "mean"), ag5=("ag5", "mean")).reset_index()
bins = [(1, 1, "단일업종"), (2, 3, "2~3업종"), (4, 10, "4~10업종"), (11, 10**9, "11업종+")]
rows = []
for lo, hi, lab in bins:
    s = prof2[(prof2.n_ksic >= lo) & (prof2.n_ksic <= hi)]
    if len(s):
        rows.append({"층": lab, "기업수": len(s), "특허수": int(s.n_pat.sum()),
                     "3자리일치": s.ag3.mean(), "5자리일치": s.ag5.mean()})
        log(f"  {lab:8s} 기업 {len(s):5,}개 | 3자리 {s.ag3.mean()*100:5.1f}% | 5자리 {s.ag5.mean()*100:5.1f}%")
DIV = pd.DataFrame(rows)

log("\n  [보유 특허 규모별]")
prof2["sz"] = pd.cut(prof2.n_pat, [0, 1, 5, 20, 100, 10**9],
                     labels=["1건", "2~5건", "6~20건", "21~100건", "100건+"])
for lab, s in prof2.groupby("sz", observed=True):
    log(f"  {str(lab):8s} 기업 {len(s):5,}개 | 3자리 {s.ag3.mean()*100:5.1f}% | 5자리 {s.ag5.mean()*100:5.1f}%")

log("\n  [★ 정제표본 = 단일업종 & 특허 20건 이하 중소기업]")
CLEAN = prof2[(prof2.n_ksic == 1) & (prof2.n_pat <= 20)]
log(f"  기업 {len(CLEAN):,}개 | 3자리 일치율 {CLEAN.ag3.mean()*100:.1f}% | 5자리 {CLEAN.ag5.mean()*100:.1f}%")
res["clean_3"] = float(CLEAN.ag3.mean()); res["clean_5"] = float(CLEAN.ag5.mean())
res["clean_n"] = int(len(CLEAN))

# ══════════════════════════════════════════════════════════════
log("\n[7] 불일치 구조 — 어느 산업이 체계적으로 어긋나는가")
sub = M[(M.ksic.str.len() >= 3) & (M.induty_code.str.len() >= 3)].copy()
sub["k3f"] = sub.ksic.str[:3]; sub["k3d"] = sub.induty_code.str[:3]
by = sub.groupby("k3f").agg(n=("출원번호", "size"), ag=("ag3", "mean")).reset_index()
by = by[by.n >= 200].sort_values("ag")
log("  [일치율 낮은 산업 8]")
for r in by.head(8).itertuples():
    top = sub[(sub.k3f == r.k3f) & (sub.ag3 == 0)].k3d.value_counts().head(2)
    log(f"    {r.k3f} n={r.n:6,} 일치 {r.ag*100:4.1f}% → 주로 {dict(top)} 로 어긋남")
log("  [일치율 높은 산업 8]")
for r in by.tail(8).itertuples():
    log(f"    {r.k3f} n={r.n:6,} 일치 {r.ag*100:4.1f}%")
by.to_csv(os.path.join(OUT, "cls_agree_by_industry.csv"), index=False, encoding="utf-8-sig")

# ══════════════════════════════════════════════════════════════
log("\n[8] 우리 분석대상 39개 산업만")
try:
    P = pd.read_csv(os.path.join(OUT, "predictive_agg.csv"), encoding="utf-8-sig", dtype={"ksic": str})
    tgt = set(P.ksic)
    T = sub[sub.ksic.isin(tgt)]
    log(f"  대조 특허 {len(T):,}건 | 3자리 일치율 {T.ag3.mean()*100:.1f}%")
    g = T.groupby("ksic").agg(n=("출원번호", "size"), ag3=("ag3", "mean")).reset_index()
    g.to_csv(os.path.join(OUT, "cls_agree_target39.csv"), index=False, encoding="utf-8-sig")
    log(f"  산업별 일치율 범위 {g.ag3.min()*100:.1f}% ~ {g.ag3.max()*100:.1f}% (중앙 {g.ag3.median()*100:.1f}%)")
    res["target39_ag3"] = float(T.ag3.mean())
except Exception as e:
    log(f"  건너뜀: {e}")

# ══════════════════════════════════════════════════════════════
log("\n[9] 감쇠편의 하한 — 라벨 잡음이 β를 얼마나 축소시켰는가")
p = res.get("clean_3", np.nan)
log(f"  정제표본 신뢰도 p = {p:.3f}")
log("  고전적 측정오차 하에서 신뢰도비율 λ ≈ p 이면 β_true = β_obs / λ")
log(f"    관측 β_PADI = -38.70  →  감쇠보정 하한 |β| ≥ {38.70/max(p,1e-9):.1f}")
log("  ※ 이는 라벨 잡음이 우리 계수를 **0쪽으로** 밀었다는 뜻 — 보고값은 보수적 추정이다.")

M.to_parquet(os.path.join(OUT, "cls_validation_patents.parquet"), index=False)
prof2.to_csv(os.path.join(OUT, "cls_validation_corps.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "cls_validation_summary.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
log(f"\n저장 → {OUT}")
