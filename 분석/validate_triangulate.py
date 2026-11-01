# -*- coding: utf-8 -*-
"""
융합데이터 KSIC 라벨 삼각검증 — 오차구조가 다른 3개 라벨의 제약비교
────────────────────────────────────────────────────────────────
  라벨 ①  융합데이터 AI 분류    : 특허 문헌 내용 기반, KSIC 10차     [피검증]
  라벨 ②  IPC-KSIC 공식 연계표  : IPC 기반, 특허청 2013(KSIC 9차)   [정답지 A · 기술축]
  라벨 ③  DART 기업 등록업종    : 사업자등록 주업종, 기업 단위        [정답지 B · 기업축]

핵심 논리:
  ②와 ③은 오차의 원천이 완전히 다르다(기술축 vs 기업축).
  ①이 ②와 잘 맞고 ③과 어긋나되 그 어긋남이 '다각화 기업'에 집중된다면,
  ①의 낮은 ③-일치율은 분류 실패가 아니라 **기업 업종 라벨의 한계**를 뜻한다.

★ 공정 비교를 위한 두 가지 필수 보정 (미보정 시 일치율이 인위적으로 붕괴한다)
  [보정1] 라벨공간 제약
      융합데이터는 50개 세세분류(=25개 소분류, 13개 중분류)만 사용한다.
      반면 연계표는 61개 전 산업에 예측한다. 연계표가 라벨공간 밖(농림어업·식료품 등)을
      지목하면 자동 불일치가 되어 불공정하다.
      → 연계표 예측을 융합데이터 라벨공간 안에서 다시 argmax 하여 '같은 선택지'로 맞춘다.
  [보정2] KSIC 개정 (9차 ↔ 10차)
      연계표의 4자리 코드(C2918/C2919 등)는 9차 체계라 10차의 29131/29132 와 절대 안 맞는다.
      → 비교 해상도를 소분류 3자리로 상한한다. 3자리는 두 개정판에서 안정적이다.
"""
import os, re, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
EXT  = os.path.join(BASE, "공모전데이터", "외부데이터")
OUT  = os.path.join(BASE, "분석", "fusion")
log  = lambda m: print(m, flush=True)
res  = {}


def ipc_norm(s):
    """'H01L 21/00' → ('H01L21/00', 'H01L', 'H01')"""
    s = re.sub(r"\s+", "", str(s)).upper()
    m = re.match(r"^([A-H]\d{2}[A-Z])(\d+/\d+)", s)
    if m:
        return f"{m.group(1)}{m.group(2)}", m.group(1), m.group(1)[:3]
    m = re.match(r"^([A-H]\d{2}[A-Z])", s)
    if m:
        return "", m.group(1), m.group(1)[:3]
    m = re.match(r"^([A-H]\d{2})", s)
    if m:
        return "", "", m.group(1)
    return "", "", ""


def ind_key(code):
    """연계표 산업코드는 해상도가 섞여 있다 → (키, 자릿수).
       'C1500'→('15',2) | 'C2610'→('261',3) | 'C2918'→('2918',4) | 'A0000'→('A',0)"""
    d = code[1:5]
    if d == "0000":
        return code[0], 0
    if d[2:4] == "00":
        return d[:2], 2
    if d[3] == "0":
        return d[:3], 3
    return d[:4], 4


# ══════════════════════════════════════════════════════════════
log("[1] 데이터 적재")
CC = pd.read_excel(os.path.join(EXT, "20131002_IPC-KSIC", "20131002_IPC-KSIC.xlsx"),
                   sheet_name="전체연계표", dtype=str)
CC.columns = ["ind", "ipc", "cnt"]
CC["cnt"] = pd.to_numeric(CC.cnt, errors="coerce").fillna(0)
CC = CC[CC.ind.str.match(r"^[A-Z]\d{4}$", na=False)].copy()
CC[["ikey", "ires"]] = CC.ind.apply(lambda c: pd.Series(ind_key(c)))
CC[["sg", "mg", "cl"]] = CC.ipc.apply(lambda s: pd.Series(ipc_norm(s)))
log(f"  연계표 IPC-산업 쌍 {len(CC):,} | 산업코드 {CC.ind.nunique()}종 "
    f"| 해상도 {CC.groupby('ires').ind.nunique().to_dict()}")

kr = pd.read_parquet(os.path.join(BASE, "공모전데이터", "작업데이터", "kr_registered.parquet"),
                     columns=["출원번호", "출원연도", "출원인명", "메인IPC", "세세분류코드"])
kr["ksic"] = kr["세세분류코드"].astype(str).str.strip()
kr = kr[kr.ksic.str.len() == 5].copy()
nz = kr["메인IPC"].apply(lambda s: pd.Series(ipc_norm(s)))
kr["sg"], kr["mg"], kr["cl"] = nz[0], nz[1], nz[2]

SPACE3 = set(kr.ksic.str[:3].unique())
SPACE2 = set(kr.ksic.str[:2].unique())
log(f"  등록특허 {len(kr):,} | 융합데이터 라벨공간: 소분류 {len(SPACE3)}종, 중분류 {len(SPACE2)}종")


# ══════════════════════════════════════════════════════════════
def make_lookup(depth):
    """[보정1+2] 연계표를 융합데이터 라벨공간으로 제약하고 depth 자리에서 argmax."""
    c = CC[CC.ires >= depth].copy()
    c["key"] = c.ikey.str[:depth]
    space = SPACE3 if depth == 3 else SPACE2

    # [보정3] 해상도 승격
    #   연계표에 3자리 항목이 아예 없는 중분류가 있다(15·21·23·30·33 → C1500·C2100·…만 존재).
    #   이 상태로 3자리 비교를 하면 해당 산업은 연계표가 절대 지목할 수 없어 영구 0% 가 된다.
    #   융합데이터 라벨공간에 그 중분류의 소분류가 **유일할 때만** 안전하게 승격한다.
    if depth == 3:
        have3 = set(CC[CC.ires >= 3].ikey.str[:2])
        promo = {}
        for m in sorted(SPACE2):
            if m not in have3:
                cand = {k for k in SPACE3 if k[:2] == m}
                if len(cand) == 1:
                    promo[m] = next(iter(cand))
        if promo:
            log(f"    [해상도 승격] {promo}  (라벨공간에 후보 유일 → 모호성 없음)")
            p = CC[(CC.ires == 2) & (CC.ikey.isin(promo))].copy()
            p["key"] = p.ikey.map(promo)
            c = pd.concat([c, p], ignore_index=True)

    c = c[c.key.isin(space)]
    tbl = {}
    for lvl in ("sg", "mg", "cl"):
        g = c[c[lvl] != ""].groupby([lvl, "key"], as_index=False).cnt.sum()
        tot = g.groupby(lvl).cnt.sum()
        g = g.sort_values("cnt", ascending=False).drop_duplicates(lvl)
        g = g.merge(tot.rename("tot"), left_on=lvl, right_index=True)
        g["conf"] = g.cnt / g.tot
        tbl[lvl] = {r[0]: (r[1], r[2]) for r in zip(g[lvl], g.key, g.conf)}
    return tbl


def predict(tbl):
    out_k, out_c, out_l = [], [], []
    for sg, mg, cl in zip(kr.sg, kr.mg, kr.cl):
        hit = tbl["sg"].get(sg)
        lv = "sub"
        if hit is None:
            hit, lv = tbl["mg"].get(mg), "main"
        if hit is None:
            hit, lv = tbl["cl"].get(cl), "class"
        if hit is None:
            out_k.append(""); out_c.append(np.nan); out_l.append("none")
        else:
            out_k.append(hit[0]); out_c.append(hit[1]); out_l.append(lv)
    return out_k, out_c, out_l


log("\n[2] ★ 라벨① vs 라벨② — 동일 선택지 제약 비교")
for depth in (3, 2):
    tbl = make_lookup(depth)
    k, c, l = predict(tbl)
    kr[f"cc{depth}"], kr[f"conf{depth}"], kr[f"lvl{depth}"] = k, c, l
    V = kr[kr[f"cc{depth}"] != ""].copy()
    V["ag"] = V.ksic.str[:depth] == V[f"cc{depth}"]
    pa = V.ksic.str[:depth].value_counts(normalize=True)
    pb = V[f"cc{depth}"].value_counts(normalize=True)
    e = sum(pa.get(x, 0) * pb.get(x, 0) for x in set(pa.index) | set(pb.index))
    o = V.ag.mean(); kp = (o - e) / (1 - e)
    lab = "소분류 3자리" if depth == 3 else "중분류 2자리"
    log(f"\n  ── {lab} (선택지 {len(SPACE3) if depth==3 else len(SPACE2)}종) ──")
    log(f"    비교가능 {len(V):,}건 ({len(V)/len(kr)*100:.1f}%)")
    log(f"    관측 일치율 {o*100:5.1f}%  |  우연 기대 {e*100:4.1f}%  |  κ={kp:.3f}  "
        f"|  무작위 대비 {o/e:.1f}배")
    res[f"cc{depth}_obs"], res[f"cc{depth}_exp"], res[f"cc{depth}_kappa"] = float(o), float(e), float(kp)
    if depth == 3:
        V3 = V
        log("\n    [연계표 확신도별 — 연계표가 확신할수록 융합데이터와 일치해야 한다]")
        V["cb"] = pd.cut(V[f"conf{depth}"], [0, .25, .5, .75, .9, 1.01],
                         labels=["~25%", "25~50%", "50~75%", "75~90%", "90%+"])
        mono = V.groupby("cb", observed=True).ag.mean()
        for lb, g in V.groupby("cb", observed=True):
            log(f"      conf {str(lb):8s} n={len(g):8,}  일치율 {g.ag.mean()*100:5.1f}%")
        log(f"    → 단조증가: {'예 — 두 분류기가 같은 신호를 보고 있다는 증거' if mono.is_monotonic_increasing else '아니오'}")
        res["cc3_monotonic"] = bool(mono.is_monotonic_increasing)
        S = V[V[f"lvl{depth}"] == "sub"]
        log(f"    [IPC 서브그룹 정밀매칭만] n={len(S):,} 일치율 {S.ag.mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════
log("\n[3] ★ 3자 동시 대조 — 중분류 2자리 공통 잣대")
try:
    M = pd.read_parquet(os.path.join(OUT, "cls_validation_patents.parquet"))
    B = kr[kr.cc2 != ""][["출원번호", "ksic", "cc2"]]
    M3 = M[["출원번호", "nm", "induty_code", "n_ksic", "n_pat"]].merge(B, on="출원번호", how="inner")
    M3 = M3[M3.induty_code.str.len() >= 2].copy()
    M3["k2_dt0"] = M3.induty_code.str[:2]
    # [보정4] DART 쪽 라벨공간 제약
    #   매칭 기업의 32%는 등록업종이 융합데이터 라벨공간(13개 중분류) 밖이다
    #   (정보통신 7.9%, 전문과학 5.7%, 도소매 4.7% …). 융합데이터는 13종 안에서만 배정하므로
    #   이들은 구조적으로 일치가 불가능하다. 제외하지 않으면 일치율이 인위적으로 깎인다.
    n0 = M3.nm.nunique()
    M3 = M3[M3.k2_dt0.isin(SPACE2)].copy()
    log(f"  [DART 라벨공간 제약] 기업 {n0:,} → {M3.nm.nunique():,} "
        f"(등록업종이 13개 중분류 밖인 기업 제외)")
    M3["k2_ai"], M3["k2_cc"], M3["k2_dt"] = M3.ksic.str[:2], M3.cc2, M3.k2_dt0
    M3["ai_cc"] = M3.k2_ai == M3.k2_cc
    M3["ai_dt"] = M3.k2_ai == M3.k2_dt
    M3["cc_dt"] = M3.k2_cc == M3.k2_dt
    log(f"  3라벨 동시보유 특허 {len(M3):,}건")
    log(f"    ①AI      ↔ ②연계표(기술축)  {M3.ai_cc.mean()*100:5.1f}%")
    log(f"    ①AI      ↔ ③DART(기업축)    {M3.ai_dt.mean()*100:5.1f}%")
    log(f"    ②연계표  ↔ ③DART            {M3.cc_dt.mean()*100:5.1f}%   ← 기준선")
    log("")
    if M3.cc_dt.mean() < M3.ai_dt.mean():
        log(f"  → ★ 특허청 공식 연계표조차 기업 업종과 {M3.cc_dt.mean()*100:.1f}%밖에 안 맞는다.")
        log(f"     융합데이터는 {M3.ai_dt.mean()*100:.1f}%로 그보다 "
            f"{M3.ai_dt.mean()/M3.cc_dt.mean():.2f}배 높다.")
        log("     즉 융합데이터 AI 분류는 공식 IPC 연계표보다 기업 실체에 더 가깝다.")
    # ★ 기업 가중이 필수다. 특허 가중이면 삼성전자 한 곳(4.2만건)이 최다각화 층을
    #   혼자 지배해 층간 비교가 무의미해진다. 기업별 평균을 먼저 낸 뒤 층별로 평균한다.
    per = M3.groupby("nm").agg(n_ksic=("n_ksic", "first"), n_pat=("n_pat", "first"),
                              ai_cc=("ai_cc", "mean"), ai_dt=("ai_dt", "mean"),
                              cc_dt=("cc_dt", "mean")).reset_index()
    log(f"\n  [다각화 수준별 — 정답지③의 한계를 드러내는 핵심 표]")
    log(f"  ※ 기업 가중(기업별 평균의 평균). 특허 가중이면 초대형 출원인 1곳이 층을 지배한다.")
    log(f"    {'층':10s} {'기업':>6s} {'특허':>8s} {'①↔②':>7s} {'①↔③':>7s} {'②↔③':>7s}")
    rows = []
    for lo, hi, lab in ((1, 1, "단일업종"), (2, 3, "2~3업종"), (4, 10, "4~10업종"), (11, 10**9, "11업종+")):
        g = per[(per.n_ksic >= lo) & (per.n_ksic <= hi)]
        gp = M3[(M3.n_ksic >= lo) & (M3.n_ksic <= hi)]
        if len(g) > 5:
            log(f"    {lab:10s} {len(g):6,} {len(gp):8,} {g.ai_cc.mean()*100:6.1f}% "
                f"{g.ai_dt.mean()*100:6.1f}% {g.cc_dt.mean()*100:6.1f}%")
            rows.append({"층": lab, "기업수": len(g), "특허수": len(gp),
                         "ai_cc": g.ai_cc.mean(), "ai_dart": g.ai_dt.mean(), "cc_dart": g.cc_dt.mean()})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "tri_by_diversification.csv"), index=False, encoding="utf-8-sig")
    if len(R) >= 3:
        d_dart = R.ai_dart.is_monotonic_decreasing
        d_cc = R.ai_cc.is_monotonic_decreasing
        log(f"\n    ①↔③(기업축) 다각화에 따라 단조 하락? {'예' if d_dart else '아니오'}")
        log(f"    ①↔②(기술축) 다각화에 따라 단조 하락? {'예' if d_cc else '아니오'}")
        if d_dart and not d_cc:
            log("    → ★ 기업축 일치율만 무너지고 기술축은 유지된다.")
            log("       불일치의 원인은 융합데이터가 아니라 '기업 주업종'이라는 정답지의 한계다.")
        res["mono_dart"], res["mono_cc"] = bool(d_dart), bool(d_cc)
    res["tri_ai_cc"] = float(M3.ai_cc.mean())
    res["tri_ai_dart"] = float(M3.ai_dt.mean())
    res["tri_cc_dart"] = float(M3.cc_dt.mean())
    M3.to_parquet(os.path.join(OUT, "tri_labels.parquet"), index=False)
except FileNotFoundError:
    log("  건너뜀 — validate_classification.py 를 먼저 실행하라")

# ══════════════════════════════════════════════════════════════
log("\n[4] 우리 분석대상 39개 산업의 라벨 신뢰도 (소분류 3자리)")
try:
    P = pd.read_csv(os.path.join(OUT, "predictive_agg.csv"), encoding="utf-8-sig", dtype={"ksic": str})
    T = kr[(kr.cc3 != "") & (kr.ksic.isin(set(P.ksic)))].copy()
    T["ag"] = T.ksic.str[:3] == T.cc3
    log(f"  대조 특허 {len(T):,}건 | ①↔② 일치율 {T.ag.mean()*100:.1f}%")
    g = T.groupby("ksic").agg(n=("출원번호", "size"), cc_agree=("ag", "mean")).reset_index()
    g.to_csv(os.path.join(OUT, "cls_cc_target39.csv"), index=False, encoding="utf-8-sig")
    log(f"  산업별 범위 {g.cc_agree.min()*100:.1f}% ~ {g.cc_agree.max()*100:.1f}% "
        f"(중앙 {g.cc_agree.median()*100:.1f}%)")
    log("  [하위 5 — 라벨잡음이 큰 산업]")
    for r in g.nsmallest(5, "cc_agree").itertuples():
        log(f"    {r.ksic} n={r.n:6,} {r.cc_agree*100:5.1f}%")
    log("  [상위 5]")
    for r in g.nlargest(5, "cc_agree").itertuples():
        log(f"    {r.ksic} n={r.n:6,} {r.cc_agree*100:5.1f}%")
    res["target39_cc3"] = float(T.ag.mean())
except Exception as e:
    log(f"  건너뜀: {e}")

kr.drop(columns=["sg", "mg", "cl"]).to_parquet(os.path.join(OUT, "cc_validation.parquet"), index=False)
with open(os.path.join(OUT, "tri_validation_summary.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
log(f"\n저장 → {OUT}")
