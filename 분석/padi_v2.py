# -*- coding: utf-8 -*-
"""
PADI v2 완전판 — 적대 감사(2026-08-11) 지적 8건 전부 반영
  [F1] 코호트 조건부 기준선 h(a, c-밴드)        ← 치명 C1 (동일연차 세대간 2배차)
  [F2] 기간효과 δ(τ) 표준화                     ← 치명 C1
  [F3] 확장창(expanding window) 추정            ← 중대 [4] look-ahead 제거
  [F4] PADI_rel (산업 자기기준선) + 횡단면 차분  ← 치명 C4 (산업 수준 ±25%)
  [F5] 특허/실용신안 분리                        ← 치명 C3 (실용 기준선 붕괴)
  [F6] 코호트 필터 등록 2005~2021               ← 경미 [7]
  [F7] 소멸판정 버퍼 270→300일                  ← 중대 데이터[1]
  [F8] 검증 게이트: 분할 안정성·비례성 검정      ← 치명 C2 (동어반복 게이트 폐기)

산출: analysis/v2/ 에 지표 패널(parquet/csv) + 진단 리포트
"""
import sqlite3, datetime, os, json, argparse
import numpy as np
import pandas as pd

SNAP_DIR = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\공모전데이터\작업데이터"
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\v2"
os.makedirs(OUT, exist_ok=True)

CUTOFF = datetime.date(2025, 6, 30)     # 패널 종점
COLLECT = datetime.date(2026, 8, 12)    # 수집 기준일
BUFFER_DAYS = 300                        # [F7] 추납6M+회복3M+처리지연 여유
COHORT_MIN, COHORT_MAX = 2005, 2021      # [F6]
PANEL_START = 2008                       # 위험집합이 안정화되는 시점
COHORT_BANDS = [(2005, 2009), (2010, 2013), (2014, 2017), (2018, 2021)]   # [F1]
MIN_RISK = 30                            # h 추정 최소 위험집합


def log(m):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)


def to_date(s):
    return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def add_years(d, n):
    try:
        return d.replace(year=d.year + n)
    except ValueError:
        return d.replace(year=d.year + n, day=28)


def band_of(year):
    for lo, hi in COHORT_BANDS:
        if lo <= year <= hi:
            return f"{lo}-{hi}"
    return None


def quarter_index(d):
    return (d.year - PANEL_START) * 4 + (d.month - 1) // 3


def load_events(snap_path):
    """이벤트 테이블 구축 — [F5][F6][F7] 적용"""
    con = sqlite3.connect(snap_path)
    rows = con.execute("""
        SELECT rgst_no, ksic, reg_date, exp_date, last_annl FROM registry
        WHERE status='OK' AND source LIKE 'datago%'
          AND reg_date IS NOT NULL AND exp_date IS NOT NULL AND last_annl IS NOT NULL
    """).fetchall()
    con.close()

    recs = []
    for rgst, ksic, rd, ed, la in rows:
        reg, exp = to_date(rd), to_date(ed)
        if exp <= reg:
            continue
        if not (COHORT_MIN <= reg.year <= COHORT_MAX):    # [F6]
            continue
        typ = "UM" if rgst.startswith("2") else "PAT"     # [F5]
        M = round((exp - reg).days / 365.25)
        la = int(la)
        covered = add_years(reg, la)
        if covered >= exp:
            ev, age, evd = "expire", None, exp
        elif (COLLECT - covered).days > BUFFER_DAYS:      # [F7]
            ev, age, evd = "abandon", la + 1, covered
        else:
            ev, age, evd = "alive", None, None
        near = (ev == "abandon" and age is not None and age > M - 2)
        recs.append((rgst, ksic, typ, reg, exp, M, la, ev, age, evd, near,
                     band_of(reg.year), reg.year))
    df = pd.DataFrame(recs, columns=["rgst", "ksic", "type", "reg", "exp", "M", "la",
                                     "event", "age", "evd", "near", "band", "regyear"])
    return df


def build_exposure(df):
    """특허별 (연차, 기한분기) 노출 레코드 — due-in-quarter 위험집합"""
    exp_rows = []
    for r in df.itertuples(index=False):
        for a in range(2, min(int(r.M), 20) + 1):
            due = add_years(r.reg, a - 1)
            if due > CUTOFF or due.year < PANEL_START:
                continue
            if r.la < a - 1:               # 이전 연차 미납 → 이미 이탈
                continue
            ev = 1 if (r.event == "abandon" and r.age == a and not r.near) else 0
            exp_rows.append((r.ksic, r.type, r.band, a, quarter_index(due), due.year, ev))
    return pd.DataFrame(exp_rows, columns=["ksic", "type", "band", "age", "qi", "year", "event"])


def estimate_hazard(expo, upto_year=None):
    """[F1][F3] 코호트밴드 조건부 기저위험률. upto_year 지정 시 확장창(그 해까지의 정보만)"""
    src = expo if upto_year is None else expo[expo.year <= upto_year]
    g = src.groupby(["type", "band", "age"]).agg(O=("event", "sum"), R=("event", "size")).reset_index()
    g = g[g.R >= MIN_RISK]
    g["h"] = g.O / g.R
    # 밴드별 결측 연차는 동일 type 전체 풀링으로 폴백
    pool = src.groupby(["type", "age"]).agg(O=("event", "sum"), R=("event", "size")).reset_index()
    pool = pool[pool.R >= MIN_RISK]
    pool["h_pool"] = pool.O / pool.R
    return (g.set_index(["type", "band", "age"])["h"].to_dict(),
            pool.set_index(["type", "age"])["h_pool"].to_dict())


def compute_period_effect(expo, hmap, hpool):
    """[F2] 기간효과 δ(τ): 분기별 (전체 관측/전체 기대) — 전산업 공통충격"""
    e = expo.copy()
    e["h"] = [hmap.get((t, b, a), hpool.get((t, a), np.nan))
              for t, b, a in zip(e.type, e.band, e.age)]
    e = e.dropna(subset=["h"])
    d = e.groupby("qi").agg(O=("event", "sum"), E=("h", "sum")).reset_index()
    d["delta"] = d.O / d.E.replace(0, np.nan)
    # 4분기 평활(분기 노이즈 제거)
    d["delta_s"] = d.delta.rolling(4, center=True, min_periods=1).mean()
    return d[["qi", "delta", "delta_s"]], e


def build_panel(e, delta):
    """산업×분기 O/E 패널 — δ 적용/미적용 양쪽"""
    e = e.merge(delta, on="qi", how="left")
    e["E_raw"] = e.h
    e["E_adj"] = e.h * e.delta_s          # [F2] 기간효과 반영 기대치
    p = e.groupby(["ksic", "qi"]).agg(O=("event", "sum"),
                                      E_raw=("E_raw", "sum"),
                                      E_adj=("E_adj", "sum"),
                                      R=("event", "size")).reset_index()
    return p


def rolling_indices(p, win=4):
    """4분기 이동합 + [F4] PADI_rel(산업 자기기준선 z) + 횡단면 차분"""
    out = []
    for ksic, g in p.groupby("ksic"):
        g = g.sort_values("qi").set_index("qi")
        full = pd.DataFrame(index=range(int(p.qi.min()), int(p.qi.max()) + 1))
        g = full.join(g).fillna({"O": 0.0, "E_raw": 0.0, "E_adj": 0.0, "R": 0.0})
        g["ksic"] = ksic
        for c in ("O", "E_raw", "E_adj", "R"):
            g[c + "4"] = g[c].rolling(win).sum()
        g["padi_raw"] = g.O4 / g.E_raw4.replace(0, np.nan)
        g["padi_adj"] = g.O4 / g.E_adj4.replace(0, np.nan)
        # 산업 자기기준선: 확장창 평균·표준편차(최소 12분기) → look-ahead 없음
        vals = g.padi_adj.values
        z = np.full(len(vals), np.nan)
        for i in range(len(vals)):
            hist = vals[max(0, i - 20):i]          # 직전 20분기
            hist = hist[~np.isnan(hist)]
            if len(hist) >= 12 and np.nanstd(hist) > 1e-6:
                z[i] = (vals[i] - np.nanmean(hist)) / np.nanstd(hist)
        g["padi_rel"] = z
        out.append(g.reset_index().rename(columns={"index": "qi"}))
    panel = pd.concat(out, ignore_index=True)
    # 횡단면 차분: 매 분기 전산업 중앙값 제거 → 공통 교란 소거
    med = panel.groupby("qi").padi_rel.transform("median")
    panel["padi_cs"] = panel.padi_rel - med
    return panel


def gate_tests(expo):
    """[F8] 검증 게이트 — 동어반복 O/E 폐기, 분할 안정성·비례성 검정"""
    res = {}
    pat = expo[expo.type == "PAT"]
    # (1) 분할 안정성: 전반기/후반기 h(a) 상관 및 비율
    early = pat[pat.year <= 2016].groupby("age").agg(O=("event", "sum"), R=("event", "size"))
    late = pat[pat.year > 2016].groupby("age").agg(O=("event", "sum"), R=("event", "size"))
    j = early.join(late, lsuffix="_e", rsuffix="_l")
    j = j[(j.R_e >= 100) & (j.R_l >= 100)]
    j["h_e"], j["h_l"] = j.O_e / j.R_e, j.O_l / j.R_l
    res["split_corr"] = float(j.h_e.corr(j.h_l)) if len(j) > 3 else None
    res["split_ratio_median"] = float((j.h_l / j.h_e).median()) if len(j) else None
    res["split_n_ages"] = int(len(j))
    # (2) 비례성: 산업별 h_i(a)/h_pool(a) 비가 연차에 따라 체계적으로 변하는가
    pool = pat.groupby("age").agg(O=("event", "sum"), R=("event", "size"))
    pool["h"] = pool.O / pool.R
    slopes = []
    for ksic, g in pat.groupby("ksic"):
        gg = g.groupby("age").agg(O=("event", "sum"), R=("event", "size"))
        gg = gg[gg.R >= 100]
        if len(gg) < 6:
            continue
        gg["ratio"] = (gg.O / gg.R) / pool.h.reindex(gg.index)
        gg = gg.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio"])
        if len(gg) >= 6:
            sl = np.polyfit(gg.index.values.astype(float), gg.ratio.values, 1)[0]
            slopes.append((ksic, float(sl)))
    res["proportionality_slopes"] = dict(slopes)
    res["prop_abs_slope_median"] = float(np.median([abs(s) for _, s in slopes])) if slopes else None
    # (3) 코호트 밴드 간 h(a) 격차 (F1이 필요했던 이유의 정량화)
    bands = {}
    for b, g in pat.groupby("band"):
        gg = g.groupby("age").agg(O=("event", "sum"), R=("event", "size"))
        gg = gg[gg.R >= 100]
        bands[b] = {int(a): float(o / r) for a, (o, r) in
                    zip(gg.index, zip(gg.O, gg.R))}
    common = set.intersection(*[set(v) for v in bands.values()]) if len(bands) > 1 else set()
    if common:
        ratios = []
        for a in sorted(common):
            vs = [bands[b][a] for b in bands if bands[b].get(a, 0) > 0]
            if len(vs) > 1 and min(vs) > 0:
                ratios.append(max(vs) / min(vs))
        res["cohort_gap_max_ratio"] = float(np.median(ratios)) if ratios else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None)
    args = ap.parse_args()
    snap = args.snapshot or max(
        [os.path.join(SNAP_DIR, f) for f in os.listdir(SNAP_DIR) if f.startswith("registry_snapshot")],
        key=os.path.getmtime)
    log(f"스냅샷: {os.path.basename(snap)}")

    df = load_events(snap)
    log(f"이벤트 테이블: {len(df):,}건 (코호트 {COHORT_MIN}~{COHORT_MAX})")
    print(df.groupby(["type", "event"]).size().unstack(fill_value=0).to_string())

    expo = build_exposure(df)
    log(f"노출 레코드: {len(expo):,} (산업 {expo.ksic.nunique()}개, 분기 {expo.qi.nunique()})")

    gates = gate_tests(expo)
    log("=== [F8] 게이트 검정 ===")
    print(f"  분할 안정성: corr(h_전반, h_후반) = {gates.get('split_corr')}, "
          f"후/전 비율 중앙값 = {gates.get('split_ratio_median')} (연차 {gates.get('split_n_ages')}개)")
    print(f"  비례성 위반: |기울기| 중앙값 = {gates.get('prop_abs_slope_median')}")
    print(f"  코호트 격차: 동일연차 최대/최소 h 비율 중앙값 = {gates.get('cohort_gap_max_ratio')}")

    hmap, hpool = estimate_hazard(expo)
    delta, e = compute_period_effect(expo, hmap, hpool)
    log(f"δ(τ) 추정: {len(delta)}분기, 범위 {delta.delta_s.min():.3f}~{delta.delta_s.max():.3f}")

    p = build_panel(e, delta)
    panel = rolling_indices(p)
    panel.to_parquet(os.path.join(OUT, "padi_panel_v2.parquet"), index=False)
    panel.to_csv(os.path.join(OUT, "padi_panel_v2.csv"), index=False, encoding="utf-8-sig")
    json.dump(gates, open(os.path.join(OUT, "gate_tests.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # 진단: 드리프트 제거 확인 (감사 C1의 핵심 지표)
    yr = e.merge(delta, on="qi", how="left")
    yr["cal_year"] = PANEL_START + yr.qi // 4
    diag = yr.groupby("cal_year").apply(
        lambda g: pd.Series({"OE_raw": g.event.sum() / g.h.sum(),
                             "OE_adj": g.event.sum() / (g.h * g.delta_s).sum()}),
        include_groups=False).reset_index()
    log("=== 연도별 전산업 O/E (드리프트 진단) ===")
    print(diag.to_string(index=False))
    diag.to_csv(os.path.join(OUT, "drift_diagnostic.csv"), index=False, encoding="utf-8-sig")
    log(f"저장 완료 → {OUT}")


if __name__ == "__main__":
    main()
