# -*- coding: utf-8 -*-
"""N1 공용 라이브러리 (파일럿 ilib.py 복사·확장).
모든 결과·무역 로더는 연도 ≤ DEV_MAX(2021)로 로드 직후 필터한다 — 최종시험(F=2022) 결과 구간 차단.
N1 폴더(cache/, R2/)에만 쓴다. _shared·원본은 읽기 전용."""
import os, numpy as np, pandas as pd
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
B = os.path.join(BASE, "분석", "모델경쟁_20260927")
HERE = os.path.join(B, "N1_기술점유무역_산업고용조기경보"); CACHE = os.path.join(HERE, "cache"); R2 = os.path.join(HERE, "R2")
META = os.path.join(B, "_shared", "meta", "*.parquet").replace("\\", "/")
MODE = os.environ.get("N1_MODE", "dev")          # dev(기본) | future(최종시험 이후 전망 전용 — 개발 중 사용 금지)
DEV_MAX = 2024 if MODE == "future" else 2021
CCS = ["KR", "US", "JP", "CN", "EP"]

# ---------------- 특허 ----------------
def load_pat(name="pat_kcy.parquet", val="n"):
    p = pd.read_parquet(os.path.join(CACHE, name))
    return p.pivot_table(index=["ksic", "y"], columns="cc", values=val, aggfunc="sum").fillna(0)

def windows(pv, t0, lag):
    e = t0 - lag; idx = pv.index.get_level_values(1)
    w1 = pv[(idx >= e-2) & (idx <= e)].groupby(level=0).sum()
    w0 = pv[(idx >= e-5) & (idx <= e-3)].groupby(level=0).sum()
    cols = sorted(set(pv.columns))
    return w1.reindex(columns=cols, fill_value=0), w0.reindex(columns=cols, fill_value=0).reindex(w1.index, fill_value=0)

def pat_feats(pv, t0, lag=1):
    """원점 t0: 출원연도 ≤ t0-lag 만. w1=[e-2,e], w0=[e-5,e-3], e=t0-lag."""
    w1, w0 = windows(pv, t0, lag)
    w1, w0 = w1.reindex(columns=CCS, fill_value=0), w0.reindex(columns=CCS, fill_value=0)
    f = pd.DataFrame(index=w1.index)
    for c in CCS:
        f[f"mom_{c}"] = np.log((w1[c]+1)/(w0[c]+1))
        f[f"lvl_{c}"] = np.log1p(w1[c])
    tot1, tot0 = w1.sum(1), w0.sum(1)
    f["mom_ALL"] = np.log((tot1+1)/(tot0+1))
    for c in CCS:
        f[f"sh_{c}"] = w1[c]/tot1.clip(lower=1)
        f[f"dsh_{c}"] = f[f"sh_{c}"] - w0[c]/tot0.clip(lower=1)
    f["gap_CN_KR"] = f.mom_CN - f.mom_KR
    f["frgn_mom"] = np.log((w1[["US", "JP", "EP"]].sum(1)+1)/(w0[["US", "JP", "EP"]].sum(1)+1))
    # N1 신규: 경쟁국 대비 격차
    f["krcn_lr"] = np.log((w1.KR+1)/(w1.CN+1)); f["dkrcn_lr"] = f.krcn_lr - np.log((w0.KR+1)/(w0.CN+1))  # KR/CN 상대규모 이동
    f["dsh_KRmCN"] = f.dsh_KR - f.dsh_CN
    f["krjp_lr"] = np.log((w1.KR+1)/(w1.JP+1)); f["dkrjp_lr"] = f.krjp_lr - np.log((w0.KR+1)/(w0.JP+1))
    f["lead_frgn"] = f.frgn_mom - f.mom_KR  # 해외 성장 − KR 성장(국가 간 격차)
    f.index.name = "ksic"
    return f

# ---------------- KOSIS ----------------
def load_kosis50(items=("T01", "T02", "T03", "T04", "T05", "T06", "T07")):
    """50개 산업 KOSIS 광업제조업조사 5자리. 10차 ≤2020 + 11차 2021을 2020 중복 비율로 체인. 마스킹==1 → NaN. 연도 ≤ DEV_MAX."""
    d = pd.read_csv(os.path.join(BASE, "공모전데이터", "작업데이터", "kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
    d = d[d.연도 <= DEV_MAX].copy()
    d.loc[d.마스킹 == 1, "값"] = np.nan
    out = {}
    for it in items:
        x = d[d.항목코드 == it]
        a = x[x.기준 == "10차"].pivot_table(index="세세분류코드", columns="연도", values="값")
        b = x[x.기준 == "11차"].pivot_table(index="세세분류코드", columns="연도", values="값")
        a = a.where(a > 0)
        for yy in range(2021, DEV_MAX + 1):   # 11차 연도를 2020 중복연도 비율로 체인(dev 에서는 2021 만)
            if yy in b.columns and 2020 in b.columns:
                a[yy] = a[2020] * (b[yy] / b[2020].where(b[2020] > 0)).reindex(a.index)
        a.index.name = "ksic"; out[it] = a
    return out

def ksic_names():
    d = pd.read_csv(os.path.join(BASE, "공모전데이터", "작업데이터", "kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
    return d.drop_duplicates("세세분류코드").set_index("세세분류코드").산업명

def lg(M, y1, y0):
    """log(M[y1]/M[y0]); 2010(경제총조사 부재)은 2009로 대체하고 연율 보정."""
    ya, yb = (2009 if y0 == 2010 else y0), (2009 if y1 == 2010 else y1)
    if ya not in M.columns or yb not in M.columns: return pd.Series(np.nan, index=M.index)
    return np.log(M[yb] / M[ya]) * ((y1 - y0) / max(yb - ya, 1))

def lvl(M, y):
    y = 2009 if y == 2010 else y
    return np.log(M[y]) if y in M.columns else pd.Series(np.nan, index=M.index)

# ---------------- 무역 ----------------
def load_map():
    m = pd.read_csv(os.path.join(BASE, "분석", "fusion", "ksic_hs6_map.csv"), dtype=str)
    m["hs6"] = m.hs6.str.zfill(6)
    m = m[["k5", "hs6"]].drop_duplicates()
    m["w"] = 1/m.groupby("hs6").k5.transform("count")
    return m

def cn_share():
    c = pd.read_csv(os.path.join(BASE, "분석", "fusion", "cn_import_yearly_ksic5.csv"), dtype={"ksic": str})
    c = c[(c.year <= DEV_MAX) & (c.ksic_level == 5)]
    return c

def kr_export_ksic():
    x = pd.read_csv(os.path.join(B, "_shared", "ext", "R2global_kr_export_hs6.csv"), dtype={"hs6": str})
    x = x[(x.year <= DEV_MAX) & (x.flow == "X")].copy(); x["hs6"] = x.hs6.str.zfill(6)
    if MODE == "future":   # 실행 오류 수정(최종시험 이후 전용): R2global 은 ≤2021 → 2022–24 는 최종시험 때 1회 수집한 N1final 파일로 보충(2019–21 값 완전 일치 확인)
        f = os.path.join(B, "_shared", "ext", "N1final_kr_export_hs6_2019_2024.csv")
        z = pd.read_csv(f, dtype={"hs6": str}); z = z[(z.year >= 2022) & (z.year <= DEV_MAX)]; z["hs6"] = z.hs6.str.zfill(6)
        x = pd.concat([x, z[["year", "flow", "hs6", "value"]]], ignore_index=True)
    return x.merge(load_map(), on="hs6").assign(v=lambda d: d.value*d.w).pivot_table(index="k5", columns="year", values="v", aggfunc="sum")

def imports_ksic():
    """한국 HS6 수입(상대국별) → KSIC 1/m 가중 합산. 반환 {partner: DataFrame(k5 × year)}"""
    d = pd.read_csv(os.path.join(BASE, "분석", "fusion", "hs6_import_yearly.csv"), dtype={"hs6": str})
    d = d[d.year <= DEV_MAX].copy(); d["hs6"] = d.hs6.str.zfill(6)
    d = d.merge(load_map(), on="hs6").assign(v=lambda z: z.value*z.w)
    return {p: g.pivot_table(index="k5", columns="year", values="v", aggfunc="sum") for p, g in d.groupby("partner")}

# ---------------- 평가 공통 ----------------
def rank_within(D, cols, key="t0"):
    D = D.copy()
    for c in cols:
        D[c] = D.groupby(key)[c].rank(pct=True).fillna(0.5)
    return D
