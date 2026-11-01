# -*- coding: utf-8 -*-
"""공용 함수 — 모든 로더는 결과연도 ≤ DEV_MAX(2021)로 즉시 필터(최종시험 F=2022 이후 결과 미접근)."""
import os, numpy as np, pandas as pd
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
B = os.path.join(BASE, "분석", "모델경쟁_20260927")
HERE = os.path.join(B, "_ideas_R2", "industry"); CACHE = os.path.join(HERE, "cache")
DEV_MAX = 2021
CCS = ["KR", "US", "JP", "CN", "EP"]

def load_pat():
    p = pd.read_parquet(os.path.join(CACHE, "pat_kcy.parquet"))
    return p.pivot_table(index=["ksic", "y"], columns="cc", values="n", aggfunc="sum").fillna(0)

def pat_feats(pv, t0, lag=1):
    """원점 t0: 출원연도 ≤ t0-lag 만. w1=[e-2,e], w0=[e-5,e-3], e=t0-lag."""
    e = t0 - lag
    idx = pv.index.get_level_values(1)
    w1 = pv[(idx >= e-2) & (idx <= e)].groupby(level=0).sum()
    w0 = pv[(idx >= e-5) & (idx <= e-3)].groupby(level=0).sum()
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
    f["frgn_mom"] = np.log((w1[["US","JP","EP"]].sum(1)+1)/(w0[["US","JP","EP"]].sum(1)+1))
    f.index.name = "ksic"
    return f

def load_map():
    m = pd.read_csv(os.path.join(BASE, "분석", "fusion", "ksic_hs6_map.csv"), dtype=str)
    m["hs6"] = m.hs6.str.zfill(6)
    return m[["k5", "hs6"]].drop_duplicates()

def load_imports():
    d = pd.read_csv(os.path.join(BASE, "분석", "fusion", "hs6_import_yearly.csv"), dtype={"hs6": str})
    d = d[d.year <= DEV_MAX].copy(); d["hs6"] = d.hs6.str.zfill(6)
    return d.pivot_table(index=["hs6", "year"], columns="partner", values="value", aggfunc="sum").fillna(0)

def mae(y, p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))

def load_kosis50():
    """50개 산업 KOSIS 광업제조업조사 5자리. 반환: {항목: DataFrame(index=ksic, columns=year)} — 10차 ≤2020 + 11차 2021을 2020 중복으로 연결(체인)."""
    d = pd.read_csv(os.path.join(BASE, "공모전데이터", "작업데이터", "kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
    d = d[d.연도 <= DEV_MAX]
    d.loc[d.마스킹 == 1, "값"] = np.nan
    out = {}
    for it in ["T01", "T02", "T03", "T04", "T06", "T07"]:
        x = d[d.항목코드 == it]
        a = x[x.기준 == "10차"].pivot_table(index="세세분류코드", columns="연도", values="값")
        b = x[x.기준 == "11차"].pivot_table(index="세세분류코드", columns="연도", values="값")
        a = a.where(a > 0)
        if 2021 in b.columns and 2020 in b.columns:
            a[2021] = a[2020] * (b[2021] / b[2020]).reindex(a.index)
        a.index.name = "ksic"; out[it] = a
    return out

def lg(M, y1, y0):
    """log(M[y1]/M[y0]); 2010(경제총조사 부재)은 2009로 대체하고 연율 보정."""
    ya, yb = (2009 if y0 == 2010 else y0), (2009 if y1 == 2010 else y1)
    if ya not in M.columns or yb not in M.columns: return pd.Series(np.nan, index=M.index)
    return np.log(M[yb] / M[ya]) * ((y1 - y0) / max(yb - ya, 1))

def app_feats(t0, lag=1):
    a = pd.read_parquet(os.path.join(CACHE, "app_kcy.parquet"))
    e = t0 - lag; f = {}
    for cc in ["KR", "CN", "US"]:
        x = a[a.cc == cc]
        w1 = x[(x.y >= e-2) & (x.y <= e)].groupby("ksic")[["n_ap", "n_new"]].sum()
        w0 = x[(x.y >= e-5) & (x.y <= e-3)].groupby("ksic")[["n_ap", "n_new"]].sum()
        w0 = w0.reindex(w1.index).fillna(0)
        f[f"apmom_{cc}"] = np.log((w1.n_ap+1)/(w0.n_ap+1))
        f[f"newmom_{cc}"] = np.log((w1.n_new+1)/(w0.n_new+1))
        f[f"newsh_{cc}"] = w1.n_new/(w1.n_ap.clip(lower=1))
    f = pd.DataFrame(f); f.index.name = "ksic"; return f

def trade_feats(t0):
    """원점 t0까지의 무역: 한국 수출(_shared/ext R2global_kr_export_hs6, KSIC 합산) 3년 성장, 대중 수입점유율 수준·3년 변화."""
    m = load_map()
    x = pd.read_csv(os.path.join(B, "_shared", "ext", "R2global_kr_export_hs6.csv"), dtype={"hs6": str})
    x = x[(x.year <= t0) & (x.flow == "X")]; x["hs6"] = x.hs6.str.zfill(6)
    x = x.merge(m, on="hs6")  # 다중 대응 HS6는 각 KSIC에 모두 합산(연결 강도 근사)
    k = x.pivot_table(index="k5", columns="year", values="value", aggfunc="sum")
    f = pd.DataFrame(index=k.index)
    f["xg3"] = np.log(k[t0]/k[t0-3]) if (t0 in k.columns and t0-3 in k.columns) else np.nan
    f["xlv"] = np.log(k[t0]) if t0 in k.columns else np.nan
    c = pd.read_csv(os.path.join(BASE, "분석", "fusion", "cn_import_yearly_ksic5.csv"), dtype={"ksic": str})
    c = c[(c.year <= t0) & (c.ksic_level == 5)].pivot_table(index="ksic", columns="year", values="cn_share_imp")
    f = f.join(pd.DataFrame({"cnsh": c[t0], "dcnsh": c[t0]-c[t0-3]}), how="outer")
    f.index.name = "ksic"; return f

def load_kosis_all(item="T04"):
    """광업제조업조사 전체 5자리(≤2021, 10차 + 11차 2021 체인)."""
    d = pd.read_csv(os.path.join(BASE, "공모전데이터", "작업데이터", "kosis_광업제조업_2008_2024.csv"), dtype={"세세분류코드": str})
    d = d[(d.연도 <= DEV_MAX) & (d.항목코드 == item)]; d.loc[d.마스킹 == 1, "값"] = np.nan
    a = d[d.기준 == "10차"].pivot_table(index="세세분류코드", columns="연도", values="값")
    b = d[d.기준 == "11차"].pivot_table(index="세세분류코드", columns="연도", values="값")
    a = a.where(a > 0)
    if 2021 in b.columns: a[2021] = a[2020] * (b[2021]/b[2020]).reindex(a.index)
    return a

def io_feats(t0, K_all, pv):
    """전후방 연결 피처(2010 접속 IO표): 하류(판매처) 부문 과거3년 출하성장 가중평균, 상류(공급처) 동일, 하류 부문 KR 특허 모멘텀."""
    Z = pd.read_parquet(os.path.join(CACHE, "io2010_Z.parquet")); link = pd.read_csv(os.path.join(CACHE, "ksic5_io3_link.csv"), dtype=str)
    y0 = t0 - 3 if t0 - 3 != 2010 else 2009
    if y0 < 2008: y0 = 2008
    Ksec = K_all[[c for c in [y0, t0] if c in K_all.columns]].join(link.set_index("k5"), how="inner").groupby("s3").sum(min_count=1)
    g = np.log(Ksec[t0]/Ksec[y0]).replace([np.inf, -np.inf], np.nan) * (3/(t0-y0)) if y0 != t0 else pd.Series(np.nan, index=Ksec.index)
    pf = pat_feats(pv, t0)
    pm = pf[["mom_KR"]].join(link.set_index("k5"), how="inner").groupby("s3").mom_KR.mean()
    out = {}
    for k5 in pf.index:
        ss = link[link.k5 == k5].s3.tolist(); ss = [s for s in ss if s in Z.index]
        if not ss: continue
        dn = Z.loc[ss].sum(0); up = Z[ss].sum(1)
        dn = dn.drop(ss, errors="ignore"); up = up.drop(ss, errors="ignore")
        def wavg(w, x):
            x = x.reindex(w.index); m = x.notna() & (w > 0)
            return (w[m]*x[m]).sum()/w[m].sum() if w[m].sum() > 0 else np.nan
        out[k5] = {"io_dn_g3": wavg(dn, g), "io_up_g3": wavg(up, g), "io_dn_pat": wavg(dn, pm), "io_up_pat": wavg(up, pm),
                   "io_dn_cov": dn[g.reindex(dn.index).notna()].sum()/max(dn.sum(), 1)}
    f = pd.DataFrame(out).T; f.index.name = "ksic"; return f

def kab_feats(t0, lag=3):
    """한국발(우선권 KR) 해외출원(US·EP·CN·JP 합) — 출원연도 ≤ t0−lag(PCT 국내단계 30개월 지연 고려해 기본 3)."""
    a = pd.read_parquet(os.path.join(CACHE, "prio_kcy.parquet")); e = t0 - lag
    a = a[a.pc.notna() & (a.pc != a.cc)]
    w1 = a[(a.y >= e-2) & (a.y <= e)]; w0 = a[(a.y >= e-5) & (a.y <= e-3)]
    k1 = w1[w1.pc == "KR"].groupby("ksic").n.sum(); k0 = w0[w0.pc == "KR"].groupby("ksic").n.sum()
    t1 = w1.groupby("ksic").n.sum(); t0_ = w0.groupby("ksic").n.sum()
    idx = t1.index.union(t0_.index); k1, k0, t1, t0_ = [s.reindex(idx).fillna(0) for s in (k1, k0, t1, t0_)]
    f = pd.DataFrame({"kab_mom": np.log((k1+1)/(k0+1)), "kab_sh": (k1+1)/(t1+2), "kab_dsh": (k1+1)/(t1+2) - (k0+1)/(t0_+2), "kab_lvl": np.log1p(k1)})
    f.index.name = "ksic"; return f
