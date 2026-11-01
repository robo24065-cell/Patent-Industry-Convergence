# -*- coding: utf-8 -*-
"""N1 패널 구축: (KSIC 5자리 50개 × 원점 t0) 피처 + 타깃. t0 = 2005..2021(2010 제외 — 경제총조사로 광업제조업조사 없음).
타깃은 load_kosis50(≤2021) 로만 계산하므로 t0+h > 2021 인 타깃은 자동으로 NaN(최종시험 구간 미접근).
피처는 원점 t0 시점 정보만: 특허 출원연도 ≤ t0−lag, 무역·KOSIS·생산지수 ≤ t0.
출력: cache/panel_n1.parquet, ext_link_*.csv(연계표·연계율)."""
import numpy as np, pandas as pd, os
from n1lib import *
K = load_kosis50(); IDX = K["T02"].index
pv = load_pat()                                      # (ksic,y)×cc 고유 출원(파일럿 p01)
pub = pd.read_parquet(f"{CACHE}/pat_pub.parquet")
krq = pd.read_parquet(f"{CACHE}/kr_qual.parquet")
cnt = pd.read_parquet(f"{CACHE}/cn_type.parquet")
prio = pd.read_parquet(f"{CACHE}/prio_kcy.parquet")
app = pd.read_parquet(f"{CACHE}/app_kcy.parquet")
C = cn_share().pivot_table(index="ksic", columns="year", values="cn_share_imp")
X = kr_export_ksic(); IM = imports_ksic()
# 생산지수(월별, 3자리 → 없으면 2자리), 연평균
pi = pd.read_csv(os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표", "kosis_광공업생산지수_월별.csv"), dtype={"산업코드": str})
pi = pi[pi.항목코드 == "T10"].copy(); pi["값"] = pd.to_numeric(pi.값, errors="coerce"); pi["year"] = pi.연월 // 100; pi = pi[pi.year <= DEV_MAX]
PI = pi.groupby(["산업코드", "year"]).값.mean().unstack()
# 수입점유비(업종별 2자리, 2010~)
ims = pd.read_csv(os.path.join(BASE, "공모전데이터", "외부데이터", "제3축", "수입점유비_업종별.csv"))
ims["DT"] = pd.to_numeric(ims.DT, errors="coerce"); ims = ims[ims.PRD_DE <= DEV_MAX].pivot_table(index="C1", columns="PRD_DE", values="DT")
# 내수·수출 출하지수(2자리, C26만 3자리)
sh = pd.read_csv(os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표", "kosis_내수수출출하지수.csv"), dtype={"C1": str})
sh["DT"] = pd.to_numeric(sh.DT, errors="coerce"); sh["year"] = sh.PRD_DE // 100; sh = sh[sh.year <= DEV_MAX]
SHd = sh[sh.ITM_ID == "T10"].groupby(["C1", "year"]).DT.mean().unstack(); SHx = sh[sh.ITM_ID == "T11"].groupby(["C1", "year"]).DT.mean().unstack()

def pick(code_tab, k5):
    for c in ["C" + k5[:3], "C" + k5[:2]]:
        if c in code_tab.index: return c
    return None
link = pd.DataFrame({"k5": IDX})
link["pi_code"] = link.k5.map(lambda k: pick(PI, k)); link["ims_code"] = link.k5.map(lambda k: "C" + k[:2] if "C" + k[:2] in ims.index else None)
link["sh_code"] = link.k5.map(lambda k: pick(SHd, k))
link["cnsh_ok"] = link.k5.isin(C.index); link["krexp_ok"] = link.k5.isin(X.index)
if MODE == "dev": link.to_csv(f"{HERE}/ext_link_kosis_indices.csv", index=False, encoding="utf-8-sig")
print("연계율: 생산지수", link.pi_code.notna().mean(), "(3자리", link.pi_code.str.len().eq(4).mean(), ") 수입점유비", link.ims_code.notna().mean(),
      "출하지수", link.sh_code.notna().mean(), "대중수입점유", link.cnsh_ok.mean(), "한국수출", link.krexp_ok.mean())

def gser(T, codes, y1, y0):
    return pd.Series({k: (np.log(T.loc[c, y1]/T.loc[c, y0]) if (c is not None and y1 in T.columns and y0 in T.columns) else np.nan) for k, c in codes.items()})
def lser(T, codes, y):
    return pd.Series({k: (T.loc[c, y] if (c is not None and y in T.columns) else np.nan) for k, c in codes.items()})

def pubfeats(t0, lag):
    """공개연도(없으면 등록연도) ≤ t0 인 출원만 쓴 KR 모멘텀·점유변화."""
    q = pub[(pub.py <= t0)]
    p = q.pivot_table(index=["ksic", "y"], columns="cc", values="n", aggfunc="sum").fillna(0)
    f = pat_feats(p, t0, lag)[["mom_KR", "dsh_KR", "lvl_KR", "frgn_mom"]]
    return f.add_suffix(f"_P{lag}")

def krqual(t0, lag):
    e = t0 - lag
    w1 = krq[(krq.y >= e-2) & (krq.y <= e)]; w0 = krq[(krq.y >= e-5) & (krq.y <= e-3)]
    def agg(w):
        g = w.groupby("ksic")
        return pd.DataFrame({"n": g.n.sum(), "ffam": w[w.ffam == 1].groupby("ksic").n.sum(), "dom": w[w.dom == 1].groupby("ksic").n.sum(),
                             "domffam": w[(w.dom == 1) & (w.ffam == 1)].groupby("ksic").n.sum()}).fillna(0)
    a1, a0 = agg(w1), agg(w0).reindex(agg(w1).index).fillna(0)
    f = pd.DataFrame(index=a1.index)
    f["ffam_sh"] = (a1.ffam+1)/(a1.n+2); f["dffam_sh"] = f.ffam_sh - (a0.ffam+1)/(a0.n+2)
    f["dom_sh"] = (a1.dom+1)/(a1.n+2); f["ddom_sh"] = f.dom_sh - (a0.dom+1)/(a0.n+2)
    f["domffam_mom"] = np.log((a1.domffam+1)/(a0.domffam+1)); f["dom_mom"] = np.log((a1.dom+1)/(a0.dom+1))
    # 등록 여부(원점 t0 시점까지 등록된 것만) — 창 w1 은 t0 기준, w0 은 t0−3 기준으로 비교(같은 경과기간)
    r1 = w1[w1.ry <= t0].groupby("ksic").n.sum().reindex(a1.index).fillna(0)
    w0b = krq[(krq.y >= e-5) & (krq.y <= e-3)]; r0 = w0b[w0b.ry <= t0-3].groupby("ksic").n.sum().reindex(a1.index).fillna(0)
    f["grant_sh"] = (r1+1)/(a1.n+2); f["dgrant_sh"] = f.grant_sh - (r0+1)/(a0.n+2)
    f["grant_mom"] = np.log((r1+1)/(r0+1))
    return f.add_suffix(f"_Q{lag}")

def cnfeats(t0, lag):
    e = t0 - lag; inv = cnt[cnt.typ.isin(["1", "8"])]
    w1 = inv[(inv.y >= e-2) & (inv.y <= e)].groupby("ksic").n.sum(); w0 = inv[(inv.y >= e-5) & (inv.y <= e-3)].groupby("ksic").n.sum()
    idx = w1.index.union(w0.index); w1, w0 = w1.reindex(idx).fillna(0), w0.reindex(idx).fillna(0)
    return pd.DataFrame({f"cninv_mom_L{lag}": np.log((w1+1)/(w0+1)), f"cninv_lvl_L{lag}": np.log1p(w1)})

def kab(t0, lag=3):
    a = prio[prio.pc.notna() & (prio.pc != prio.cc)]; e = t0 - lag
    w1 = a[(a.y >= e-2) & (a.y <= e)]; w0 = a[(a.y >= e-5) & (a.y <= e-3)]
    k1 = w1[w1.pc == "KR"].groupby("ksic").n.sum(); k0 = w0[w0.pc == "KR"].groupby("ksic").n.sum()
    t1 = w1.groupby("ksic").n.sum(); t0_ = w0.groupby("ksic").n.sum()
    c1 = w1[w1.pc == "CN"].groupby("ksic").n.sum(); c0 = w0[w0.pc == "CN"].groupby("ksic").n.sum()
    idx = t1.index.union(t0_.index); k1, k0, t1, t0_, c1, c0 = [s.reindex(idx).fillna(0) for s in (k1, k0, t1, t0_, c1, c0)]
    return pd.DataFrame({"kab_mom": np.log((k1+1)/(k0+1)), "kab_sh": (k1+1)/(t1+2), "kab_dsh": (k1+1)/(t1+2)-(k0+1)/(t0_+2), "kab_lvl": np.log1p(k1),
                         "cab_mom": np.log((c1+1)/(c0+1)), "kab_vs_cab": np.log((k1+1)/(c1+1)) - np.log((k0+1)/(c0+1))})

def appf(t0, lag=2):
    e = t0 - lag; f = {}
    for cc in ["KR", "CN"]:
        x = app[app.cc == cc]
        w1 = x[(x.y >= e-2) & (x.y <= e)].groupby("ksic")[["n_ap", "n_new"]].sum()
        w0 = x[(x.y >= e-5) & (x.y <= e-3)].groupby("ksic")[["n_ap", "n_new"]].sum().reindex(w1.index).fillna(0)
        f[f"apmom_{cc}"] = np.log((w1.n_ap+1)/(w0.n_ap+1)); f[f"newmom_{cc}"] = np.log((w1.n_new+1)/(w0.n_new+1))
    return pd.DataFrame(f)

REG = pd.read_parquet(f"{CACHE}/reg_ky.parquet").set_index(["ksic", "y"]).sort_index()
def regfeats(t0):
    """등록원부 권리 동학(사건일 ≤ t0): 3년 창 소멸/설정 비율, 이전 비율, 존속 스톡 성장, 등록당 청구항."""
    g = REG.reset_index(); cum = g[g.y <= t0].groupby("ksic")[["n_reg", "n_ext"]].sum(); cum3 = g[g.y <= t0-3].groupby("ksic")[["n_reg", "n_ext"]].sum()
    w1 = g[g.y.between(t0-2, t0)].groupby("ksic")[["n_reg", "n_ext", "n_ext_fee", "n_trf", "claims"]].sum()
    w0 = g[g.y.between(t0-5, t0-3)].groupby("ksic")[["n_reg", "n_ext", "n_ext_fee", "n_trf", "claims"]].sum().reindex(w1.index).fillna(0)
    st1 = (cum.n_reg - cum.n_ext).reindex(w1.index); st0 = (cum3.n_reg - cum3.n_ext).reindex(w1.index)
    f = pd.DataFrame(index=w1.index)
    f["rg_extr"] = np.log((w1.n_ext_fee+1)/(st0.clip(lower=0)+10)); f["rg_dextr"] = f.rg_extr - np.log((w0.n_ext_fee+1)/(st0.clip(lower=0)+10))
    f["rg_trfr"] = np.log((w1.n_trf+1)/(st1.clip(lower=0)+10)); f["rg_stock_g"] = np.log((st1.clip(lower=0)+10)/(st0.clip(lower=0)+10))
    f["rg_regmom"] = np.log((w1.n_reg+1)/(w0.n_reg+1)); f["rg_clm"] = np.log((w1.claims+1)/(w1.n_reg+1)); f["rg_dclm"] = f.rg_clm - np.log((w0.claims+1)/(w0.n_reg+1))
    return f

KAP = pd.read_parquet(f"{CACHE}/kr_ap.parquet")
def apfeats(t0, lag=2):
    """KR 출원인 구성(출원 시 기재 첫 출원인): 집중도·유형 비중·상위5 제외 기업 모멘텀. 출원연도 ≤ t0−lag."""
    e = t0 - lag; out = {}
    for tag, lo, hi in [("1", e-2, e), ("0", e-5, e-3)]:
        w = KAP[KAP.y.between(lo, hi)]; a = w.groupby(["ksic", "ap"]).n.sum().reset_index()
        tot = a.groupby("ksic").n.sum(); a["s"] = a.n / a.ksic.map(tot)
        hhi = a.groupby("ksic").s.apply(lambda x: (x**2).sum()); top5 = a.sort_values("n", ascending=False).groupby("ksic").head(5)
        t5 = top5.groupby("ksic").n.sum() / tot
        ty = w.groupby(["ksic", "typ"]).n.sum().unstack(fill_value=0).reindex(columns=["C", "I", "P", "U"], fill_value=0)
        cor = w[w.typ == "C"].merge(top5[["ksic", "ap"]].assign(top=1), on=["ksic", "ap"], how="left")
        sme = cor[cor.top.isna()].groupby("ksic").n.sum()
        out[tag] = pd.DataFrame({"tot": tot, "hhi": hhi, "t5": t5, "ui": (ty.U + ty.I) / tot, "p": ty.P / tot, "sme": sme}).fillna(0)
    a1, a0 = out["1"], out["0"].reindex(out["1"].index).fillna(0)
    return pd.DataFrame({"ap_hhi": a1.hhi, "ap_dhhi": a1.hhi - a0.hhi, "ap_t5": a1.t5, "ap_dt5": a1.t5 - a0.t5, "ap_ui": a1.ui, "ap_dui": a1.ui - a0.ui,
                         "ap_p": a1.p, "ap_dp": a1.p - a0.p, "ap_smemom": np.log((a1.sme+1)/(a0.sme+1))})

FP = pd.read_parquet(f"{CACHE}/kr_fam_pit.parquet")
def fampit(t0, lag=2):
    """원점 시점 해외패밀리 비중 변화: 창 w1 은 (외국우선권 or 첫 해외공개연도 ≤ t0), w0 은 같은 경과로 (… ≤ t0−3)."""
    e = t0 - lag
    def sh(lo, hi, cut):
        w = FP[FP.y.between(lo, hi)]; n = w.groupby("ksic").n.sum()
        f = w[(w.dom == 0) | (w.fpy <= cut)].groupby("ksic").n.sum().reindex(n.index).fillna(0)
        return (f+1)/(n+2)
    s1, s0 = sh(e-2, e, t0), sh(e-5, e-3, t0-3)
    return pd.DataFrame({f"ffam_pit_Q{lag}": s1, f"dffam_pit_Q{lag}": s1 - s0.reindex(s1.index)})

rows = []
codes_pi = dict(zip(link.k5, link.pi_code)); codes_ims = dict(zip(link.k5, link.ims_code)); codes_sh = dict(zip(link.k5, link.sh_code))
T_LAST = 2024 if MODE == "future" else 2021
for t0 in range(2005, T_LAST + 1):   # 2010 원점은 모듈 D(수출)용 피처만 쓰고, 주 타깃(종사자)은 NaN 처리
    d = pd.DataFrame(index=IDX); d["t0"] = t0
    d = d.join(pat_feats(pv, t0, 1))
    d = d.join(pat_feats(pv, t0, 2).add_suffix("_L2")).join(pat_feats(pv, t0, 3).add_suffix("_L3"))
    d = d.join(pubfeats(t0, 1)).join(pubfeats(t0, 2))
    d = d.join(krqual(t0, 2)).join(krqual(t0, 3)).join(cnfeats(t0, 2)).join(kab(t0, 3)).join(appf(t0, 2)).join(regfeats(t0)).join(apfeats(t0, 2)).join(fampit(t0, 2))
    # 무역
    d["cnsh"] = C[t0] if t0 in C.columns else np.nan
    d["dcnsh"] = (C[t0] - C[t0-3]) if (t0 in C.columns and t0-3 in C.columns) else np.nan
    d["dcnsh1"] = (C[t0] - C[t0-1]) if t0-1 in C.columns else np.nan
    d["xg3"] = np.log(X[t0]/X[t0-3]); d["xlv"] = np.log(X[t0])
    mw, mc = IM["WLD"], IM["CN"]
    d["mw_g3"] = np.log(mw[t0]/mw[t0-3]); d["mc_g3"] = np.log((mc[t0]+1)/(mc[t0-3]+1))
    xs, ms = X[[t0-2, t0-1, t0]].sum(1), mw[[t0-2, t0-1, t0]].sum(1).reindex(X.index)
    xs0, ms0 = X[[t0-5, t0-4, t0-3]].sum(1), mw[[t0-5, t0-4, t0-3]].sum(1).reindex(X.index)
    d["tsi"] = (xs-ms)/(xs+ms); d["dtsi"] = d.tsi - ((xs0-ms0)/(xs0+ms0)).reindex(d.index)   # 무역특화지수 수준·변화
    d["xm_lr"] = np.log(xs/ms).reindex(d.index)
    # KOSIS 자기 이력(원점 이전)
    for it, nm in [("T02", "emp"), ("T04", "shp"), ("T03", "wage"), ("T06", "va"), ("T01", "est"), ("T07", "cap")]:
        d[f"g3_{nm}"] = lg(K[it], t0, t0-3) if t0-3 >= 2008 else np.nan
        d[f"g1_{nm}"] = lg(K[it], t0, t0-1) if t0-1 >= 2008 else np.nan
        d[f"lv_{nm}"] = lvl(K[it], t0) if t0 >= 2008 else np.nan
    d["prod"] = d.lv_va - d.lv_emp; d["wpw"] = d.lv_wage - d.lv_emp; d["capint"] = d.lv_cap - d.lv_emp
    d["size_est"] = d.lv_emp - d.lv_est
    # 생산지수·수입점유비·출하지수(2·3자리)
    d["pi_g3"] = gser(PI, codes_pi, t0, t0-3); d["pi_g1"] = gser(PI, codes_pi, t0, t0-1)
    d["ims"] = lser(ims, codes_ims, t0); d["dims"] = lser(ims, codes_ims, t0) - lser(ims, codes_ims, t0-3)
    d["shx_g3"] = gser(SHx, codes_sh, t0, t0-3); d["shd_g3"] = gser(SHd, codes_sh, t0, t0-3); d["shxd_g3"] = d.shx_g3 - d.shd_g3
    # 타깃(≤2021 만 존재) — 원점 내 중앙값 차감
    for it, nm in [("T02", "emp"), ("T04", "shp"), ("T03", "wage"), ("T06", "va"), ("T01", "est")]:
        for h in [2, 3, 4]:
            d[f"y_{nm}_h{h}"] = lg(K[it], t0+h, t0) if (t0 >= 2008 and t0+h <= DEV_MAX and t0+h != 2010) else np.nan
    rows.append(d.reset_index())
P = pd.concat(rows, ignore_index=True)
ycols = [c for c in P.columns if c.startswith("y_")]
for c in ycols + [c for c in P.columns if c.startswith("g3_") or c.startswith("g1_")]:
    P[c] = P[c] - P.groupby("t0")[c].transform("median")
P.loc[P.t0 == 2010, ycols] = np.nan   # 2010 경제총조사 연도: 주 타깃 원점 아님
P["y_emp"] = P["y_emp_h3"]
# 산업 고유 추세: 결과가 t0 이전에 끝난 과거 원점(t0' ≤ t0−3)의 종사자 3년 상대성장 순위 평균
yr = P.groupby("t0").y_emp.rank(pct=True); H = pd.DataFrame({"ksic": P.ksic, "t0": P.t0, "yr": yr})
P["hist_y"] = [H[(H.ksic == k) & (H.t0 <= t - 3)].yr.mean() for k, t in zip(P.ksic, P.t0)]
P["hist_n"] = [H[(H.ksic == k) & (H.t0 <= t - 3)].yr.notna().sum() for k, t in zip(P.ksic, P.t0)]
# 산업연관 구조(원점별 이용 가능한 표: t0 ≤ 2018 → 2010표, t0 ≥ 2019 → 2015표)
S = pd.read_parquet(f"{CACHE}/io_struct.parquet")[["ksic", "tab", "io_exr", "io_imr", "io_lab", "io_inv", "io_upm"]]
P["tab"] = np.where(P.t0 >= 2019, 2015, 2010); P = P.merge(S, on=["ksic", "tab"], how="left")
# 상호작용(원점 내 순위 중심화 곱): 기술점유 × 중국 수입노출
r = lambda c: P.groupby("t0")[c].rank(pct=True).fillna(0.5) - 0.5
P["ix_cn_dsh"] = r("cnsh") * r("dsh_KR_L2"); P["ix_cn_mom"] = r("cnsh") * r("mom_KR_L2"); P["ix_dcn_dsh"] = r("dcnsh") * r("dsh_KR_L2")
P["ix_cn_ffam"] = r("cnsh") * r("dffam_sh_Q2")
# 합성 지수(원점 내 순위 평균, 부호 사전 고정): 공선성 완화
P["kr_md"] = r("mom_KR_L2") + r("dsh_KR_L2")                 # KR 모멘텀+점유변화
P["glob_idx"] = r("frgn_mom") + r("kab_mom")                 # 해외 모멘텀+한국발 해외출원 모멘텀
P["pat_idx"] = r("mom_KR_L2") + r("dsh_KR_L2") + r("frgn_mom") + r("kab_mom") - r("dffam_pit_Q2")
if MODE == "dev":
    assert P.loc[P.t0 > 2018, "y_emp"].notna().sum() == 0, "최종시험 구간 타깃 누출"
P.to_parquet(f"{CACHE}/panel_n1.parquet" if MODE == "dev" else f"{CACHE}/panel_n1_future.parquet")
print("행", len(P), "원점", sorted(P.t0.unique()), "열", P.shape[1])
print("y_emp 비결측 원점별", P.groupby("t0").y_emp.count().to_dict())
cov = P[P.t0.between(2008, 2018)].drop(columns=["ksic", "t0"]).notna().mean().round(3)
if MODE == "dev": cov.to_csv(f"{HERE}/ext_link_feature_coverage.csv", encoding="utf-8-sig", header=["coverage_dev"])
print(cov[cov < 0.95].to_string())
