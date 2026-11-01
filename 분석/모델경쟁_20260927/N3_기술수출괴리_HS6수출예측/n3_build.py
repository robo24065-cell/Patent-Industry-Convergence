# -*- coding: utf-8 -*-
"""N3 패널 구축 — HS6 품목 × 원점 t0, 전 후보 피처 포함.
사용: python n3_build.py X|M [max_t0]
  X = 한국 HS6 수출(G2 본 과제), M = 한국 HS6 수입(G1 보조).
  개발 모드 기본 max_t0 = 2018 (결과 t0+1..t0+3 ≤ 2021). 🔒 t0=2021 행은 final 모드(final_run_r2.py)에서만 만든다.
타깃: y_raw = log(Σ[t0+1..t0+3] / Σ[t0-2..t0]), y = y_raw − 원점 내 중앙값. 9년(t0-5..t0+3) 모두 양(+)인 품목만(파일럿과 동일, HS 개정 단절 제거).
피처(모두 ≤ t0 정보):
  OWN  lvl mom3 mom1 vol imp_ratio imp_mom k_mom3 (파일럿 B1)
  PAT  KSIC 5개국 특허 수준·점유(pl_*, p_*, q_*) — 최초공표연도 ≤ t0
  X5   5개국 수출(Comtrade) HS6 한국 점유·중국 점유 및 변화(수집연도 ye ≤ t0), KSIC 수준 괴리(기술점유−수출점유)
  KS   KOSIS 광업제조업조사 KSIC 출하·종사자·부가가치 수준·3년 변화(≤ t0, 2010 결측은 직전 연도)
  IO   2010 산업연관표 전방(수요산업) 출하 모멘텀·특허 포지션, 후방(공급산업) 모멘텀, 수출비중
  CN   KSIC 대중 수입점유(≤ t0) 및 3년 변화, 품목 대중 수입비중
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_n3 import *

X5_SPARSE = [2004, 2006, 2009, 2012, 2015, 2018, 2021]   # 5개국 수출 사용 연도(3년 간격, 동결 규칙)


def build(flow="X", max_t0=2018, trade_max=None, verbose=True, x5mode="sparse", allow_final=False, t0_min=2005,
          require_future=True, pat_file="pat_panel.parquet", kosis_max=KOSIS_MAX_YEAR, x5_max=2021):
    """allow_final=False 이면 결과 구간이 2021을 넘는 원점·무역연도를 만들지 않는다(🔒)."""
    trade_max = trade_max or (max_t0 + 3 if require_future else max_t0)
    if not allow_final:
        assert trade_max <= TRADE_MAX_YEAR and (max_t0 + 3 <= TRADE_MAX_YEAR or not require_future), "최종시험 구간 접근 금지"
    m = load_map()
    imp = load_imports(max_year=trade_max)
    Wm = trade_panel(imp, "WLD"); Cm = trade_panel(imp, "CN").reindex_like(Wm)
    if flow == "M":
        W = Wm; C = Cm
    else:
        ex = load_exports(max_year=trade_max)
        W = ex.pivot_table(index="hs6", columns="year", values="value", aggfunc="sum")
        C = Wm.reindex_like(W); CNi = Cm.reindex_like(W)
    pan = pd.read_parquet(os.path.join(CACHE, pat_file))
    K = load_kosis(kosis_max)
    Z, io_exp, ksub = load_io()
    cnk = load_cn_ksic()
    x5 = load_5c_exports(max_year=x5_max)
    if x5mode == "sparse":
        x5 = x5[x5.year.isin(X5_SPARSE)]
    x5y = sorted(x5.year.unique())
    x5p = {y: x5[x5.year == y].pivot_table(index="hs6", columns="cc", values="export_usd", aggfunc="sum")
           .reindex(columns=list(COUNTRIES.values())).fillna(0) for y in x5y}
    hk = m.merge(pd.Series(W.index, name="hs6"), on="hs6")
    # IO 정적 피처: 소분류 수출비중
    io_tot = Z.sum(axis=1)
    rows = []
    for t0 in range(t0_min, max_t0 + 1):
        yrsA, yrsB, yrsF = [t0 - 2, t0 - 1, t0], [t0 - 5, t0 - 4, t0 - 3], [t0 + 1, t0 + 2, t0 + 3]
        need = yrsB + yrsA + (yrsF if require_future else [])
        if min(need) < W.columns.min() or max(need) > W.columns.max():
            continue
        sub = W[need]
        s = sub[(sub > 0).all(axis=1)]
        A, B = s[yrsA].sum(1), s[yrsB].sum(1)
        Fw = s[yrsF].sum(1) if require_future else A * np.nan
        df = pd.DataFrame({"hs6": s.index, "t0": t0})
        df["lvl"] = np.log(A.values)
        df["mom3"] = np.log(A / B).values
        df["mom1"] = np.log(s[t0] / s[t0 - 1]).values
        df["vol"] = np.log(s[yrsB + yrsA]).diff(axis=1).std(axis=1).values
        # 장기 이력(≤ t0): 직전 창 모멘텀(t0-8..t0-6 → B), 고점 대비 낙폭
        yrsL = [t0 - 8, t0 - 7, t0 - 6]
        if min(yrsL) >= W.columns.min():
            L = W.loc[s.index, yrsL]
            Ls = L.sum(1).where((L > 0).all(axis=1))
            df["mom_prev"] = np.log(B / Ls).values
        else:
            df["mom_prev"] = np.nan
        df["dd"] = np.log(s[t0] / s[yrsB + yrsA].max(axis=1)).values
        cA = C.loc[s.index, yrsA].fillna(0).sum(1); cB = C.loc[s.index, yrsB].fillna(0).sum(1)
        if flow == "M":
            df["cn_sh"] = (cA / A).clip(0, 1).values; df["cn_sh_chg"] = (cA / A - cB / B).clip(-1, 1).values
        else:
            df["imp_ratio"] = np.log((cA + 1) / A).values; df["imp_mom"] = np.log((cA + 1) / (cB + 1)).values
            nA = CNi.loc[s.index, yrsA].fillna(0).sum(1); nB = CNi.loc[s.index, yrsB].fillna(0).sum(1)
            df["im_cn_sh"] = (nA / (cA + 1)).clip(0, 1).values
            df["im_cn_chg"] = (nA / (cA + 1) - nB / (cB + 1)).clip(-1, 1).values
        df["y_raw"] = np.log(Fw / A).values
        df["share_in_k"] = np.nan
        # --- KSIC 수준 무역 모멘텀(파일럿)
        tmp = pd.DataFrame({"hs6": s.index, "A": A.values, "B": B.values}).merge(hk, on="hs6")
        kt = tmp.groupby("k5")[["A", "B"]].sum(); kt["k_mom3"] = np.log(kt.A / kt.B); kt["k_lvl"] = np.log(kt.A)
        # 품목의 KSIC 내 비중(단일 KSIC 기준 첫 연결)
        # --- 특허
        pf = patent_feats(t0, pan)
        # --- 5개국 수출(Comtrade) : ye ≤ t0 최근 수집연도
        ye_list = [y for y in x5y if y <= t0]
        x5h = pd.DataFrame(index=s.index)
        x5k = pd.DataFrame(index=pd.Index(pf.k5, name="k5"))
        if ye_list:
            ye = max(ye_list); P = x5p[ye].reindex(s.index)
            tot = P.sum(axis=1)
            for cc in ["KR", "CN", "JP", "US", "DE"]:
                x5h["x5_" + cc] = (P[cc] / tot).where(tot > 0)
            x5h["x5_age"] = t0 - ye
            yp_list = [y for y in x5y if y <= ye - 3]
            if yp_list:
                yp = max(yp_list); Pp = x5p[yp].reindex(s.index); tp = Pp.sum(axis=1)
                for cc in ["KR", "CN"]:
                    x5h["x5c_" + cc] = (x5h["x5_" + cc] - (Pp[cc] / tp).where(tp > 0)) * 3.0 / (ye - yp)
                x5h["x5_tot_m"] = np.log((tot + 1) / (tp + 1)) * 3.0 / (ye - yp)   # 5개국 합(세계수요 대리) 변화
            # KSIC 수준 수출점유(연결 HS6 합, 표본 무관 전체 연결 품목)
            Pk = x5p[ye].merge(m[["hs6", "k5"]], left_index=True, right_on="hs6").groupby("k5")[["KR", "CN", "JP", "US", "DE"]].sum()
            tk = Pk.sum(axis=1)
            for cc in ["KR", "CN", "JP", "US", "DE"]:
                x5k["xk_" + cc] = (Pk[cc] / tk).reindex(x5k.index)
        x5k = x5k.reset_index()
        # --- KOSIS
        kf = kosis_feats(t0, K)
        # --- IO: 소분류별 출하 모멘텀(KOSIS 전체 KSIC → 소분류), 특허 포지션
        io_rows = None
        if kf is not None and "ks_m_ship" in kf:
            km = kf[["k5", "ks_m_ship", "ks_l_ship"]].dropna().merge(ksub, on="k5")
            km["S1"] = np.exp(km.ks_l_ship) * km.w
            km["S0"] = km.S1 / np.exp(km.ks_m_ship)   # (연율 보정 3년 기준 근사)
            sm = km.groupby("sub")[["S1", "S0"]].sum()
            mom_sub = np.log(sm.S1 / sm.S0)
        else:
            mom_sub = pd.Series(dtype=float)
        pk = pf.merge(ksub[ksub.k5.isin(pf.k5)], on="k5")
        pk["cnt"] = np.expm1(pk.pl_KR) + np.expm1(pk.pl_FOR) + np.expm1(pk.pl_KRout)
        psub = pk.groupby("sub").apply(lambda g: pd.Series({"p_outshare": np.average(g.p_outshare, weights=g.w),
                                                            "pl_KR": np.log1p((np.expm1(g.pl_KR) * g.w).sum())}), include_groups=False)
        io_list = []
        subs = Z.index
        Zm = Z.copy(); np.fill_diagonal(Zm.values, 0.0)
        for sct in subs:
            r = {"sub": sct}
            wd = Zm.loc[sct]; wu = Zm[sct]
            if len(mom_sub):
                md = mom_sub.reindex(subs)
                ok = md.notna() & (wd > 0)
                r["io_dn_m"] = float((wd[ok] * md[ok]).sum() / wd[ok].sum()) if wd[ok].sum() > 0 else np.nan
                r["io_dn_cov"] = float(wd[ok].sum() / wd.sum()) if wd.sum() > 0 else np.nan
                ok = md.notna() & (wu > 0)
                r["io_up_m"] = float((wu[ok] * md[ok]).sum() / wu[ok].sum()) if wu[ok].sum() > 0 else np.nan
            okp = wd.index.isin(psub.index) & (wd > 0)
            if wd[okp].sum() > 0:
                ww = wd[okp]; pp = psub.reindex(ww.index)
                r["io_dn_pout"] = float((ww * pp.p_outshare).sum() / ww.sum())
                r["io_dn_pKR"] = float((ww * pp.pl_KR).sum() / ww.sum())
                r["io_dn_pcov"] = float(ww.sum() / wd.sum())
            r["io_exp_ratio"] = float(io_exp.get(sct, np.nan) / (io_tot.get(sct, np.nan) + io_exp.get(sct, 0) + 1e-9))
            io_list.append(r)
        iof = pd.DataFrame(io_list).set_index("sub")
        ioK = ksub.merge(iof, left_on="sub", right_index=True)
        ioK = ioK.groupby("k5").apply(lambda g: pd.Series({c: np.nansum(g[c] * g.w) / g.w[g[c].notna()].sum() if g[c].notna().any() else np.nan
                                                            for c in iof.columns}), include_groups=False).reset_index()
        # --- KSIC 대중 수입점유
        cy = [y for y in cnk.columns if y <= t0]
        cnf = pd.DataFrame({"k5": cnk.index, "cnk_sh": cnk[max(cy)].values,
                            "cnk_chg": (cnk[max(cy)] - cnk[max(cy) - 3]).values if (max(cy) - 3) in cnk.columns else np.nan})
        # --- KSIC 피처 결합
        KF = pf.merge(kt[["k_mom3", "k_lvl"]].reset_index(), on="k5", how="left").merge(x5k, on="k5", how="left")
        if kf is not None:
            KF = KF.merge(kf, on="k5", how="left")
        KF = KF.merge(ioK, on="k5", how="left").merge(cnf, on="k5", how="left")
        # 괴리 피처: 기술 점유(한국 기원 해외출원 점유) − 수출 점유(5개국 중 한국)
        if "xk_KR" in KF:
            KF["gap_kr"] = KF.q_KRout - KF.xk_KR
            KF["gap_kr_lr"] = np.log((KF.q_KRout + 0.01) / (KF.xk_KR + 0.01))
            KF["gap_cn"] = KF.q_CNout - KF.xk_CN
            KF["gap_jp"] = KF.q_JPout - KF.xk_JP
            KF["gap_kr5"] = KF.p_krshare5 - KF.xk_KR
        # 생산기반 대비 수출: KSIC 수출(USD) 대비 출하(백만원) 로그비 — 수준 차이는 환율로 흔들리므로 모멘텀 차이도
        if "ks_l_ship" in KF:
            KF["ks_expint"] = KF.k_lvl - KF.ks_l_ship
            if "ks_m_ship" in KF:
                KF["ks_exp_minus_ship_m"] = KF.k_mom3 - KF.ks_m_ship
        # 품목 결합: 단순평균(파일럿) + 출하 가중(e)
        H = hk.merge(KF, on="k5")
        num = [c for c in H.columns if c not in ("io", "k5", "hs6")]
        ff = H.groupby("hs6")[num].mean()
        if "ks_l_ship" in H:
            H["wt"] = np.exp(H.ks_l_ship.fillna(H.ks_l_ship.min()))
            H["wt"] = H.wt / H.groupby("hs6").wt.transform("sum")
            wcols = [c for c in num if c.startswith(("pl_", "p_", "q_"))]
            fw = H[wcols].multiply(H.wt, axis=0).groupby(H.hs6).sum()
            fw.columns = ["w_" + c for c in wcols]
            ff = ff.join(fw)
        ff["n_ksic"] = hk.groupby("hs6").size().reindex(ff.index)
        df = df.merge(ff, left_on="hs6", right_index=True, how="left").merge(x5h, left_on="hs6", right_index=True, how="left")
        if "xk_KR" in df and "x5_KR" in df:
            df["gap_kr_h"] = df.q_KRout - df.x5_KR
            df["gap_cn_h"] = df.q_CNout - df.x5_CN
            df["gap_jp_h"] = df.q_JPout - df.x5_JP
            df["gap_us_h"] = df.q_USout - df.x5_US
            df["gap_eu_h"] = df.q_EUout - df.x5_DE
            df["x5_KR_rel"] = df.x5_KR - df.xk_KR
            if "x5c_KR" in df:
                df["gapm_kr_h"] = (df.q_KRout - df.q_KRout_B) - df.x5c_KR
                df["gapm_cn_h"] = (df.q_CNout - df.q_CNout_B) - df.x5c_CN
        # 품목의 KSIC 내 비중
        df["share_in_k"] = df.lvl - df.k_lvl
        rows.append(df)
        if verbose:
            print(t0, len(df), flush=True)
    D = pd.concat(rows, ignore_index=True).copy()
    D["y"] = D.y_raw - D.groupby("t0").y_raw.transform("median")
    D["k5"] = D.hs6.map(m.groupby("hs6").k5.first())
    return D


if __name__ == "__main__":
    FLOW = sys.argv[1] if len(sys.argv) > 1 else "X"
    MAX_T0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2018
    X5MODE = sys.argv[3] if len(sys.argv) > 3 else "sparse"
    assert MAX_T0 <= 2018, "개발 패널은 t0 ≤ 2018 (final은 final_run_r2.py)"
    D = build(FLOW, MAX_T0, x5mode=X5MODE)
    out = os.path.join(CACHE, f"panel_{FLOW}{'' if X5MODE == 'sparse' else '_' + X5MODE}.parquet")
    D.to_parquet(out, index=False)
    print("표본", D.groupby("t0").size().to_dict())
    print("열", len(D.columns))
    print(D.isna().mean().round(3)[D.isna().mean() > 0].to_string())
