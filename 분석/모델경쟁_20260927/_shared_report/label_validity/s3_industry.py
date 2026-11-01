# -*- coding: utf-8 -*-
"""3) 산업 단위: KSIC별 연간 특허 수(제공 라벨 vs 연계표 배정) ↔ KOSIS 광업제조업조사 5자리(출하·종사자·사업체), 무역(ksic_hs6_map 경유 수출·수입).
개발 기간만: 특허 출원 2005–2017, KOSIS 2008–2017(2010 경제총조사로 없음), 무역 2005–2017.
설계 A(5자리, 50개): 제공 = 그 KSIC 라벨 특허 수 / 연계표 = 그 KSIC가 속한 연계표 그룹으로 배정된 특허 수
   (연계표는 5자리를 지목할 수 없으므로 연계표만 가진 분석자가 5자리 산업에 쓸 수 있는 최선의 계열).
설계 B(그룹, 27개): 제공 = 그룹 안 라벨 특허 수 / 연계표 = 그룹 배정 특허 수 / 외부지표 = 그룹 안 50개 코드 합.
주의: 모집단은 제공 데이터(50개 KSIC에 분류된 특허)다. 연계표 측은 이미 제공 라벨로 고른 모집단 위에서 배정되므로 연계표에 유리한 쪽이다.
지표: (a) 단면 수준 Spearman, (b) 장기 증가율 상관(동시·3년 선행), (c) 패널 Δlog(산업·연도 고정효과 제거) 상관, 시차 0–3.
차이(제공−연계표)는 산업 단위 부트스트랩 800회 95% CI."""
import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lv_common import *
from scipy.stats import spearmanr, pearsonr

Y0, Y1 = 2005, DEV_END
kr = pd.read_parquet(os.path.join(OUT, "cache_kr_cw.parquet"))
kr = kr[(kr.appl_year >= Y0) & (kr.appl_year <= Y1)]
K50 = sorted(kr.ksic.unique())
k2g = kr.drop_duplicates("ksic").set_index("ksic").grp
years = list(range(Y0, Y1 + 1))
prov5 = kr.groupby(["ksic", "appl_year"]).size().unstack(fill_value=0).reindex(index=K50, columns=years, fill_value=0)
cwg = kr[kr.cw.notna()].groupby(["cw", "appl_year"]).size().unstack(fill_value=0).reindex(columns=years, fill_value=0)
provg = kr.groupby(["grp", "appl_year"]).size().unstack(fill_value=0).reindex(columns=years, fill_value=0)
cw5 = pd.DataFrame({k: cwg.loc[k2g[k]] if k2g[k] in cwg.index else 0 for k in K50}).T.reindex(columns=years)
log("특허 패널", prov5.shape, "합", int(prov5.values.sum()))

# 외부 지표
ko = pd.read_csv(os.path.join(BASE, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
ko = ko[(ko.기준 == "10차") & (ko.마스킹 == 0) & ko.항목코드.isin(["T01", "T02", "T04"])]
ext = {}
for code, nm in (("T04", "출하액"), ("T02", "종사자수"), ("T01", "사업체수")):
    ext[nm] = ko[ko.항목코드 == code].pivot_table(index="세세분류코드", columns="연도", values="값", aggfunc="sum").reindex(index=K50)
mp = pd.read_csv(os.path.join(BASE, r"분석\fusion\ksic_hs6_map.csv"), dtype=str).drop_duplicates(["k5", "hs6"])
mp["w"] = 1.0 / mp.groupby("hs6").k5.transform("nunique")
ex = pd.read_csv(os.path.join(EXT, "trade_kr_export_hs6_uv_2000_2021.csv"), dtype={"hs6": str}, usecols=["year", "hs6", "export_usd"])
im = pd.read_csv(os.path.join(BASE, r"분석\fusion\hs6_import_yearly.csv"), dtype={"hs6": str})
im = im[im.partner == "WLD"].rename(columns={"value": "import_usd"})
for nm, d, col in (("수출액", ex, "export_usd"), ("수입액", im, "import_usd")):
    z = d.merge(mp, on="hs6")
    z["v"] = z[col] * z.w
    ext[nm] = z.pivot_table(index="k5", columns="year", values="v", aggfunc="sum").reindex(index=K50)
for nm in ext:
    ext[nm] = ext[nm].reindex(columns=[y for y in years if y in ext[nm].columns])
    log(nm, "산업", int(ext[nm].notna().any(axis=1).sum()), "연도", list(ext[nm].columns)[:2], "…", list(ext[nm].columns)[-1])


def to_group(df):
    return df.groupby(k2g.reindex(df.index).values).sum(min_count=1)


def metrics(P, Yd):
    """P: 특허 패널(산업×연도), Yd: 외부지표 패널. 반환: dict 지표→값"""
    idx = P.index.intersection(Yd.dropna(how="all").index)
    P, Yd = P.loc[idx].astype(float), Yd.loc[idx].astype(float)
    lp, ly = np.log1p(P), np.log(Yd.where(Yd > 0))
    yc = [y for y in Yd.columns if y >= 2008]
    out = {"n": len(idx)}
    a, b = lp[yc].mean(axis=1), ly[yc].mean(axis=1)
    ok = a.notna() & b.notna()
    out["수준_Spearman"] = spearmanr(a[ok], b[ok])[0]
    # 장기 증가율: Y 2008–09 → 2016–17
    gy = ly[[2016, 2017]].mean(axis=1) - ly[[2008, 2009]].mean(axis=1) if 2008 in ly.columns and 2009 in ly.columns else None
    for lag in (0, 3):
        gp = lp[[2016 - lag, 2017 - lag]].mean(axis=1) - lp[[2008 - lag, 2009 - lag]].mean(axis=1)
        ok = gp.notna() & gy.notna()
        out[f"장기증가율_r_lag{lag}"] = pearsonr(gp[ok], gy[ok])[0]
        out[f"장기증가율_rho_lag{lag}"] = spearmanr(gp[ok], gy[ok])[0]
    # 패널 Δlog (연속 연도만), 산업·연도 고정효과 제거
    for lag in (0, 1, 2, 3):
        rows = []
        for t in Yd.columns:
            if t - 1 in Yd.columns and t - 1 - lag >= Y0:
                dy = ly[t] - ly[t - 1]
                dp = lp[t - lag] - lp[t - 1 - lag]
                rows.append(pd.DataFrame({"i": idx, "t": t, "dy": dy.values, "dp": dp.values}))
        D = pd.concat(rows).dropna()
        for c in ("dy", "dp"):
            D[c] = D[c] - D.groupby("i")[c].transform("mean")
            D[c] = D[c] - D.groupby("t")[c].transform("mean")
        out[f"패널Δlog_r_lag{lag}"] = float(np.corrcoef(D.dp, D.dy)[0, 1])
    return out


rng = np.random.default_rng(0)
rows = []
for design, Pp, Pc, conv in (("A_5자리(50)", prov5, cw5, lambda d: d), ("B_그룹(27)", provg, cwg.reindex(provg.index).fillna(0), to_group)):
    for nm, Yd in ext.items():
        Yg = conv(Yd)
        mp_ = metrics(Pp, Yg); mc_ = metrics(Pc, Yg)
        idx = Pp.index.intersection(Yg.dropna(how="all").index).values
        bs = []
        for b in range(int(os.environ.get("NB", "800"))):
            s = rng.choice(idx, len(idx))
            # 부트스트랩 표본에 같은 산업이 중복되면 인덱스를 새로 붙인다
            nid = [f"{x}_{j}" for j, x in enumerate(s)]
            Pb = Pp.loc[s].set_axis(nid); Cb = Pc.loc[s].set_axis(nid); Yb = Yg.loc[s].set_axis(nid)
            try:
                m1, m2 = metrics(Pb, Yb), metrics(Cb, Yb)
                bs.append({k: m1[k] - m2[k] for k in m1 if k != "n"})
            except Exception:
                pass
        bs = pd.DataFrame(bs)
        for k in mp_:
            if k == "n":
                continue
            lo, hi = np.nanpercentile(bs[k], [2.5, 97.5])
            rows.append({"설계": design, "외부지표": nm, "지표": k, "n산업": mp_["n"], "제공라벨": round(mp_[k], 4),
                         "연계표": round(mc_[k], 4), "차이": round(mp_[k] - mc_[k], 4), "차이_95CI_lo": round(lo, 4), "차이_95CI_hi": round(hi, 4)})
        log(design, nm, "done")
O = pd.DataFrame(rows)
O.to_csv(os.path.join(OUT, "lv3_industry_correlations.csv"), index=False, encoding="utf-8-sig")
print(O.to_string())
# 요약: 제공>연계표 비율, CI가 0 위/아래
S = O.groupby("설계").apply(lambda g: pd.Series({"비교수": len(g), "제공>연계표": int((g.차이 > 0).sum()),
                                               "CI>0": int((g.차이_95CI_lo > 0).sum()), "CI<0": int((g.차이_95CI_hi < 0).sum())}))
print(S)
S.to_csv(os.path.join(OUT, "lv3_summary.csv"), encoding="utf-8-sig")
prov5.to_csv(os.path.join(OUT, "lv3_patent_counts_provided_5d.csv"), encoding="utf-8-sig")
cwg.to_csv(os.path.join(OUT, "lv3_patent_counts_crosswalk_group.csv"), encoding="utf-8-sig")
log("done")
