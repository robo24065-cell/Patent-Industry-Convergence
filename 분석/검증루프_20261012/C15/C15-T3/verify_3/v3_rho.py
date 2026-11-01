# -*- coding: utf-8 -*-
r"""C15-T3 verify_3 (반박 검증): C-RCA-PKR·C-LAND-pkr_H를 대안 A/B로 독립 재계산. duckdb·s1/s4 코드 미사용.
라벨: (1) 직접 배정(alt_KR 제공 라벨 + rec_conc sg 표 + links REG), (2) s1 k_reg/k_pure, (3) verify_1 v_pkr 와 대조.
읽기만. 쓰기: verify_3/ 안."""
import sys, json, time
sys.stdout.reconfigure(encoding="utf-8"); sys.dont_write_bytecode = True
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.dataset as ds
from scipy.stats import rankdata
BASE = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"; C5 = MC + "/_shared_report/fusion_advantage/c5_R10"
W = BASE + "/분석/검증루프_20261012/C15/C15-T3"; V = W + "/verify_3"
RC = MC + "/T3_기술셀부상예측/ESSENTIAL_R7/rec_conc.parquet"
LINKS = BASE + "/분석/검증루프_20261012/C10/C10-T3/cache/links.parquet"
t0 = time.time(); log = lambda *a: print(*a, f"[{time.time()-t0:.0f}s]", flush=True)
CCS = ["KR", "US", "JP", "CN", "EP"]; FOR = CCS[1:]
OUT = {}

def mode_tab(df, key, val="k", minn=1):
    g = df.groupby([key, val]).size().rename("n").reset_index()
    tot = g.groupby(key).n.transform("sum"); g = g[tot >= minn]
    g["ki"] = g[val].astype(int)
    g = g.sort_values([key, "n", "ki"], ascending=[True, False, True]).drop_duplicates(key)
    return g.set_index(key)[val]

# ---- KR 제공 라벨, IPC 표(출원<=2019 KR 문서 전체) ----
kr = pq.read_table(f"{C5}/alt_assign/alt_KR.parquet", columns=["appl_no", "appl_year", "ksic_prov"]).to_pandas()
rcd = ds.dataset(RC)
krsg = rcd.to_table(columns=["appl_no", "sg"], filter=ds.field("cc") == "KR").to_pandas()
OUT["kr_docs"] = len(kr); OUT["kr_rc_rows"] = len(krsg); OUT["kr_rc_dup"] = int(krsg.appl_no.duplicated().sum())
krk = kr[kr.appl_year <= 2019].merge(krsg, on="appl_no", how="inner"); del krsg
krk = krk[krk.sg.notna() & (krk.sg != "")]
krk = krk.assign(mg=krk.sg.str.split("/").str[0], sc=krk.sg.str[:4]).rename(columns={"ksic_prov": "k"})
TAB = {lv: mode_tab(krk[[lv, "k"]], lv, minn=3) for lv in ("sg", "mg", "sc")}
log("IPC 표", {k: len(v) for k, v in TAB.items()})
kprov = kr.set_index("appl_no").ksic_prov
lk = pq.read_table(LINKS).to_pandas(); lk = lk[lk.link == "REG"]
OUT["reg_rows"] = len(lk); OUT["reg_dup_rows"] = int(lk.duplicated(["cc", "appl_no", "kr_appl_no"]).sum())
lk["k"] = lk.kr_appl_no.map(kprov); OUT["reg_kr_unmatched"] = int(lk.k.isna().sum()); lk = lk.dropna(subset=["k"])
lkd = lk.drop_duplicates(["cc", "appl_no", "kr_appl_no"])
lkd["key"] = lkd.cc + "|" + lkd.appl_no; lk["key"] = lk.cc + "|" + lk.appl_no
FAM = mode_tab(lkd[["key", "k"]], "key"); FAMnd = mode_tab(lk[["key", "k"]], "key")
log("REG", OUT["reg_rows"], OUT["reg_dup_rows"], OUT["reg_kr_unmatched"], len(FAM))

# ---- 지표 창 문서(발명, 출원 2015–19) ----
cols = ["cc", "appl_no", "appl_year", "kr_origin", "ksic_prov", "code", "k_txt", "k_free", "k_pkr"]
parts = []; cmp = {}
for cc in CCS:
    d = pq.read_table(f"{C5}/alt_assign/alt_{cc}.parquet", columns=cols + ["is_inv"]).to_pandas()
    d = d[d.is_inv & d.appl_year.between(2015, 2019)].drop(columns="is_inv")
    if cc != "KR":
        sg = rcd.to_table(columns=["appl_no", "sg"], filter=ds.field("cc") == cc).to_pandas()
        d = d.merge(sg, on="appl_no", how="left"); del sg
        s = d.sg.fillna(""); mg = s.str.split("/").str[0]; sc = s.str[:4]
        pure = s.map(TAB["sg"]); pure = pure.fillna(mg.map(TAB["mg"])); pure = pure.fillna(sc.map(TAB["sc"]))
        key = cc + "|" + d.appl_no
        d["k_pureM"] = pure.values; d["k_regM"] = key.map(FAM).fillna(pure).values; d["k_regMnd"] = key.map(FAMnd).fillna(pure).values
        s1 = pq.read_table(f"{W}/cache/pkr_alt_{cc}.parquet", columns=["appl_no", "k_reg", "k_pure", "k_r8"]).to_pandas()
        v1 = pq.read_table(f"{W}/verify_1/cache/v_pkr_{cc}.parquet").to_pandas().rename(columns={"k": "k_v1"})
        d = d.merge(s1, on="appl_no", how="left").merge(v1, on="appl_no", how="left")
        eq = lambda a, b: float((d[a].fillna("NA") == d[b].fillna("NA")).mean())
        cmp[cc] = dict(n=len(d), regM_vs_s1=eq("k_regM", "k_reg"), regM_vs_v1=eq("k_regM", "k_v1"), regMnd_vs_s1=eq("k_regMnd", "k_reg"),
                       pureM_vs_s1=eq("k_pureM", "k_pure"), r8_vs_stored=eq("k_r8", "k_pkr"), famshare=float(key.isin(FAM.index).mean()),
                       agree_prov=dict(stored=float((d.k_pkr == d.ksic_prov).mean()), regM=float((d.k_regM == d.ksic_prov).mean()), pureM=float((d.k_pureM == d.ksic_prov).mean())))
        d = d.drop(columns=["sg"])
    else:
        for c in ("k_regM", "k_regMnd", "k_pureM", "k_reg", "k_pure", "k_r8", "k_v1"): d[c] = d.ksic_prov
    parts.append(d); log(cc, len(d))
D = pd.concat(parts, ignore_index=True); del parts
OUT["label_compare_window"] = cmp; log(json.dumps(cmp, ensure_ascii=False))

km = pd.read_csv(MC + "/T3_기술셀부상예측/ESSENTIAL_R7/conc_ksic_map.csv", dtype=str)[["ksic", "code"]]
km["w"] = 1 / km.groupby("code").ksic.transform("size"); K50 = sorted(km.ksic)
def agg(col):
    x = D[D[col].notna()]; return x.groupby([col, "cc", "appl_year", "kr_origin"]).size().rename("n").reset_index().rename(columns={col: "k", "appl_year": "y", "kr_origin": "o"})
C = {"prov": agg("ksic_prov"), "txt": agg("k_txt"), "free": agg("k_free")}
cw = D[["code", "cc", "appl_year", "kr_origin"]].merge(km, on="code")
C["cweq"] = cw.groupby(["ksic", "cc", "appl_year", "kr_origin"]).w.sum().rename("n").reset_index().rename(columns={"ksic": "k", "appl_year": "y", "kr_origin": "o"})
ARMS = {"stored": "k_pkr", "r8": "k_r8", "regM": "k_regM", "regMnd": "k_regMnd", "pureM": "k_pureM", "s1reg": "k_reg", "s1pure": "k_pure", "v1reg": "k_v1"}
for a, c in ARMS.items(): C["pkr_" + a] = agg(c)

ek = pd.read_csv(MC + "/_shared_report/fusion_advantage/map5/m4_exports_ksic_5rep.csv", dtype={"k5": str})
v = ek[ek.year.isin([2015, 2016, 2017])].groupby(["k5", "rep"]).v.sum().unstack(); sh = v / v.sum()
RCA = sh.div(sh.mean(1), axis=0).KR.reindex(K50)

def rho(a, b):
    ra, rb = rankdata(a), rankdata(b); return float(np.corrcoef(ra, rb)[0, 1])
def abroad(lab, y0, y1):
    d = C[lab]; d = d[(d.cc != "KR") & d.y.between(y0, y1)]
    k = d[d.o].groupby("k").n.sum().reindex(K50).fillna(0); a = d.groupby("k").n.sum().reindex(K50)
    return (k / a) / (k.sum() / a.sum())
def counts(lkr, lfor, y0, y1):
    o = {}
    for cc in CCS:
        d = C[lkr if cc == "KR" else lfor]; o[cc] = d[(d.cc == cc) & d.y.between(y0, y1)].groupby("k").n.sum()
    n = pd.DataFrame(o).reindex(K50).fillna(0); s = n / n.sum(); return s.div(s.mean(1), axis=0)
def boot(a, c, I):
    out = np.empty(len(I))
    for j, i in enumerate(I):
        x, y = a[i], c[i]
        out[j] = np.nan if (np.ptp(x) == 0 or np.ptp(y) == 0) else rho(x, y)
    return out

res = {}
for arm in ARMS:
    pk = "pkr_" + arm; rng = np.random.default_rng(20261005); B = 2000
    abp = abroad("prov", 2017, 2019)
    rr = {}
    for lab in ["cweq", "txt", "free", pk]:
        d = pd.concat([abp, abroad(lab, 2017, 2019), RCA], axis=1).dropna(); a, b, c = d.iloc[:, 0].values, d.iloc[:, 1].values, d.iloc[:, 2].values
        n = len(d); I = rng.integers(0, n, (B, n))
        if lab != pk: continue
        bp, ba = boot(a, c, I), boot(b, c, I); bd = bp - ba
        pct = lambda z: [round(float(np.nanpercentile(z, 2.5)), 3), round(float(np.nanpercentile(z, 97.5)), 3)]
        rr = dict(n=n, rho_prov=rho(a, c), rho_alt=rho(b, c), ci_alt=pct(ba), d=rho(a, c) - rho(b, c), ci_d=pct(bd))
        lo, hi = rr["ci_d"]; rr["verdict"] = "제공 우위" if (rr["d"] > 0 and lo > 0) else ("대체 우위" if (rr["d"] < 0 and hi < 0) else "같다")
        rr["keeps"] = rr["ci_alt"][0] > 0
    rp = counts("prov", "prov", 2017, 2019); lp = rp.idxmax(1); krp = set(lp.index[lp == "KR"])
    rt = counts("prov", pk, 2017, 2019); ld = rt.idxmax(1); krl = set(ld.index[ld == "KR"])
    rhos = {c: rho(rt[c].values, rp[c].values) for c in CCS}
    rr["land"] = dict(gt1=int((rt.KR > 1).sum()), lead=len(krl), kept=len(krl & krp), rho_KR=rhos["KR"], min_rho=min(rhos.values()),
                      leaders={c: int((ld == c).sum()) for c in CCS}, big_change=bool(len(krl & krp) < 4 or min(rhos.values()) < 0.7))
    res[arm] = rr; log(arm, round(rr["rho_alt"], 5), rr["ci_alt"], round(rr["d"], 5), rr["ci_d"], rr["verdict"], rr["land"]["gt1"], rr["land"]["lead"], rr["land"]["kept"], round(rr["land"]["rho_KR"], 5), rr["land"]["big_change"])
OUT["results"] = res
json.dump(OUT, open(V + "/v3_rho.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
log("done")
