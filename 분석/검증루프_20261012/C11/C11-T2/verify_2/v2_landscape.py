# -*- coding: utf-8 -*-
r"""C11-T2 verify_2: 168행 한국 RTA rho 범위 정정판('0.19-0.37') 반박 시도.
duckdb·작업자 s7·verify_1·C10 s3/v5/alt_fix 코드를 쓰지 않는다. pyarrow 필터 + pandas groupby 로 해외 FREE/TXT 라벨을 직접 재배정.
읽기만: c5_R10/alt_assign, rec_conc, TXT_R9/txt_assign, T1 kr_txtfree, conc_ksic_map, C10 links.parquet(REG/PUB), _ideas_R8 fam_link, C10 alt_fix(대조만).
쓰기: verify_2/v2_out.json"""
import sys, json, time, glob
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq, pyarrow.dataset as ds
B = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석"
MC = B + "/모델경쟁_20260927"; T3 = MC + "/T3_기술셀부상예측"; C10 = B + "/검증루프_20261012/C10/C10-T3"
OUT = B + "/검증루프_20261012/C11/C11-T2/verify_2/v2_out.json"
CCS = ["KR", "US", "JP", "CN", "EP"]; FOR = CCS[1:]
t0 = time.time(); log = lambda *a: print(*a, f"[{time.time()-t0:.0f}s]", flush=True)
km = pd.read_csv(T3 + "/ESSENTIAL_R7/conc_ksic_map.csv", dtype=str)[["ksic", "code"]]
K50 = sorted(km.ksic); nk = km.groupby("code").ksic.size(); MULTI = set(nk[nk > 1].index)
ALT = ds.dataset(glob.glob(MC + "/_shared_report/fusion_advantage/c5_R10/alt_assign/*.parquet"))
cols = ["cc", "appl_no", "appl_year", "is_inv", "ksic_prov", "code", "k_txt", "m_txt", "k_free", "m_free"]
D = ALT.to_table(columns=cols, filter=(ds.field("is_inv") == True) & (ds.field("appl_year") >= 2017) & (ds.field("appl_year") <= 2019)).to_pandas()
log("창", D.groupby("cc").size().to_dict())
assert not D.duplicated(["cc", "appl_no"]).any()
RC = ds.dataset(T3 + "/ESSENTIAL_R7/rec_conc.parquet")
kfree = pq.read_table(T3.replace("T3_기술셀부상예측", "T1_산업대기실") + "/TXT_R9/cache/kr_txtfree.parquet").to_pandas().dropna(subset=["k_free"])
kfree = kfree.drop_duplicates("appl_no").set_index("appl_no").k_free


def mode_pick(df, keys, lab="k"):
    """keys 별 lab 최빈, 동률은 정수값 작은 것."""
    g = df.groupby(keys + [lab]).size().rename("n").reset_index()
    g["ki"] = g[lab].astype(int)
    g = g.sort_values(keys + ["n", "ki"], ascending=[True] * len(keys) + [False, True])
    return g.drop_duplicates(keys)[keys + [lab]]


def lv(sg):
    return sg, sg.str.split("/", n=1).str[0], sg.str.slice(0, 4)


# ---- sg for KR (all) and foreign window docs ----
rcK = RC.to_table(columns=["cc", "appl_no", "sg"], filter=ds.field("cc") == "KR").to_pandas()
assert not rcK.duplicated("appl_no").any()
sgF = []
for cc in FOR:
    keys = pa.array(D.loc[D.cc == cc, "appl_no"].values)
    r = RC.to_table(columns=["appl_no", "sg"], filter=(ds.field("cc") == cc) & pc.is_in(ds.field("appl_no"), value_set=keys)).to_pandas(); r["cc"] = cc; sgF.append(r)
sgF = pd.concat(sgF); assert not sgF.duplicated(["cc", "appl_no"]).any()
F = D[D.cc != "KR"].merge(sgF, on=["cc", "appl_no"], how="left")
log("해외 sg 결측", float(F.sg.isna().mean()))
F["sg_"], F["mg_"], F["sc_"] = lv(F.sg.fillna(""))
# ---- FREE 표 (KR 출원<=2019, 코드 무관) ----
KRall = ALT.to_table(columns=["appl_no", "appl_year"], filter=ds.field("cc") == "KR").to_pandas()
KRt = KRall[KRall.appl_year <= 2019].merge(rcK[["appl_no", "sg"]], on="appl_no", how="left")
KRt["k"] = KRt.appl_no.map(kfree); KRt = KRt.dropna(subset=["k"])
KRt["sg_"], KRt["mg_"], KRt["sc_"] = lv(KRt.sg.fillna(""))
free_tab = pd.Series(np.nan, index=F.index, dtype=object)
for L in ("sg_", "mg_", "sc_"):
    s = KRt[KRt[L] != ""]; tot = s.groupby(L).size(); s = s[s[L].isin(tot[tot >= 3].index)]
    m = mode_pick(s, [L]).set_index(L).k
    free_tab = free_tab.fillna(F[L].map(m).where(F[L] != ""))
F["free_tab"] = free_tab
nf = F.m_free.str.startswith("ipc_")
R = {"free_tab_eq_stored_nonfamily": float((F.loc[nf, "free_tab"] == F.loc[nf, "k_free"]).mean()),
     "free_tab_none_eq": float(F.loc[F.m_free == "none", "free_tab"].isna().mean())}
log("FREE 표", R)
# ---- TXT 표 (KR txt_assign method='text', 코드 안) ----
TA = ds.dataset(T3 + "/TXT_R9/txt_assign.parquet")
taK = TA.to_table(columns=["appl_no", "code", "ksic_txt", "method"], filter=ds.field("cc") == "KR").to_pandas()
kt = taK[taK.method == "text"].merge(rcK[["appl_no", "sg"]], on="appl_no", how="inner").rename(columns={"ksic_txt": "k"})
kt["sg_"], kt["mg_"], kt["sc_"] = lv(kt.sg.fillna(""))
taF = []
for cc in FOR:
    keys = pa.array(D.loc[D.cc == cc, "appl_no"].values)
    r = TA.to_table(columns=["appl_no", "code", "ksic_txt", "method"], filter=(ds.field("cc") == cc) & pc.is_in(ds.field("appl_no"), value_set=keys)).to_pandas(); r["cc"] = cc; taF.append(r)
taF = pd.concat(taF); assert not taF.duplicated(["cc", "appl_no"]).any()
F = F.merge(taF.rename(columns={"code": "code_ta", "ksic_txt": "k_ta", "method": "m_ta"}), on=["cc", "appl_no"], how="left")
F["inpool"] = F.k_ta.notna(); F["multi"] = F.code_ta.isin(MULTI) & F.inpool
R["pool_share"] = float(F.inpool.mean()); R["pool_code_eq_alt_code"] = float((F.loc[F.inpool, "code_ta"] == F.loc[F.inpool, "code"]).mean())
R["pool_kta_eq_stored_ktxt"] = float((F.loc[F.inpool, "k_ta"] == F.loc[F.inpool, "k_txt"]).mean())
txt_tab = pd.Series(np.nan, index=F.index, dtype=object)
for L in ("sg_", "mg_", "sc_"):
    s = kt[kt[L] != ""]; tot = s.groupby(["code", L]).size(); ok = tot[tot >= 3].reset_index()[["code", L]]
    s = s.merge(ok, on=["code", L]); m = mode_pick(s, ["code", L]).rename(columns={"code": "code_ta", "k": "kk"})
    v = F[["code_ta", L]].merge(m, on=["code_ta", L], how="left").kk.values
    txt_tab = txt_tab.fillna(pd.Series(v, index=F.index).where(F[L] != ""))
mc = mode_pick(kt, ["code"]).set_index("code").k
txt_tab = txt_tab.fillna(F.code_ta.map(mc))
F["txt_tab"] = txt_tab
nfm = F.multi & (F.m_ta != "family")
R["txt_tab_eq_stored_nonfamily"] = float((F.loc[nfm, "txt_tab"] == F.loc[nfm, "k_ta"]).mean()); R["n_multi_nonfam"] = int(nfm.sum())
log("TXT 표", R["txt_tab_eq_stored_nonfamily"])
krtxt = taK[["appl_no", "code", "ksic_txt"]].rename(columns={"appl_no": "kr_appl_no", "code": "code_ta", "ksic_txt": "k"})


def get_links(arm):
    if arm == "r8":
        L = pq.read_table(MC + "/_ideas_R8/cache/fam_link.parquet").to_pandas()
    else:
        L = ds.dataset(C10 + "/cache/links.parquet").to_table(columns=["cc", "appl_no", "kr_appl_no"], filter=ds.field("link") == arm.upper()).to_pandas()
    return L[L.cc != "KR"].drop_duplicates()


FREE, TXT, NLINK = {}, {}, {}
for arm in ("r8", "reg", "pub", "pure"):
    if arm == "pure":
        ff = F.free_tab.values; tt = np.where(F.multi, F.txt_tab, F.k_txt)
    else:
        L = get_links(arm)
        L = L.merge(F[["cc", "appl_no"]], on=["cc", "appl_no"])   # 창 해외 문서만
        Lf = L.assign(k=L.kr_appl_no.map(kfree)).dropna(subset=["k"])
        fm = mode_pick(Lf, ["cc", "appl_no"]).rename(columns={"k": "kf"})
        ff = F[["cc", "appl_no"]].merge(fm, on=["cc", "appl_no"], how="left").kf
        ff = ff.fillna(F.free_tab).values
        Lt = L.merge(F.loc[F.multi, ["cc", "appl_no", "code_ta"]], on=["cc", "appl_no"]).merge(krtxt, on=["kr_appl_no", "code_ta"])
        tm = mode_pick(Lt, ["cc", "appl_no"]).rename(columns={"k": "kt"})
        tfam = F[["cc", "appl_no"]].merge(tm, on=["cc", "appl_no"], how="left").kt
        tt = np.where(F.multi, tfam.fillna(F.txt_tab), F.k_txt)   # 풀 밖·단일코드 = 저장 k_txt
        NLINK[arm] = dict(win_link_docs=int(L[["cc", "appl_no"]].drop_duplicates().shape[0]), free_fam=int(fm.shape[0]), txt_fam=int(tm.shape[0]))
    FREE[arm] = pd.Series(ff, index=F.index); TXT[arm] = pd.Series(tt, index=F.index)
R["n_link"] = NLINK
R["free_r8_eq_stored"] = float((FREE["r8"].fillna("NA") == F.k_free.fillna("NA")).mean())
R["txt_r8_eq_stored"] = float((TXT["r8"].fillna("NA") == F.k_txt.fillna("NA")).mean())
log("r8 재현", R["free_r8_eq_stored"], R["txt_r8_eq_stored"])
# 대조: C10 alt_fix (값만 비교)
AF = ds.dataset(glob.glob(C10 + "/cache/alt_fix_*.parquet"))
af = AF.to_table(columns=["cc", "appl_no", "k_free_reg", "k_txt_reg", "k_free_pub", "k_txt_pub", "k_free_pure", "k_txt_pure"],
                 filter=(ds.field("cc") != "KR") & (ds.field("is_inv") == True) & (ds.field("appl_year") >= 2017) & (ds.field("appl_year") <= 2019)).to_pandas()
af = F[["cc", "appl_no"]].merge(af, on=["cc", "appl_no"], how="left")
R["vs_altfix"] = {f"{lab}_{arm}": float(((FREE if lab == "free" else TXT)[arm].fillna("NA").values == af[f"k_{lab}_{arm}"].fillna("NA").values).mean())
                  for lab in ("free", "txt") for arm in ("reg", "pub", "pure")}
log("alt_fix 대조", R["vs_altfix"])
# ---- RTA ----
KR = D[D.cc == "KR"]


def cnt(kr_lab, for_lab):
    n = {"KR": kr_lab.dropna().value_counts()}
    for cc in FOR:
        n[cc] = for_lab[(F.cc == cc).values].dropna().value_counts()
    return pd.DataFrame(n).reindex(K50).fillna(0.0)[CCS]


def rta(n):
    s = n / n.sum(0); return s.div(s.mean(1), axis=0)


def rho(a, b):
    return float(np.corrcoef(a.rank(method="average"), b.rank(method="average"))[0, 1])


P = rta(cnt(KR.ksic_prov, F.ksic_prov)); ldP = P.idxmax(1); krP = set(ldP[ldP == "KR"].index)
R["prov"] = dict(gt1=int((P.KR > 1).sum()), lead=len(krP))
wk = km.assign(w=1.0 / km.code.map(nk))
Dc = D.merge(wk, on="code")
ncw = Dc.groupby(["ksic", "cc"]).w.sum().unstack().reindex(K50).fillna(0.0)[CCS]
rows = {}


def add(name, rt):
    ld = rt.idxmax(1); kl = set(ld[ld == "KR"].index)
    rows[name] = dict(gt1=int((rt.KR > 1).sum()), lead=len(kl), kept=len(kl & krP), rho=rho(rt.KR, P.KR))


add("cweq_F", rta(ncw))
for arm in ("r8", "reg", "pub", "pure"):
    add(f"free_F_{arm}", rta(cnt(KR.k_free, FREE[arm])))
    add(f"txt_F_{arm}", rta(cnt(KR.k_txt, TXT[arm])))
    add(f"free_H_{arm}", rta(cnt(KR.ksic_prov, FREE[arm])))
    add(f"txt_H_{arm}", rta(cnt(KR.ksic_prov, TXT[arm])))
R["rows"] = rows
rng = {}
for arm in ("r8", "reg", "pub", "pure"):
    v = [rows["cweq_F"]["rho"], rows[f"txt_F_{arm}"]["rho"], rows[f"free_F_{arm}"]["rho"]]
    ld = [rows["cweq_F"]["lead"], rows[f"txt_F_{arm}"]["lead"], rows[f"free_F_{arm}"]["lead"]]
    rng[arm] = dict(min=min(v), max=max(v), text=f"{min(v):.2f}–{max(v):.2f}", lead=f"{min(ld)}–{max(ld)}")
R["range_168"] = rng
for k, v in rows.items(): print(k, v)
print(json.dumps(rng, ensure_ascii=True, indent=1))
json.dump(R, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
log("end")
