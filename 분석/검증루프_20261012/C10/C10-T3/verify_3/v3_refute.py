# -*- coding: utf-8 -*-
r"""C10-T3 verify_3: E-C10T3-05 (FREE-IPC ρ 0.571→0.608, Δ −0.026→−0.064, TXT 0.444 Δ +0.101) 반박 시도.
작업자(scripts/s1–s3)·verify_0(v1/v2)·fam_reg.parquet·links.parquet·alt_fix를 쓰지 않는다.
- 연결: _shared/meta 해외 family 원문 토큰을 직접 분해해 (a) R8 방식(KR+13자리를 출원번호로) (b) 13자리를 KR 공개번호로 (c) 등록번호 9자리 를 만들고 KSIC 일치로 결함 확인.
- FREE: c1 규칙(연결 KR 문서 k_free 최빈→IPC 표 sg→mg→sc, 최소 3건, 동률 작은 코드)을 pandas로 재구현. 표는 KR 출원≤2019 문서로 새로 만든다.
- TXT: s01 규칙(풀 안 복수코드 해외 문서 = 연결 KR 풀 문서 같은 코드 ksic_txt 최빈 → (code,sg/mg/sc)→code 표)을 재구현.
- ρ: c2 paired()와 같은 rng(seed 20261005) 호출 순서 cweq→txt→free.
쓰기: verify_3/v3_out.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.dataset as ds
from scipy.stats import spearmanr
V = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(V, *[".."] * 5))
MC = os.path.join(BASE, "분석", "모델경쟁_20260927")
t0 = time.time(); log = lambda *a: print(*a, f"[{time.time()-t0:.0f}s]", flush=True)
OUT = {}
meta = ds.dataset(os.path.join(MC, "_shared", "meta"), format="parquet")
def obj(df):
    return df.astype({c: object for c in df.columns if str(df[c].dtype) == "category"})
STAGE = sys.argv[1] if len(sys.argv) > 1 else "1"
CA = os.path.join(V, "cache_A.parquet"); CL = os.path.join(V, "cache_LREG.parquet")
if STAGE == "2":
    OUT = json.load(open(os.path.join(V, "v3_out.json"), encoding="utf-8"))
    A = pd.read_parquet(CA); LREG = pd.read_parquet(CL)
    r8 = pq.read_table(os.path.join(MC, "_ideas_R8", "cache", "fam_link.parquet")).to_pandas()
    rc = obj(pq.read_table(os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "rec_conc.parquet"), columns=["cc", "appl_no", "sg"]).to_pandas())
    FREE = {k: A["free_" + k].where(A["free_" + k].notna(), None) for k in ("r8", "reg", "pub", "pure")}
    def mode_by(df, keys, val, min_n=1):
        g = df.groupby(keys + [val]).size().rename("n").reset_index()
        if min_n > 1:
            g = g[g.groupby(keys).n.transform("sum") >= min_n]
        g["kk"] = g[val].astype(int)
        return g.sort_values(keys + ["n", "kk"], ascending=[True] * len(keys) + [False, True]).drop_duplicates(keys).set_index(keys)[val]
if STAGE == "1":
    parts = []
    for b in meta.to_batches(columns=["cc", "appl_no", "family"], filter=(ds.field("cc") != "KR"), batch_size=500000):
        fm = obj(b.to_pandas()); fm = fm[fm.family.notna() & fm.family.str.contains("KR", regex=False)]
        tk = fm.assign(t=fm.family.str.split("|")).explode("t")[["cc", "appl_no", "t"]]
        parts.append(tk[tk.t.str.startswith("KR", na=False)])
    tok = pd.concat(parts).drop_duplicates(); del parts, fm, tk
    p13 = tok[tok.t.str.fullmatch(r"KR(10|20)\d{11}")].assign(n=lambda d: d.t.str[2:])[["cc", "appl_no", "n"]]
    r9 = tok[tok.t.str.fullmatch(r"KR(10|20)\d{7}")].assign(n=lambda d: d.t.str[2:])[["cc", "appl_no", "n"]]
    log("tokens p13", len(p13), "r9", len(r9))
    kr = obj(meta.to_table(columns=["appl_no", "rgst_no", "ksic"], filter=ds.field("cc") == "KR").to_pandas())
    kpn = pq.read_table(os.path.join(BASE, "공모전데이터", "작업데이터", "kr_patents.parquet")).select([0, 5]).to_pandas()
    kpn.columns = ["appl_no", "pubno"]; kpn["pubno"] = kpn.pubno.str.strip()
    kpn = kpn[kpn.pubno.str.fullmatch(r"\d{13}", na=False)]
    fk = obj(meta.to_table(columns=["cc", "appl_no", "ksic"], filter=(ds.field("cc") != "KR")).to_pandas()).rename(columns={"ksic": "fk"})
    kks = kr[["appl_no", "ksic"]].rename(columns={"appl_no": "kr_appl_no", "ksic": "kk"})
    def qual(L):
        x = L.merge(fk, on=["cc", "appl_no"]).merge(kks, on="kr_appl_no")
        return {cc: [int(len(g)), round(float((g.fk == g.kk).mean()) * 100, 2)] for cc, g in x.groupby("cc")}
    L_appl = p13.merge(kr[["appl_no"]].rename(columns={"appl_no": "n"}), on="n").rename(columns={"n": "kr_appl_no"}).drop_duplicates()
    L_pub = p13.merge(kpn.rename(columns={"pubno": "n", "appl_no": "kr_appl_no"}), on="n")[["cc", "appl_no", "kr_appl_no"]].drop_duplicates()
    krr = kr[kr.rgst_no.notna() & (kr.rgst_no.str.strip().str.len() >= 9)].assign(n=lambda d: d.rgst_no.str[:9])[["appl_no", "n"]].rename(columns={"appl_no": "kr_appl_no"})
    L_reg9 = r9.merge(krr, on="n")[["cc", "appl_no", "kr_appl_no"]].drop_duplicates()
    OUT["link_quality_pairs_n_agree%"] = {"p13_as_appl(R8)": qual(L_appl), "p13_as_pub": qual(L_pub), "reg9": qual(L_reg9)}
    log(OUT["link_quality_pairs_n_agree%"])
    r8 = pq.read_table(os.path.join(MC, "_ideas_R8", "cache", "fam_link.parquet")).to_pandas()
    key = lambda d: set(map(tuple, d[["cc", "appl_no", "kr_appl_no"]].values))
    s_r8, s_ap = key(r8), key(L_appl)
    OUT["r8_equals_p13_as_appl"] = [len(s_r8), len(s_ap), len(s_r8 & s_ap)]
    s_reg_file = key(pq.read_table(os.path.join(MC, "_shared_report", "label_origin_R10", "cache", "fam_reg.parquet"), columns=["cc", "appl_no", "kr_appl_no"]).to_pandas())
    s_reg = key(L_reg9)
    OUT["my_reg9_vs_fam_reg_file"] = [len(s_reg), len(s_reg_file), len(s_reg & s_reg_file)]
    LREG = pd.concat([L_reg9, L_appl[L_appl.cc == "CN"]]).drop_duplicates()
    log(OUT["r8_equals_p13_as_appl"], OUT["my_reg9_vs_fam_reg_file"])
    del tok, p13, r9, fk, kpn
    # ---------------- B. FREE 라벨 ----------------
    rc = pq.read_table(os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "rec_conc.parquet"), columns=["cc", "appl_no", "sg"]).to_pandas()
    rc = obj(rc)
    A = obj(ds.dataset(os.path.join(MC, "_shared_report", "fusion_advantage", "c5_R10", "alt_assign"), format="parquet").to_table(
        columns=["cc", "appl_no", "appl_year", "is_inv", "kr_origin", "ksic_prov", "code", "k_in_code", "k_txt", "m_txt", "k_free", "m_free"],
        filter=(ds.field("cc") != "KR") & ds.field("is_inv") & (ds.field("appl_year") >= 2015) & (ds.field("appl_year") <= 2019)).to_pandas())
    A = A.merge(rc, on=["cc", "appl_no"], how="left")
    A["mg"] = A.sg.str.split("/").str[0]; A["sc"] = A.sg.str[:4]
    kf = pq.read_table(os.path.join(MC, "T1_산업대기실", "TXT_R9", "cache", "kr_txtfree.parquet")).to_pandas()[["appl_no", "k_free"]].dropna()
    kry = obj(meta.to_table(columns=["appl_no", "appl_year"], filter=ds.field("cc") == "KR").to_pandas())
    T = kry[kry.appl_year <= 2019].merge(rc[rc.cc == "KR"][["appl_no", "sg"]], on="appl_no").merge(kf, on="appl_no")
    T["mg"] = T.sg.str.split("/").str[0]; T["sc"] = T.sg.str[:4]
    def mode_by(df, keys, val, min_n=1):
        g = df.groupby(keys + [val]).size().rename("n").reset_index()
        if min_n > 1:
            g = g[g.groupby(keys).n.transform("sum") >= min_n]
        g["kk"] = g[val].astype(int)
        return g.sort_values(keys + ["n", "kk"], ascending=[True] * len(keys) + [False, True]).drop_duplicates(keys).set_index(keys)[val]
    tab = pd.Series(index=A.index, dtype=object)
    for lvl in ("sg", "mg", "sc"):
        src = T[T[lvl].notna() & (T[lvl] != "")]
        tab = tab.fillna(A[lvl].map(mode_by(src, [lvl], "k_free", 3)))
    nonfam = A.m_free.isin(["ipc_sg", "ipc_mg", "ipc_sc"])
    OUT["free_table_rebuild_eq_stored_nonfamily"] = round(float((tab[nonfam] == A.k_free[nonfam]).mean()), 6)
    def fam_mode(L, src, val):
        x = L.merge(src.rename(columns={"appl_no": "kr_appl_no"}), on="kr_appl_no")
        m = mode_by(x, ["cc", "appl_no"], val)
        return pd.Series(pd.MultiIndex.from_frame(A[["cc", "appl_no"]]).map(m), index=A.index)
    f_r8 = fam_mode(r8, kf, "k_free"); f_reg = fam_mode(LREG, kf, "k_free"); f_pub = fam_mode(pd.concat([L_pub[L_pub.cc != "CN"], L_appl[L_appl.cc == "CN"]]).drop_duplicates(), kf, "k_free")
    FREE = {"r8": f_r8.fillna(tab), "reg": f_reg.fillna(tab), "pub": f_pub.fillna(tab), "pure": tab}
    OUT["free_r8_eq_stored"] = round(float((FREE["r8"].fillna("NA") == A.k_free.fillna("NA")).mean()), 6)
    num = A.kr_origin & A.appl_year.between(2017, 2019)
    OUT["numerator_n_stored_free_labeled"] = int((num & A.k_free.notna()).sum())
    OUT["numerator_family_share_stored"] = [int((num & (A.m_free == "family")).sum()), round(float((A.m_free == "family")[num & A.k_free.notna()].mean()) * 100, 2)]
    OUT["numerator_family_share_reg"] = round(float(f_reg.notna()[num & FREE["reg"].notna()].mean()) * 100, 2)
    OUT["numerator_family_share_pub"] = round(float(f_pub.notna()[num & FREE["pub"].notna()].mean()) * 100, 2)
    OUT["recall_free"] = {k: {cc: round(float((v[A.cc == cc] == A.ksic_prov[A.cc == cc]).mean()) * 100, 3) for cc in ("US", "JP", "CN", "EP")} for k, v in FREE.items()}
    log({k: v for k, v in OUT.items() if not k.startswith("link")})
    for k, vv in FREE.items(): A["free_" + k] = vv
    A.to_parquet(CA, index=False); LREG.to_parquet(CL, index=False)
    json.dump(OUT, open(os.path.join(V, "v3_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    sys.exit(0)

# ---------------- C. TXT 라벨 ----------------
km = pd.read_csv(os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "conc_ksic_map.csv"), dtype=str)[["ksic", "code"]]
multi = set(km.groupby("code").ksic.size().loc[lambda s: s > 1].index)
ta = pq.read_table(os.path.join(MC, "T3_기술셀부상예측", "TXT_R9", "txt_assign.parquet"), columns=["cc", "appl_no", "code", "ksic_txt", "method"]).to_pandas()
ta = obj(ta)
krt = ta[(ta.cc == "KR") & (ta.method == "text")].merge(rc[rc.cc == "KR"][["appl_no", "sg"]], on="appl_no")
krt["mg"] = krt.sg.str.split("/").str[0]; krt["sc"] = krt.sg.str[:4]
X = A.merge(ta.rename(columns={"code": "tcode", "ksic_txt": "t3k", "method": "t3m"}), on=["cc", "appl_no"], how="left")
assert len(X) == len(A)
inpool_multi = X.t3m.notna() & X.tcode.isin(multi)
ttab = pd.Series(index=X.index, dtype=object)
for lvl in ("sg", "mg", "sc"):
    src = krt[krt[lvl].notna() & (krt[lvl] != "")]
    m = mode_by(src, ["code", lvl], "ksic_txt", 3)
    ttab = ttab.fillna(pd.Series(pd.MultiIndex.from_frame(X[["tcode", lvl]].fillna("")).map(m), index=X.index))
ttab = ttab.fillna(X.tcode.map(mode_by(krt, ["code"], "ksic_txt")))
tabm = inpool_multi & X.t3m.isin(["ipc_sg", "ipc_mg", "ipc_sc", "code_mode"])
OUT["txt_table_rebuild_eq_stored"] = round(float((ttab[tabm] == X.t3k[tabm]).mean()), 6)
krpool = ta[ta.cc == "KR"][["appl_no", "code", "ksic_txt"]].rename(columns={"appl_no": "kr_appl_no", "code": "kcode"})
def txt_fam(L):
    x = L.merge(X.loc[inpool_multi, ["cc", "appl_no", "tcode"]], on=["cc", "appl_no"]).merge(krpool, on="kr_appl_no")
    x = x[x.kcode == x.tcode]
    m = mode_by(x, ["cc", "appl_no"], "ksic_txt")
    return pd.Series(pd.MultiIndex.from_frame(X[["cc", "appl_no"]]).map(m), index=X.index)
def txt_lab(L):
    f = txt_fam(L)
    lab = X.k_txt.copy()                      # 풀 밖·단일 코드: 저장값(패밀리 경로 없음)
    lab[inpool_multi] = f[inpool_multi].fillna(ttab[inpool_multi])
    return lab
TXT = {"r8": txt_lab(r8), "reg": txt_lab(LREG)}
OUT["txt_r8_eq_stored"] = round(float((TXT["r8"].fillna("NA") == X.k_txt.fillna("NA")).mean()), 6)
OUT["recall_txt"] = {k: {cc: round(float((v[X.cc == cc] == X.ksic_prov[X.cc == cc]).mean()) * 100, 3) for cc in ("US", "JP", "CN", "EP")} for k, v in TXT.items()}
log(OUT["txt_table_rebuild_eq_stored"], OUT["txt_r8_eq_stored"], OUT["recall_txt"])
# ---------------- D. ρ (c2 paired 재현) ----------------
K50 = sorted(km.ksic)
ek = pd.read_csv(os.path.join(MC, "_shared_report", "fusion_advantage", "map5", "m4_exports_ksic_5rep.csv"), dtype={"k5": str})
v = ek[ek.year.isin([2015, 2016, 2017])].groupby(["k5", "rep"]).v.sum().unstack(); s_ = v / v.sum(); RCA = s_.div(s_.mean(1), axis=0).KR.reindex(K50)
P3 = A.appl_year.between(2017, 2019)
def abroad(lab, w=None):
    w = pd.Series(1.0, index=A.index) if w is None else w
    d = pd.DataFrame({"k": lab, "o": A.kr_origin, "w": w})[P3 & lab.notna()]
    k = d[d.o].groupby("k").w.sum().reindex(K50).fillna(0); a = d.groupby("k").w.sum().reindex(K50)
    return (k / a) / (k.sum() / a.sum())
# cweq: 코드 안 KSIC 동일가중
kmw = km.assign(w=1 / km.groupby("code").ksic.transform("size"))
cw = A[["code", "kr_origin", "appl_year"]].reset_index().merge(kmw, on="code")
cw = cw[cw.appl_year.between(2017, 2019)]
kc = cw[cw.kr_origin].groupby("ksic").w.sum().reindex(K50).fillna(0); ac = cw.groupby("ksic").w.sum().reindex(K50)
ab_cweq = (kc / ac) / (kc.sum() / ac.sum())
ab_prov = abroad(A.ksic_prov)
sp = lambda x, y: spearmanr(x, y).statistic
def paired(rng, xp, xa, y, B=2000):
    d = pd.concat([xp, xa, y], axis=1).dropna(); a, b, c = d.iloc[:, 0].values, d.iloc[:, 1].values, d.iloc[:, 2].values
    rp, ra = sp(a, c), sp(b, c); n = len(d); I = rng.integers(0, n, (B, n))
    bp = np.array([sp(a[i], c[i]) for i in I]); ba = np.array([sp(b[i], c[i]) for i in I])
    ci = lambda z: [round(float(np.nanpercentile(z, 2.5)), 3), round(float(np.nanpercentile(z, 97.5)), 3)]
    return dict(n=n, rho_prov=round(rp, 4), rho_alt=round(ra, 4), ci_alt=ci(ba), d=round(rp - ra, 4), ci_d=ci(bp - ba))
RHO = {}
for arm in ("r8", "reg"):
    rng = np.random.default_rng(20261005)
    r_cw = paired(rng, ab_prov, ab_cweq, RCA)
    r_txt = paired(rng, ab_prov, abroad(TXT[arm]), RCA)
    r_free = paired(rng, ab_prov, abroad(FREE[arm]), RCA)
    RHO[arm] = dict(cweq=r_cw, txt=r_txt, free=r_free)
for arm in ("pub", "pure"):
    rng = np.random.default_rng(20261005)
    paired(rng, ab_prov, ab_cweq, RCA); paired(rng, ab_prov, abroad(TXT["reg"]), RCA)
    RHO[arm] = dict(free=paired(rng, ab_prov, abroad(FREE[arm]), RCA))
OUT["rho"] = RHO
log(json.dumps(RHO, ensure_ascii=False))
json.dump(OUT, open(os.path.join(V, "v3_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log("done")
