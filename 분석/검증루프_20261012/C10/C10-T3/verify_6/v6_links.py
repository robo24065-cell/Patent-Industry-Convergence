# -*- coding: utf-8 -*-
r"""C10-T3 verify_6: 연결 품질·편향·상속 정확도 독립 재계산(pandas/pyarrow, 작업자 cache·scripts 미사용). 쓰기: verify_6/v6_out.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.dataset as ds
V = os.path.dirname(os.path.abspath(__file__))
AN = os.path.abspath(os.path.join(V, "..", "..", "..", ".."))
MC = os.path.join(AN, "모델경쟁_20260927"); BASE = os.path.dirname(AN)
t = time.time()
def log(*a):
    print(*a, f"[{time.time()-t:.0f}s]", flush=True)
meta = ds.dataset(os.path.join(MC, "_shared", "meta"), format="parquet")
def obj(df):
    return df.astype({c: object for c in df.columns if str(df[c].dtype) in ("category", "string")})
KR = obj(meta.to_table(columns=["appl_no", "appl_year", "rgst_no", "ksic"], filter=ds.field("cc") == "KR").to_pandas())
KR["rg9"] = KR.rgst_no.fillna("").str[:9]
kp = obj(pq.read_table(os.path.join(BASE, "공모전데이터", "작업데이터", "kr_patents.parquet"), columns=["출원번호", "공개번호"]).to_pandas())
kp["pb"] = kp["공개번호"].fillna("").str.strip()
kp = kp[kp.pb != ""].rename(columns={"출원번호": "kr_appl_no"})[["kr_appl_no", "pb"]]
kf = obj(pq.read_table(os.path.join(MC, "T1_산업대기실", "TXT_R9", "cache", "kr_txtfree.parquet")).to_pandas()).dropna()
rc = obj(pq.read_table(os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "rec_conc.parquet"), columns=["cc", "appl_no", "sg"]).to_pandas())
assert not rc.duplicated(["cc", "appl_no"]).any()
reg_file = obj(pq.read_table(os.path.join(MC, "_shared_report", "label_origin_R10", "cache", "fam_reg.parquet"), columns=["cc", "appl_no", "kr_appl_no"]).to_pandas()).drop_duplicates()
r8 = obj(pq.read_table(os.path.join(MC, "_ideas_R8", "cache", "fam_link.parquet")).to_pandas()).drop_duplicates()
# IPC 표 (KR 출원<=2019 ∩ rec_conc ∩ kr_txtfree)
T = KR[KR.appl_year <= 2019][["appl_no"]].merge(rc[rc.cc == "KR"][["appl_no", "sg"]], on="appl_no").merge(kf, on="appl_no")
T["mg"] = T.sg.str.split("/").str[0]
T["sc"] = T.sg.str[:4]
def mode_tab(df, key):
    d = df[df[key].notna() & (df[key] != "")]
    g = d.groupby([key, "k_free"]).size().rename("n").reset_index()
    g = g[g.groupby(key).n.transform("sum") >= 3]
    g = g.assign(kk=g.k_free.astype(int)).sort_values([key, "n", "kk"], ascending=[True, False, True])
    return g.drop_duplicates(key).set_index(key).k_free
TABS = {l: mode_tab(T, l) for l in ("sg", "mg", "sc")}
log("IPC 표", {l: len(v) for l, v in TABS.items()})
kfmap = kf.set_index("appl_no").k_free
krk = KR.set_index("appl_no").ksic
rg2appl = KR[KR.rg9 != ""].drop_duplicates("rg9").set_index("rg9").appl_no
pbmap = kp.drop_duplicates("pb").set_index("pb").kr_appl_no
OUT = {"quality": {}, "overlap": {}, "bias": {}, "inherit": {}, "reg_rebuild": {}}
def fam_mode(link):
    x = link.assign(k=link.kr_appl_no.map(kfmap)).dropna(subset=["k"])
    g = x.groupby(["appl_no", "k"]).size().rename("n").reset_index()
    g = g.assign(kk=g.k.astype(int)).sort_values(["appl_no", "n", "kk"], ascending=[True, False, True])
    return g.drop_duplicates("appl_no").set_index("appl_no").k
num_parts = []
for cc in ("US", "JP", "EP", "CN"):
    cols = ["appl_no", "appl_year", "ksic", "prio_cc"] + ([] if cc == "CN" else ["family"])
    F = obj(meta.to_table(columns=cols, filter=ds.field("cc") == cc).to_pandas())
    if cc != "CN":
        fam = F[F.family.fillna("").str.contains("KR")][["appl_no", "family"]]
        tk = fam.assign(t=fam.family.str.split("|")).explode("t")[["appl_no", "t"]].drop_duplicates()
        tk = tk[tk.t.str.startswith("KR")].copy()
        p13 = tk.t.str.fullmatch(r"KR(10|20)\d{11}")
        p11 = tk.t.str.fullmatch(r"KR(19|20)\d{9}")
        r9 = tk.t.str.fullmatch(r"KR(10|20)\d{7}")
        tk["pub13"] = np.where(p13, tk.t.str[2:], np.where(p11, "10" + tk.t.str[2:], None))
        pub = tk[p13 | p11].copy()
        pub["kr_appl_no"] = pub.pub13.map(pbmap)
        pub = pub.dropna(subset=["kr_appl_no"])[["appl_no", "kr_appl_no"]].drop_duplicates()
        regi = tk[r9].copy()
        regi["kr_appl_no"] = regi.t.str[2:].map(rg2appl)
        regi = regi.dropna(subset=["kr_appl_no"])[["appl_no", "kr_appl_no"]].drop_duplicates()
        regf = reg_file[reg_file.cc == cc][["appl_no", "kr_appl_no"]]
        a = set(map(tuple, regi.values)); b = set(map(tuple, regf.values))
        OUT["reg_rebuild"][cc] = dict(indep=len(a), file=len(b), inter=len(a & b))
        reg = regf
        del fam, tk, a, b
    else:
        pub = reg = r8[r8.cc == "CN"][["appl_no", "kr_appl_no"]]
    fk = F.set_index("appl_no").ksic
    q = {}
    for nm, L in (("reg", reg), ("pub", pub)):
        q[nm] = dict(pairs=len(L), fdocs=int(L.appl_no.nunique()),
                     agree_pair=round(float((L.appl_no.map(fk).values == L.kr_appl_no.map(krk).values).mean()), 5))
    OUT["quality"][cc] = q
    if cc != "CN":
        both = set(reg.appl_no) & set(pub.appl_no)
        same = set(map(tuple, reg.values)) & set(map(tuple, pub.values))
        OUT["overlap"][cc] = dict(n_reg=int(reg.appl_no.nunique()), n_pub=int(pub.appl_no.nunique()), n_both=len(both),
                                  share_same=round(len({x[0] for x in same} & both) / len(both), 5))
    F["kr_origin"] = F.prio_cc.fillna("").str.split("|").str[0] == "KR"
    W = F[F.appl_year.between(2015, 2019)].copy()
    W["lr"] = W.appl_no.isin(set(reg.appl_no))
    W["lp"] = W.appl_no.isin(set(pub.appl_no))
    bb = dict(n=len(W))
    for c in ("lr", "lp"):
        bb[f"share_{c}"] = round(float(W[c].mean()), 5)
        bb[f"prioKR_linked_{c}"] = round(float(W.loc[W[c], "kr_origin"].mean()), 5) if W[c].any() else None
        bb[f"prioKR_unlinked_{c}"] = round(float(W.loc[~W[c], "kr_origin"].mean()), 5)
        if W[c].any():
            pa = W.loc[W[c], "ksic"].fillna("NA").value_counts(normalize=True)
            pb_ = W.loc[~W[c], "ksic"].fillna("NA").value_counts(normalize=True)
            bb[f"tvd_{c}"] = round(float(pa.sub(pb_, fill_value=0).abs().sum() / 2), 4)
    OUT["bias"][cc] = bb
    del W
    # 상속 정확도: 발명 ∩ rec_conc
    L_ = F.appl_no.str.len(); s5 = F.appl_no.str[4:5]
    if cc == "CN":
        inv = L_.isin([12, 14]) & s5.isin(["1", "8"])
    elif cc == "JP":
        inv = L_.isin([8, 17])
    else:
        inv = pd.Series(True, index=F.index)
    S = F[inv & F.appl_year.between(2015, 2019)][["appl_no", "appl_year", "ksic", "kr_origin"]].merge(rc[rc.cc == cc][["appl_no", "sg"]], on="appl_no")
    del F
    S["mg"] = S.sg.str.split("/").str[0]
    S["sc"] = S.sg.str[:4]
    ipc = pd.Series(None, index=S.index, dtype=object)
    for l in ("sg", "mg", "sc"):
        ipc = ipc.fillna(S[l].map(TABS[l]))
    S["ipc"] = ipc
    for nm, L in (("reg", reg), ("pub", pub)):
        S[f"inh_{nm}"] = S.appl_no.map(fam_mode(L))
    res = {}
    for win, m in (("2015-19_inv", pd.Series(True, index=S.index)), ("numer_KR_2017-19", S.kr_origin & S.appl_year.between(2017, 2019))):
        s = S[m]
        r = dict(n=len(s), acc_ipc_all=round(float((s.ipc == s.ksic).mean()), 5))
        for nm in ("reg", "pub"):
            h = s[f"inh_{nm}"].notna(); sh = s[h]
            r[nm] = dict(famshare=round(float(h.mean()), 5), n_fam=int(h.sum()),
                         acc_inh=round(float((sh[f"inh_{nm}"] == sh.ksic).mean()), 5) if h.any() else None,
                         acc_ipc_same_docs=round(float((sh.ipc == sh.ksic).mean()), 5) if h.any() else None,
                         acc_ipc_nonfam=round(float((s[~h].ipc == s[~h].ksic).mean()), 5),
                         recall_fixed=round(float((s[f"inh_{nm}"].fillna(s.ipc) == s.ksic).mean()), 5))
        res[win] = r
    num_parts.append(S[S.kr_origin & S.appl_year.between(2017, 2019)][["ksic", "ipc", "inh_reg", "inh_pub"]])
    OUT["inherit"][cc] = res
    del S
    log(cc, json.dumps({k: OUT[k].get(cc) for k in OUT}, ensure_ascii=False))
N = pd.concat(num_parts)
OUT["numer_pooled"] = {"n": len(N), "acc_ipc_all": round(float((N.ipc == N.ksic).mean()), 5)}
for nm in ("reg", "pub"):
    h = N[f"inh_{nm}"].notna()
    OUT["numer_pooled"][nm] = dict(famshare=round(float(h.mean()), 5),
                                   acc_inh=round(float((N[h][f"inh_{nm}"] == N[h].ksic).mean()), 5),
                                   acc_ipc_same_docs=round(float((N[h].ipc == N[h].ksic).mean()), 5))
json.dump(OUT, open(os.path.join(V, "v6_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("done", OUT["numer_pooled"])
