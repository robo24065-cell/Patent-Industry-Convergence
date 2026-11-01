# -*- coding: utf-8 -*-
r"""[3] 강한 공개 대조 팔 피처 — 원본 TXT_R9 b_feat.py 복사·수정.
라벨: TXT-FREE(kr_txtfree) 라벨의 KR 메인 IPC 최빈표(R12 r12_prep.py hard() 규칙 그대로). 원점 t마다 표 = alt_KR KR 문서 중 출원연도 ≤ t.
  FIPC = H-sg-3, HSC1 = H-sc-1, HMG3 = H-mg-3. 같은 빌더로 FREE50도 다시 만들어 X_o의 FREE50과 대조(빌더 검증).
행(법인×서브클래스)은 NONCIRC_R8 cache/A_o<t>.parquet 키 그대로(읽기만).
사용: python -B z_feat.py 2002 2003 ... → cache/Z_o<t>.parquet, res/zfeat/z_o<t>.json"""
import sys
sys.dont_write_bytecode = True
from c10_common import *
import scipy.sparse as sp
import pyarrow.parquet as pq, pyarrow.compute as pc

KSIC50 = ['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924',
          '25932', '26111', '26112', '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323',
          '26422', '26429', '26519', '26529', '27199', '27211', '27212', '27301', '28111', '28122', '28511', '28902',
          '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280', '30393', '31202', '31312',
          '33301', '33309']
KI = {k: i for i, k in enumerate(KSIC50)}
os.makedirs(os.path.join(RES, "zfeat"), exist_ok=True)


def load_docs():
    """alt_KR KR 문서 + rec_conc KR 메인 IPC sg + TXT-FREE. 반환 DataFrame(appl_no, ay, kf(int, -1 결측), sg, mg, sc)."""
    a = pd.read_parquet(ALT_KR, columns=["appl_no", "appl_year", "k_free", "m_free", "ksic_prov"])
    r = pq.read_table(REC, columns=["cc", "appl_no", "sg"], filters=[("cc", "=", "KR")]).to_pandas()
    r = r.drop(columns="cc").drop_duplicates("appl_no")
    f = pd.read_parquet(os.path.join(SRC_CACHE, "kr_txtfree.parquet"))
    d = a.merge(r, on="appl_no", how="left").merge(f.rename(columns={"k_free": "k_free_t1"}), on="appl_no", how="left")
    d["sg"] = d.sg.where(d.sg.notna() & (d.sg != ""))
    d["mg"] = d.sg.str.split("/").str[0]; d["sc4"] = d.sg.str[:4]
    d["kf"] = d.k_free_t1.map(KI).fillna(-1).astype(int)
    return d


def tables(d, t):
    """출원연도 ≤ t 문서로 sg/mg/sc 키 × 50 라벨 건수표."""
    m0 = (d.appl_year.values <= t) & (d.kf.values >= 0)
    out = {}
    for lvl in ("sg", "mg", "sc4"):
        g = d.loc[m0 & d[lvl].notna().values, [lvl, "kf"]]
        ct = g.groupby([lvl, "kf"]).size().unstack(fill_value=0).reindex(columns=range(50), fill_value=0)
        out[lvl] = ct
    return out


def hard(T, keys, start, nmin):
    """R12 hard(): start 수준에서 키 건수 ≥ nmin이면 최빈(동률 작은 코드 = argmax 첫 번째), 아니면 상위 수준, 끝까지 없으면 -1."""
    lv = ["sg", "mg", "sc4"][["sg", "mg", "sc4"].index(start):]
    out = np.full(len(keys), -1, dtype=np.int16); todo = np.ones(len(keys), bool)
    for l in lv:
        M = T[l].reindex(keys[l].values).values  # NaN for missing key
        M = np.nan_to_num(M, nan=0.0)
        n = M.sum(1); ok = todo & (n >= nmin)
        out[ok] = M[ok].argmax(1); todo &= ~ok
    return out


def rownorm(M):
    s = np.asarray(M.sum(1)).ravel(); s[s == 0] = 1
    return sp.diags(1 / s) @ M


def ctx_feats(V, col, t, fi, si, nF, nS, ff, ss, Hd, Hn):
    Vl = V[V[col].notna()]
    K = np.sort(Vl[col].unique()); ki = {k: i for i, k in enumerate(K)}
    M = rownorm(sp.csr_matrix((np.ones(len(Vl)), (Vl.applicantCd.map(fi).values, Vl[col].map(ki).values)), shape=(nF, len(K)))).toarray()
    Q = rownorm(sp.csr_matrix((np.ones(len(Vl)), (Vl.sc.map(si).values, Vl[col].map(ki).values)), shape=(nS, len(K)))).toarray()
    nm = np.linalg.norm(M, axis=1); nq = np.linalg.norm(Q, axis=1)
    dot = np.einsum("ij,ij->i", M[ff], Q[ss])
    F = {"dot": dot, "cos": dot / np.maximum(nm[ff] * nq[ss], 1e-12)}
    mden = M.sum(0); mden[mden == 0] = 1
    Aa = (M.T @ Hd) / mden[:, None]
    En = (M.T @ Hn) / mden[:, None]
    F["adopt"] = np.einsum("ij,ij->i", M[ff], Aa[:, ss].T)
    F["entry"] = np.einsum("ij,ij->i", M[ff], En[:, ss].T)
    main = M.argmax(1); hasm = M.sum(1) > 0
    F["qmain"] = np.where(hasm[ff], Q[ss, main[ff]], 0)
    g1 = Vl[Vl.ay.between(t - 2, t)][col].value_counts(); g0 = Vl[Vl.ay.between(t - 5, t - 3)][col].value_counts()
    kg = np.log1p(pd.Series(K).map(g1).fillna(0).values) - np.log1p(pd.Series(K).map(g0).fillna(0).values)
    F["mommain"] = np.where(hasm[ff], kg[main[ff]], 0)
    F["sindmom"] = Q[ss] @ kg
    return F, len(K)


def build(t, D0, P):
    T = tables(D0, t)
    # T1 풀 문서에 라벨 부여(문서 단위)
    keys = P[["sg", "mg", "sc4"]]
    lab = {"FIPC": hard(T, keys, "sg", 3), "HSC1": hard(T, keys, "sc4", 1), "HMG3": hard(T, keys, "mg", 3)}
    P = P.copy()
    for a, v in lab.items():
        P[a] = pd.Series(v, index=P.index).where(v >= 0).map(lambda i: KSIC50[int(i)] if pd.notna(i) else None)
    V = P[P.ky <= t]
    firms_all = np.sort(V.applicantCd.unique()); fi = {f: i for i, f in enumerate(firms_all)}
    SC = np.sort(P.sc.unique()); si = {s: i for i, s in enumerate(SC)}
    nF, nS = len(firms_all), len(SC)
    C = sp.csr_matrix((np.ones(len(V)), (V.applicantCd.map(fi).values, V.sc.map(si).values)), shape=(nF, nS))
    Hd = (C > 0).astype(np.float64).toarray()
    fy = V.groupby(["applicantCd", "sc"]).ay.min().reset_index()
    Hn = sp.csr_matrix(((fy.ay >= t - 3).astype(float).values, (fy.applicantCd.map(fi).values, fy.sc.map(si).values)), shape=(nF, nS)).toarray()
    A0 = pd.read_parquet(os.path.join(NC8, "cache", f"A_o{t}.parquet"), columns=["applicantCd", "sc"])
    X0 = pd.read_parquet(os.path.join(SRC_CACHE, f"X_o{t}.parquet"), columns=["applicantCd", "sc"] + cx("FREE50"))
    assert (A0.applicantCd.values == X0.applicantCd.values).all() and (A0.sc.values == X0.sc.values).all()
    ff = A0.applicantCd.map(fi).values.astype(int); ss = A0.sc.map(si).values.astype(int)
    D = A0[["applicantCd", "sc"]].copy()
    info = {"t": t, "rows": len(D), "vis_docs": int(len(V)), "nF": nF, "nS": nS}
    F, nk = ctx_feats(V, "k_free", t, fi, si, nF, nS, ff, ss, Hd, Hn)
    info["FREE50_rebuild_maxabs"] = {x: float(np.max(np.abs(F[x].astype(np.float32) - X0[f"FREE50_{x}"].values))) for x in CTX}
    for arm in ("FIPC", "HSC1", "HMG3"):
        F, nk = ctx_feats(V, arm, t, fi, si, nF, nS, ff, ss, Hd, Hn)
        for x in CTX:
            D[f"{arm}_{x}"] = F[x].astype(np.float32)
        info[arm] = {"n_labels": int(nk), "vis_assigned": float(V[arm].notna().mean()),
                     "vis_agree_prov": float((V[arm] == V.ksic).mean()), "vis_agree_free": float((V[arm] == V.k_free).mean())}
    info["vis_free_agree_prov"] = float((V.k_free == V.ksic).mean())
    D.to_parquet(os.path.join(CACHE, f"Z_o{t}.parquet"), index=False)
    json.dump(info, open(os.path.join(RES, "zfeat", f"z_o{t}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(t, info)


if __name__ == "__main__":
    D0 = load_docs()
    P = pd.read_parquet(os.path.join(NC8, "cache", "patents.parquet"))
    P = P.merge(pd.read_parquet(os.path.join(SRC_CACHE, "kr_txtfree.parquet")), on="appl_no", how="left")
    P = P.merge(D0[["appl_no", "sg", "mg", "sc4"]], on="appl_no", how="left")
    log("docs", len(D0), "pool rows", len(P), "pool sg matched", float(P.sg.notna().mean()))
    for t in map(int, sys.argv[1:]):
        build(t, D0, P)
