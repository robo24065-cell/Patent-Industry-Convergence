# [C19-T3 사본] 원본: C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared_report\label_origin_R10\lo_tests.py — 경로만 바꿈(쓰기→C19_OUT, META→C19_META, fam_reg→C19_OUT/cache)
# -*- coding: utf-8 -*-
r"""R10 label_origin — P1·P2·P3 (PREREG.txt). 읽기만, 쓰기는 이 폴더.
입력: cache/kr_lo.parquet, cache/fam_reg.parquet(등록번호 기반 패밀리, lo_fam.py), cache/pat_firm.parquet, cache/firm_reg.parquet
출력: res_p1.csv, res_p1_disagree.csv, res_p2.csv, res_p3.csv, res_tests.json"""
import os as _os
_C19_OUT = _os.environ['C19_OUT']; _C19_META = _os.environ['C19_META']
_os.makedirs(_os.path.join(_C19_OUT, 'cache'), exist_ok=True); _os.makedirs(_os.path.join(_C19_OUT, 'fp', 'cache', '_tmp'), exist_ok=True)
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
OUT = _C19_OUT; CACHE = os.path.join(MC, r"_shared_report\label_origin_R10", "cache")
_T = time.time()
def log(*a): print(*a, f"[{round(time.time()-_T)}s]", flush=True)
KSIC50 = ['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924',
          '25932', '26111', '26112', '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323',
          '26422', '26429', '26519', '26529', '27199', '27211', '27212', '27301', '28111', '28122', '28511', '28902',
          '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280', '30393', '31202', '31312',
          '33301', '33309']
K2I = {k: i for i, k in enumerate(KSIC50)}
rng = np.random.default_rng(20261005)
J = {}

kr = pd.read_parquet(os.path.join(CACHE, "kr_lo.parquet"), columns=["appl_no", "ay", "ksic", "sc", "mg", "sg", "k_free", "app"])
kr["appl_no"] = kr.appl_no.astype(str)
fam = pd.read_parquet(os.path.join(_C19_OUT, "cache", "fam_reg.parquet")); fam["kr_appl_no"] = fam.kr_appl_no.astype(str)
nk = fam.groupby(["cc", "appl_no"]).kr_appl_no.transform("nunique"); fam["one2one"] = nk == 1
pri = {"US": 0, "EP": 1, "JP": 2, "CN": 3}; fam["pr"] = fam.cc.map(pri)
def fam_major(f):
    c = f.groupby(["kr_appl_no", "fk"]).agg(c=("fk", "size"), p=("pr", "min")).reset_index()
    return c.sort_values(["kr_appl_no", "c", "p"], ascending=[True, False, True]).drop_duplicates("kr_appl_no").set_index("kr_appl_no").fk
kr["fam_k"] = kr.appl_no.map(fam_major(fam)); kr["fam_k11"] = kr.appl_no.map(fam_major(fam[fam.one2one]))
kr["fam_us"] = kr.appl_no.map(fam_major(fam[fam.cc == "US"]))
J["P0"] = {"kr_with_family": int(kr.fam_k.notna().sum()), "agree_kr_vs_famMajor": round(float((kr.ksic == kr.fam_k)[kr.fam_k.notna()].mean()), 4),
           "agree_kr_vs_fam_one2one": round(float((kr.ksic == kr.fam_k11)[kr.fam_k11.notna()].mean()), 4),
           "agree_kr_vs_US": round(float((kr.ksic == kr.fam_us)[kr.fam_us.notna()].mean()), 4),
           "agree_FREE_vs_kr_on_family_subset": round(float((kr.k_free == kr.ksic)[kr.fam_k.notna()].mean()), 4),
           "agree_FREE_vs_fam_on_family_subset": round(float((kr.k_free == kr.fam_k)[kr.fam_k.notna()].mean()), 4)}
log("P0", J["P0"])

# ============ P1
pf = pd.read_parquet(os.path.join(CACHE, "pat_firm.parquet")); reg = pd.read_parquet(os.path.join(CACHE, "firm_reg.parquet"))
X = pf.merge(kr[kr.fam_k.notna()][["appl_no", "ksic", "fam_k", "fam_k11", "k_free"]], on="appl_no")
rows, drows = [], []
for src in ("nps", "dart"):
    R = reg[["applicantCd", src]].dropna(); R = R[R[src].isin(KSIC50)]
    Y = X.merge(R.rename(columns={src: "r5"}), on="applicantCd")
    for lab, Z in (("가족 다수결", Y), ("가족 1:1만", Y[Y.fam_k11.notna()].assign(fam_k=lambda d: d.fam_k11))):
        Z = Z.assign(mK=(Z.ksic == Z.r5).astype(float), mF=(Z.fam_k == Z.r5).astype(float), mT=(Z.k_free == Z.r5).astype(float))
        g = Z.groupby("applicantCd"); T = pd.DataFrame({"n": g.size(), "K": g.mK.mean(), "F": g.mF.mean(), "T": g.mT.mean()})
        nf = len(T); M = T[["K", "F", "T"]].values; n = T.n.values.astype(float)
        idx = rng.integers(0, nf, (1000, nf)); bs = M[idx].mean(1)
        nb = n[idx]; bw = (M[idx] * nb[:, :, None]).sum(1) / nb.sum(1)[:, None]
        val = M.mean(0); vw = (M * n[:, None]).sum(0) / n.sum()
        for wt, v, b in (("기업평균", val, bs), ("특허가중", vw, bw)):
            r = {"기준": src.upper(), "패밀리": lab, "가중": wt, "기업": nf, "특허": int(n.sum()),
                 "KR라벨": round(v[0], 4), "해외라벨": round(v[1], 4), "TXTFREE": round(v[2], 4)}
            for nm, i, j in (("KR-해외", 0, 1), ("해외-FREE", 1, 2), ("KR-FREE", 0, 2)):
                d = b[:, i] - b[:, j]
                r[nm] = round(v[i] - v[j], 4); r[nm + "_lo"] = round(float(np.percentile(d, 2.5)), 4); r[nm + "_hi"] = round(float(np.percentile(d, 97.5)), 4)
            rows.append(r)
        D = Z[Z.ksic != Z.fam_k]
        a, bb = int((D.ksic == D.r5).sum()), int((D.fam_k == D.r5).sum())
        # 기업 부트스트랩 비율 CI
        gD = D.groupby("applicantCd"); TD = pd.DataFrame({"a": gD.apply(lambda d: (d.ksic == d.r5).sum()), "b": gD.apply(lambda d: (d.fam_k == d.r5).sum())})
        ii = rng.integers(0, len(TD), (1000, len(TD))); ra = TD.a.values[ii].sum(1); rb = TD.b.values[ii].sum(1)
        rr = (ra + 0.5) / (rb + 0.5)
        drows.append({"기준": src.upper(), "패밀리": lab, "불일치특허": len(D), "불일치기업": len(TD), "KR=등록": a, "해외=등록": bb,
                      "둘다아님": len(D) - a - bb, "r=KR/해외": round(a / max(bb, 1), 3),
                      "r_lo": round(float(np.percentile(rr, 2.5)), 3), "r_hi": round(float(np.percentile(rr, 97.5)), 3)})
    log("P1", src)
P1 = pd.DataFrame(rows); P1D = pd.DataFrame(drows)
P1.to_csv(os.path.join(OUT, "res_p1.csv"), index=False, encoding="utf-8-sig"); P1D.to_csv(os.path.join(OUT, "res_p1_disagree.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(P1.to_string()); print(P1D.to_string())

# ============ P2
def H_by_app(app_c, lab_c, nlab):
    key = app_c.astype(np.int64) * nlab + lab_c
    c = np.bincount(key, minlength=(app_c.max() + 1) * nlab).reshape(-1, nlab).astype(np.float64)
    n = c.sum(1); return (c * (c - 1)).sum(1) / np.maximum(n * (n - 1), 1), n

def perm_within(lab_c, strat_c, rng):
    p0 = np.argsort(strat_c, kind="stable"); p1 = np.lexsort((rng.random(len(strat_c)), strat_c))
    out = np.empty_like(lab_c); out[p0] = lab_c[p1]; return out

def p2_run(D, labcols, strata, minn, tag, reps=20):
    """D: 표본 전체(치환 풀). 집중도는 minn건 이상 출원인 평균."""
    res, per = [], {}
    appc, appu = pd.factorize(D.app); nA = len(appu)
    cnt = np.bincount(appc, minlength=nA); keep = cnt >= minn
    for lc in labcols:
        lab = D[lc].map(K2I).values.astype(np.int64)
        Hobs, n = H_by_app(appc, lab, 50)
        for sn, scols in strata.items():
            if lc in scols: continue
            sc_ = D.groupby(scols, sort=False).ngroup().values
            H0 = np.zeros(nA)
            for _ in range(reps):
                H0 += H_by_app(appc, perm_within(lab, sc_, rng), 50)[0]
            H0 /= reps
            E = Hobs - H0
            res.append({"표본": tag, "라벨": lc, "층": sn, "출원인": int(keep.sum()), "특허": int(cnt[keep].sum()),
                        "H관측": round(float(Hobs[keep].mean()), 4), "H기대": round(float(H0[keep].mean()), 4),
                        "초과E": round(float(E[keep].mean()), 4), "상대초과E/H0": round(float(E[keep].mean() / H0[keep].mean()), 3)})
            per[(lc, sn)] = E[keep]
    return res, per

lab_ok = kr.ksic.isin(KSIC50) & kr.k_free.isin(KSIC50) & kr.sg.fillna("").ne("") & kr.app.ne("")
D = kr[lab_ok].copy(); D["ay5"] = (D.ay // 5) * 5
strata = {"S1 메인IPC서브그룹": ["sg"], "S2 S1×출원5년": ["sg", "ay5"], "S3 S1×TXT-FREE": ["sg", "k_free"]}
r_all, _ = p2_run(D, ["ksic", "k_free"], strata, 10, "KR 전체(출원인≥10건)")
log("P2 전체")
DF = D[D.fam_k.isin(KSIC50)].copy()
r_fam, per = p2_run(DF, ["ksic", "fam_k", "k_free"], {"S1 메인IPC서브그룹": ["sg"], "S2 S1×출원5년": ["sg", "ay5"], "S3 S1×TXT-FREE": ["sg", "k_free"]}, 5, "패밀리 보유 KR(출원인≥5건)")
P2 = pd.DataFrame(r_all + r_fam)
ratio = []
for sn in ("S1 메인IPC서브그룹", "S2 S1×출원5년", "S3 S1×TXT-FREE"):
    eK, eF = per[("ksic", sn)], per[("fam_k", sn)]; m = len(eK)
    ii = rng.integers(0, m, (1000, m)); rb = eK[ii].mean(1) / eF[ii].mean(1)
    ratio.append({"층": sn, "E_KR": round(float(eK.mean()), 4), "E_해외": round(float(eF.mean()), 4), "비율": round(float(eK.mean() / eF.mean()), 3),
                  "lo": round(float(np.percentile(rb, 2.5)), 3), "hi": round(float(np.percentile(rb, 97.5)), 3)})
J["P2_ratio_family"] = ratio
P2.to_csv(os.path.join(OUT, "res_p2.csv"), index=False, encoding="utf-8-sig")
print(P2.to_string()); print(pd.DataFrame(ratio).to_string()); log("P2 끝")

# ============ P3
def p3_run(D, labcols, elig_lab, minn, tag):
    D = D.reset_index(drop=True)
    appc, appu = pd.factorize(D.app); nA = len(appu)
    labE = D[elig_lab].map(K2I).values
    C = np.zeros((nA, 50)); np.add.at(C, (appc, labE), 1)
    n = C.sum(1); srt = -np.sort(-C, 1)
    elig = (n >= minn) & (srt[:, 1] >= 0.10 * n)
    out = []
    sgc, sgu = pd.factorize(D.sg); mgc, mgu = pd.factorize(D.mg)
    for lc in labcols:
        lab = D[lc].map(K2I).values
        CA = np.zeros((nA, 50)); np.add.at(CA, (appc, lab), 1)
        res = {}
        for lvl, gc in (("sg", sgc), ("mg", mgc)):
            G = np.zeros((gc.max() + 1, 50)); np.add.at(G, (gc, lab), 1)
            k = pd.factorize(pd.Series(appc.astype(np.int64) * (gc.max() + 1) + gc))[0]
            AG = np.zeros((k.max() + 1, 50)); np.add.at(AG, (k, lab), 1)
            res[lvl] = (G, AG, gc, k)
        sel = np.where(elig[appc])[0]
        ipcm = np.full(len(D), -1); tot_s = np.zeros(len(D))
        for lvl in ("mg", "sg"):  # sg 우선, 부족하면 mg
            G, AG, gc, k = res[lvl]
            for s in range(0, len(sel), 200000):
                ix = sel[s:s + 200000]; V = G[gc[ix]] - AG[k[ix]]; tt = V.sum(1)
                ok = tt >= 5; ipcm[ix[ok]] = V[ok].argmax(1); tot_s[ix[ok]] = tt[ok]
        V = CA[appc[sel]].copy(); V[np.arange(len(sel)), lab[sel]] -= 1; appm = np.full(len(D), -1); appm[sel] = V.argmax(1)
        m = np.zeros(len(D), bool); m[sel] = True; m &= (ipcm >= 0) & (ipcm != appm)
        a_ = (lab == ipcm)[m]; b_ = (lab == appm)[m]; aa = appc[m]
        # 출원인 부트스트랩
        A = pd.DataFrame({"g": aa, "a": a_, "b": b_, "o": 1}).groupby("g").sum()
        ii = rng.integers(0, len(A), (500, len(A))); sa = A.a.values[ii].sum(1); sb = A.b.values[ii].sum(1); so = A.o.values[ii].sum(1)
        out.append({"표본": tag, "라벨": lc, "대상출원인": int(elig.sum()), "IPC최빈≠출원인최빈 특허": int(m.sum()),
                    "a=라벨=IPC최빈": round(float(a_.mean()), 4), "b=라벨=출원인최빈": round(float(b_.mean()), 4),
                    "b/a": round(float(b_.mean() / a_.mean()), 3), "b/a_lo": round(float(np.percentile(sb / sa, 2.5)), 3),
                    "b/a_hi": round(float(np.percentile(sb / sa, 97.5)), 3), "둘다아님": round(float(1 - a_.mean() - b_.mean()), 4)})
    return out
mg_ok = D.mg.fillna("").ne("")
r3 = p3_run(D[mg_ok], ["ksic", "k_free"], "ksic", 10, "KR 전체(출원인≥10건, 다업종)")
r3 += p3_run(DF[DF.mg.fillna("").ne("")], ["ksic", "fam_k", "k_free"], "ksic", 10, "패밀리 보유 KR(출원인≥10건, 다업종)")
P3 = pd.DataFrame(r3); P3.to_csv(os.path.join(OUT, "res_p3.csv"), index=False, encoding="utf-8-sig"); print(P3.to_string())
json.dump(J, open(os.path.join(OUT, "res_tests.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("done")
