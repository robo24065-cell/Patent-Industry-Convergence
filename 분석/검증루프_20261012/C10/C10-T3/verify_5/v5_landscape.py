# -*- coding: utf-8 -*-
r"""C10-T3 verify_5: 지형도(5개국 RTA, 출원 2017–19, 발명) F·H 시나리오를 독립 재계산해 E-C10T3-06 반박 시도.
작업자 s3_rho.py·alt_fix·links.parquet·저장 alt_assign의 해외 라벨 열은 쓰지 않는다. 해외 라벨은 원천(meta, rec_conc, kr_txtfree,
fam_reg, fam_link, txt_assign)에서 직접 만든다. KR 쪽 라벨(k_free, k_txt)은 결함과 무관하므로 저장 alt_assign KR에서 읽되,
kr_txtfree·txt_assign과 대조한다. 비교용으로 작업자 alt_fix와 문서 단위 라벨 일치도 낸다(읽기만).
쓰기: verify_5/v5_out.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.dataset as ds
from scipy.stats import spearmanr
V = os.path.dirname(os.path.abspath(__file__))
MC = os.path.abspath(os.path.join(V, "..", "..", "..", "..", "모델경쟁_20260927"))
C5 = os.path.join(MC, "_shared_report", "fusion_advantage", "c5_R10")
WK = os.path.abspath(os.path.join(V, ".."))
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
OUT = {}
CCS = ["KR", "US", "JP", "CN", "EP"]
km = pd.read_csv(os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "conc_ksic_map.csv"), dtype=str)[["ksic", "code"]]
K50 = sorted(km.ksic); kc = km.groupby("code").ksic.agg(list)
single = {c: v[0] for c, v in kc.items() if len(v) == 1}; multi = {c for c, v in kc.items() if len(v) > 1}
wq = (1 / km.groupby("code").ksic.size()).to_dict()

# ---- 해외 문서(발명, 출원 2017–19) ----
meta = ds.dataset(os.path.join(MC, "_shared", "meta"), format="parquet")
F = meta.to_table(columns=["cc", "appl_no", "appl_year", "ksic", "prio_cc"],
                  filter=(ds.field("cc") != "KR") & (ds.field("appl_year") >= 2017) & (ds.field("appl_year") <= 2019)).to_pandas()
F = F.astype({c: object for c in F.columns if str(F[c].dtype) == "category"})
L_ = F.appl_no.str.len(); s5 = F.appl_no.str[4:5]
F = F[np.where(F.cc == "CN", L_.isin([12, 14]) & s5.isin(["1", "8"]), np.where(F.cc == "JP", L_.isin([8, 17]), True))].copy()
rc = pq.read_table(os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "rec_conc.parquet"), columns=["cc", "appl_no", "sg", "code"]).to_pandas()
rc = rc.astype({c: object for c in rc.columns if str(rc[c].dtype) == "category"})
F = F.merge(rc, on=["cc", "appl_no"], how="inner").reset_index(drop=True)
F.loc[~F.code.isin(kc.index), "code"] = None
F["mg"] = F.sg.str.split("/").str[0]; F["sc"] = F.sg.str[:4]
log("해외 발명 2017–19(rec_conc 있음)", F.cc.value_counts().to_dict())
assert not F.duplicated(["cc", "appl_no"]).any()
FIX = pd.MultiIndex.from_frame(F[["cc", "appl_no"]])


def first_mode(df, keys, val):
    g = df.groupby(keys + [val]).size().rename("n").reset_index().assign(kk=lambda d: d[val].astype(int))
    g = g.sort_values(keys + ["n", "kk"], ascending=[True] * len(keys) + [False, True]).drop_duplicates(keys)
    return g.set_index(keys)[val]


# ---- FREE-IPC: KR kr_txtfree 라벨의 (sg→mg→sc) 최빈표(키당 ≥3, KR 출원 ≤2019) + 패밀리 상속 ----
Kmeta = meta.to_table(columns=["appl_no", "appl_year"], filter=ds.field("cc") == "KR").to_pandas()
Kmeta = Kmeta.astype({c: object for c in Kmeta.columns if str(Kmeta[c].dtype) == "category"})
kf = pq.read_table(os.path.join(MC, "T1_산업대기실", "TXT_R9", "cache", "kr_txtfree.parquet")).to_pandas().dropna()
rck = rc[rc.cc == "KR"][["appl_no", "sg"]]
T = Kmeta[Kmeta.appl_year <= 2019].merge(rck, on="appl_no").merge(kf, on="appl_no")
T["mg"] = T.sg.str.split("/").str[0]; T["sc"] = T.sg.str[:4]
tabF = pd.Series(index=F.index, dtype=object)
for lvl in ("sg", "mg", "sc"):
    g = T[T[lvl].notna() & (T[lvl] != "")]
    g = g[g.groupby(lvl)[lvl].transform("size") >= 3]
    tb = first_mode(g, [lvl], "k_free")
    tabF = tabF.fillna(F[lvl].map(tb))
reg = pq.read_table(os.path.join(MC, "_shared_report", "label_origin_R10", "cache", "fam_reg.parquet"), columns=["cc", "appl_no", "kr_appl_no"]).to_pandas()
r8 = pq.read_table(os.path.join(MC, "_ideas_R8", "cache", "fam_link.parquet")).to_pandas()
reg = reg.astype(object); r8 = r8.astype(object)
LINK = {"r8": r8.drop_duplicates(), "reg": pd.concat([reg, r8[r8.cc == "CN"]]).drop_duplicates()}


def fam_free(link):
    x = link.merge(kf.rename(columns={"appl_no": "kr_appl_no"}), on="kr_appl_no")
    g = first_mode(x, ["cc", "appl_no"], "k_free")
    return pd.Series(FIX.map(g), index=F.index)


FREE = {"pure": tabF}
for nm, L in LINK.items():
    FREE[nm] = fam_free(L).fillna(tabF)

# ---- TXT: 코드 없음→None, 단일→그 KSIC, 복수→(풀 안) 같은 코드 KR 문서 ksic_txt 상속 → (코드,sg/mg/sc)≥3 → 코드 최빈 ----
ta = pq.read_table(os.path.join(MC, "T3_기술셀부상예측", "TXT_R9", "txt_assign.parquet"), columns=["cc", "appl_no", "code", "ksic_txt", "method"]).to_pandas()
ta = ta.astype({c: object for c in ta.columns if str(ta[c].dtype) == "category"})
krt = ta[(ta.cc == "KR") & (ta.method == "text")].merge(rck, on="appl_no")
krt["mg"] = krt.sg.str.split("/").str[0]; krt["sc"] = krt.sg.str[:4]
labT = pd.Series(index=F.index, dtype=object)
for lvl in ("sg", "mg", "sc"):
    g = krt[krt.code.notna() & krt[lvl].notna() & (krt[lvl] != "")]
    g = g[g.groupby(["code", lvl]).code.transform("size") >= 3]
    tb = first_mode(g, ["code", lvl], "ksic_txt")
    labT = labT.fillna(pd.Series(pd.MultiIndex.from_frame(F[["code", lvl]]).map(tb), index=F.index))
labT = labT.fillna(F.code.map(first_mode(krt[krt.code.notna()], ["code"], "ksic_txt")))
labT[F.code.isna()] = None
sg_ = F.code.map(single); labT[sg_.notna()] = sg_[sg_.notna()]
pool = FIX.isin(pd.MultiIndex.from_frame(ta[ta.cc != "KR"][["cc", "appl_no"]]))
krk = ta[ta.cc == "KR"][["appl_no", "code", "ksic_txt"]].rename(columns={"appl_no": "kr_appl_no", "code": "kcode"})
fo = ta[(ta.cc != "KR") & ta.code.isin(multi)][["cc", "appl_no", "code"]]
TXT = {"pure": labT}
for nm, L in LINK.items():
    x = L.merge(fo, on=["cc", "appl_no"]).merge(krk, on="kr_appl_no"); x = x[x.kcode == x.code]
    fam = pd.Series(FIX.map(first_mode(x, ["cc", "appl_no"], "ksic_txt")), index=F.index)
    fam[~pool | ~F.code.isin(multi)] = None
    l2 = fam.fillna(labT); l2[F.code.isna()] = None
    TXT[nm] = l2
# r8 TXT는 저장 txt_assign 값(풀 안 문서는 txt_assign ksic_txt 그대로)이 원래 정의 — 그것도 따로 만든다
tav = pd.Series(FIX.map(ta[ta.cc != "KR"].set_index(["cc", "appl_no"]).ksic_txt), index=F.index)
st = labT.copy(); m_ = F.code.isin(multi) & tav.notna(); st[m_] = tav[m_]; st[F.code.isna()] = None
TXT["stored_def"] = st
log("라벨 구성 완료")

# ---- KR 쪽(결함 무관): 저장 alt_assign KR ----
AK = pq.read_table(os.path.join(C5, "alt_assign", "alt_KR.parquet"), columns=["appl_no", "appl_year", "is_inv", "ksic_prov", "code", "k_txt", "k_free"]).to_pandas()
AK = AK.astype({c: object for c in AK.columns if str(AK[c].dtype) == "category"})
AK = AK[AK.is_inv & AK.appl_year.between(2017, 2019)]
chk = AK.merge(kf.rename(columns={"k_free": "kf2"}), on="appl_no", how="left")
OUT["kr_free_eq_kr_txtfree"] = float((chk.k_free.fillna("x") == chk.kf2.fillna("x")).mean())
# KR alt_fix 와 동일성
AFK = pq.read_table(os.path.join(WK, "cache", "alt_fix_KR.parquet"), columns=["appl_no", "is_inv", "appl_year", "k_free_reg", "k_txt_reg", "k_free_stored", "k_txt_stored"]).to_pandas()
AFK = AFK[AFK.is_inv & AFK.appl_year.between(2017, 2019)].merge(AK[["appl_no", "k_free", "k_txt"]], on="appl_no")
OUT["kr_altfix_reg_eq_stored"] = {c: float((AFK[f"k_{c}_reg"].fillna("x") == AFK[f"k_{c}"].fillna("x")).mean()) for c in ("free", "txt")}
log("KR 대조", OUT["kr_free_eq_kr_txtfree"], OUT["kr_altfix_reg_eq_stored"])


def cnt_kr(col, weighted=False):
    if weighted:
        d = AK[AK.code.notna()].merge(km, on="code"); d["w"] = d.code.map(wq)
        return d.groupby("ksic").w.sum()
    return AK.groupby(col).size()


def cnt_for(lab, cc):
    m = (F.cc == cc) & lab.notna()
    return lab[m].groupby(lab[m]).size()


def cnt_for_cweq(cc):
    d = F[(F.cc == cc) & F.code.notna()][["code"]].merge(km, on="code"); d["w"] = d.code.map(wq)
    return d.groupby("ksic").w.sum()


def table(kr, forlab=None, cweq=False):
    out = {"KR": kr}
    for cc in CCS[1:]:
        out[cc] = cnt_for_cweq(cc) if cweq else cnt_for(forlab, cc)
    return pd.DataFrame(out).reindex(K50).fillna(0)


def rta(n):
    sh = n / n.sum(); return sh.div(sh.mean(1), axis=0)


rt_prov = rta(table(cnt_kr("ksic_prov"), F.ksic)); lp = rt_prov.idxmax(1); krp = set(lp.index[lp == "KR"])
OUT["prov"] = dict(kr_rta_gt1=int((rt_prov.KR > 1).sum()), kr_leader=len(krp))


def land(rt):
    ld = rt.idxmax(1); krl = set(ld.index[ld == "KR"])
    r = spearmanr(rt.KR, rt_prov.KR).statistic
    return dict(kr_rta_gt1=int((rt.KR > 1).sum()), kr_leader=len(krl), kept=len(krl & krp), rho_KR=round(float(r), 3), rho_KR_raw=float(r))


L = {}
L["cweq_F"] = land(rta(table(cnt_kr(None, True), cweq=True)))
for arm in ("stored_def", "r8", "reg", "pure"):
    L[f"txt_F_{arm}"] = land(rta(table(cnt_kr("k_txt"), TXT[arm])))
    L[f"txt_H_{arm}"] = land(rta(table(cnt_kr("ksic_prov"), TXT[arm])))
for arm in ("r8", "reg", "pure"):
    L[f"free_F_{arm}"] = land(rta(table(cnt_kr("k_free"), FREE[arm])))
    L[f"free_H_{arm}"] = land(rta(table(cnt_kr("ksic_prov"), FREE[arm])))
OUT["landscape"] = L
for k, v in L.items(): log(k, v)

# ---- 작업자 alt_fix와 문서 단위 대조(해외, 발명 2017–19) ----
cmp = {}
for cc in CCS[1:]:
    A = pq.read_table(os.path.join(WK, "cache", f"alt_fix_{cc}.parquet"), columns=["appl_no", "appl_year", "is_inv", "k_free_reg", "k_txt_reg", "k_free_stored", "k_txt_stored"]).to_pandas()
    A = A.astype({c: object for c in A.columns if str(A[c].dtype) == "category"})
    A = A[A.is_inv & A.appl_year.between(2017, 2019)]
    sub = F[F.cc == cc][["appl_no"]].copy()
    for nm, s in (("free_reg", FREE["reg"]), ("txt_reg", TXT["reg"]), ("free_stored", FREE["r8"]), ("txt_stored", TXT["stored_def"])):
        sub[nm] = s[F.cc == cc].values
    M = sub.merge(A, on="appl_no", how="outer", indicator=True)
    cmp[cc] = dict(n_mine=int(len(sub)), n_altfix=int(len(A)), both=int((M._merge == "both").sum()))
    B = M[M._merge == "both"]
    for nm in ("free_reg", "txt_reg", "free_stored", "txt_stored"):
        cmp[cc][nm] = float((B[nm].fillna("x").astype(str) == B[f"k_{nm}"].fillna("x").astype(str)).mean())
OUT["doc_agree_vs_altfix"] = cmp
log("문서 대조", cmp)
json.dump(OUT, open(os.path.join(V, "v5_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log("done")
