# -*- coding: utf-8 -*-
r"""C10-T3 verify_0 v1: 독립 재계산(pandas/pyarrow, duckdb·alt_assign·alt_fix 미사용).
FREE-IPC(L_REG 연결 상속 + IPC 표)와 TXT(L_REG 상속, 표 값은 txt_assign 저장값) 해외 재현율(발명특허, 출원 2015–19)과
ρ(해외 출원 RTA 2017–19 ↔ 수출 RCA 2015–17)를 원천(_shared/meta, rec_conc, kr_txtfree, fam_reg, fam_link CN, txt_assign, map5)에서 직접 만든다.
반박 시도: (r1) 결함 R8 연결로 같은 코드를 돌려 저장값(51.0/48.2/53.1, ρ 0.571)을 재현하는가, (r2) 동률을 무작위로 깨면 값이 얼마나 흔들리나,
(r3) 해외 문서가 여러 KR 문서에 붙을 때 '첫 KR 문서'만 쓰면 어떻게 되나, (r4) ρ CI를 다른 seed로.
쓰기: verify_0/v1_out.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.dataset as ds
from scipy.stats import spearmanr
V = os.path.dirname(os.path.abspath(__file__))
MC = os.path.abspath(os.path.join(V, "..", "..", "..", "..", "모델경쟁_20260927"))
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
meta = ds.dataset(os.path.join(MC, "_shared", "meta"), format="parquet")
cols = ["cc", "appl_no", "appl_year", "ksic", "prio_cc"]
F = meta.to_table(columns=cols, filter=(ds.field("cc") != "KR") & (ds.field("appl_year") >= 2015) & (ds.field("appl_year") <= 2019)).to_pandas()
K = meta.to_table(columns=["appl_no", "appl_year"], filter=ds.field("cc") == "KR").to_pandas()
F = F.astype({c: object for c in F.columns if str(F[c].dtype) == "category"}); K = K.astype({c: object for c in K.columns if str(K[c].dtype) == "category"})
L_ = F.appl_no.str.len(); s5 = F.appl_no.str[4:5]
F["is_inv"] = np.where(F.cc == "CN", L_.isin([12, 14]) & s5.isin(["1", "8"]), np.where(F.cc == "JP", L_.isin([8, 17]), True))
F = F[F.is_inv].copy(); F["kr_origin"] = F.prio_cc.fillna("").str.split("|").str[0] == "KR"
log("해외 발명 2015–19", F.cc.value_counts().to_dict())
rc = pq.read_table(os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "rec_conc.parquet"), columns=["cc", "appl_no", "sg", "code"]).to_pandas()
F = F.merge(rc, on=["cc", "appl_no"], how="inner")   # c1과 같이 rec_conc 있는 문서만
kf = pq.read_table(os.path.join(MC, "T1_산업대기실", "TXT_R9", "cache", "kr_txtfree.parquet")).to_pandas().dropna()
rck = rc[rc.cc == "KR"][["appl_no", "sg"]]
T = K[K.appl_year <= 2019].merge(rck, on="appl_no").merge(kf, on="appl_no")
T["mg"] = T.sg.str.split("/").str[0]; T["sc"] = T.sg.str[:4]
F["mg"] = F.sg.str.split("/").str[0]; F["sc"] = F.sg.str[:4]


def mode_tab(df, key, val, min_n=3, rng=None):
    g = df[df[key].notna() & (df[key] != "")].groupby([key, val]).size().rename("n").reset_index()
    tot = g.groupby(key).n.transform("sum"); g = g[tot >= min_n]
    if rng is None:
        g = g.assign(kk=g[val].astype(int)).sort_values([key, "n", "kk"], ascending=[True, False, True])
    else:
        g = g.assign(r=rng.random(len(g))).sort_values([key, "n", "r"], ascending=[True, False, True])
    return g.drop_duplicates(key).set_index(key)[val]


def free_table_labels(rng=None):
    out = pd.Series(index=F.index, dtype=object)
    for lvl in ("sg", "mg", "sc"):
        tab = mode_tab(T, lvl, "k_free", rng=rng); m = F[lvl].map(tab); out = out.fillna(m)
    return out


reg = pq.read_table(os.path.join(MC, "_shared_report", "label_origin_R10", "cache", "fam_reg.parquet"), columns=["cc", "appl_no", "kr_appl_no"]).to_pandas()
r8 = pq.read_table(os.path.join(MC, "_ideas_R8", "cache", "fam_link.parquet")).to_pandas()
LREG = pd.concat([reg, r8[r8.cc == "CN"]]).drop_duplicates()


def fam_labels(link, src, val, rng=None, first_only=False):
    x = link.merge(src[["appl_no", val]].rename(columns={"appl_no": "kr_appl_no"}), on="kr_appl_no")
    if first_only:
        x = x.sort_values(["cc", "appl_no", "kr_appl_no"]).drop_duplicates(["cc", "appl_no"])
    g = x.groupby(["cc", "appl_no", val]).size().rename("n").reset_index()
    if rng is None:
        g = g.assign(kk=g[val].astype(int)).sort_values(["cc", "appl_no", "n", "kk"], ascending=[True, True, False, True])
    else:
        g = g.assign(r=rng.random(len(g))).sort_values(["cc", "appl_no", "n", "r"], ascending=[True, True, False, True])
    g = g.drop_duplicates(["cc", "appl_no"]).set_index(["cc", "appl_no"])[val]
    return pd.Series(pd.MultiIndex.from_frame(F[["cc", "appl_no"]]).map(g), index=F.index)


tabF = free_table_labels()
OUT = {}
def recall(lab):
    return {cc: round(float((lab[F.cc == cc] == F.ksic[F.cc == cc]).mean()) * 100, 3) for cc in ("US", "JP", "CN", "EP")}
FREE = {"pure": tabF, "reg": fam_labels(LREG, kf, "k_free").fillna(tabF), "r8": fam_labels(r8, kf, "k_free").fillna(tabF),
        "reg_first_only": fam_labels(LREG, kf, "k_free", first_only=True).fillna(tabF)}
OUT["free_recall"] = {k: recall(v) for k, v in FREE.items()}
log("FREE recall", OUT["free_recall"])
rng = np.random.default_rng(7)
FREE["reg_randtie"] = fam_labels(LREG, kf, "k_free", rng=rng).fillna(free_table_labels(rng=rng))
OUT["free_recall"]["reg_randtie"] = recall(FREE["reg_randtie"])
# TXT: txt_assign 저장값(풀 안) 중 family 문서는 표 값으로 되돌릴 수 없으므로(표 값 미보관) 'family'를 L_REG 상속으로 바꾼 효과만 독립 계산
ta = pq.read_table(os.path.join(MC, "T3_기술셀부상예측", "TXT_R9", "txt_assign.parquet"), columns=["cc", "appl_no", "code", "ksic_txt", "method"]).to_pandas()
km = pd.read_csv(os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "conc_ksic_map.csv"), dtype=str)
multi = set(km.groupby("code").ksic.size().loc[lambda s: s > 1].index)
krt = ta[ta.cc == "KR"][["appl_no", "code", "ksic_txt"]].rename(columns={"appl_no": "kr_appl_no", "code": "kcode"})
fo = ta[(ta.cc != "KR") & ta.code.isin(multi)][["cc", "appl_no", "code"]]
x = LREG.merge(fo, on=["cc", "appl_no"]).merge(krt, on="kr_appl_no"); x = x[x.kcode == x.code]
g = x.groupby(["cc", "appl_no", "ksic_txt"]).size().rename("n").reset_index().assign(kk=lambda d: d.ksic_txt.astype(int))
g = g.sort_values(["cc", "appl_no", "n", "kk"], ascending=[True, True, False, True]).drop_duplicates(["cc", "appl_no"])
OUT["txt_family_docs_reg_in_pool"] = g.cc.value_counts().to_dict()
OUT["txt_family_docs_r8_in_pool"] = ta[ta.method == "family"].cc.value_counts().to_dict()
json.dump(OUT, open(os.path.join(V, "v1_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
# ---- ρ ----
ek = pd.read_csv(os.path.join(MC, "_shared_report", "fusion_advantage", "map5", "m4_exports_ksic_5rep.csv"), dtype={"k5": str})
v = ek[ek.year.isin([2015, 2016, 2017])].groupby(["k5", "rep"]).v.sum().unstack(); s_ = v / v.sum(); RCA = s_.div(s_.mean(1), axis=0).KR
K50 = sorted(km.ksic); RCA = RCA.reindex(K50)
P3 = F.appl_year.between(2017, 2019)


def abroad_rta(lab):
    d = pd.DataFrame({"k": lab, "o": F.kr_origin})[P3 & lab.notna()]
    k = d[d.o].groupby("k").size().reindex(K50).fillna(0); a = d.groupby("k").size().reindex(K50)
    return (k / a) / (k.sum() / a.sum())


def rho(x):
    d = pd.concat([x, RCA], axis=1).dropna(); return float(spearmanr(d.iloc[:, 0], d.iloc[:, 1]).statistic), d


rp, dp = rho(abroad_rta(F.ksic))
OUT["rho_prov"] = round(rp, 4)
OUT["rho_free"] = {k: round(rho(abroad_rta(vv))[0], 4) for k, vv in FREE.items()}
# r4: 다른 seed 부트스트랩 CI (제공−FREE reg)
xa = abroad_rta(FREE["reg"]); d = pd.concat([abroad_rta(F.ksic), xa, RCA], axis=1).dropna().values
rng2 = np.random.default_rng(99); bs = []
for _ in range(2000):
    i = rng2.integers(0, len(d), len(d))
    bs.append((spearmanr(d[i, 0], d[i, 2]).statistic, spearmanr(d[i, 1], d[i, 2]).statistic))
bs = np.array(bs)
OUT["free_reg_ci_seed99"] = dict(ci_alt=np.nanpercentile(bs[:, 1], [2.5, 97.5]).round(3).tolist(), ci_d=np.nanpercentile(bs[:, 0] - bs[:, 1], [2.5, 97.5]).round(3).tolist())
log("rho", OUT["rho_prov"], OUT["rho_free"], OUT["free_reg_ci_seed99"])
json.dump(OUT, open(os.path.join(V, "v1_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log("done")
