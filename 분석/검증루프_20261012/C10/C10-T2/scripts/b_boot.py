# -*- coding: utf-8 -*-
r"""[2][3] 공식 seed 예측(학습 seed 0·1, 음성 seed 1000+t)의 군집 부트스트랩. B=2000을 rep 구간으로 쪼개 실행.
군집: firm(법인 Poisson) | ksic_t(목표 KSIC: 대상 sc의 최빈 제공 KSIC, 풀 특허 ky≤2012) | ksic_f(법인 최빈 제공 KSIC, ky≤2012)
      | sc(IPC 서브클래스) | firm_sc(곱 가중) | firm_ksic_t(곱 가중). 원점 간 같은 군집은 같은 가중.
가중: rep r마다 default_rng([20261013, 군집번호, r]) — 구간을 나눠도 같은 값.
사용: python -B b_boot.py <cltype> <lo> <hi> → cache/boot_<cltype>_<lo>_<hi>.npz (auc[rep, arm, origin])"""
import sys
sys.dont_write_bytecode = True
from c10_common import *
from b_lib import AucPrep

CLT = {"firm": 1, "ksic_t": 2, "ksic_f": 3, "sc": 4, "firm_sc": 5, "firm_ksic_t": 6}
ARMS_OFF = ["P50", "FREE50", "TXT50", "FREEIPC50", "HSC1_50", "HMG3_50", "P50FIPC", "P50HSC1", "P50HMG3"]
ZTAGS = ["FREEIPC50-HSC1_50-P50HSC1", "HMG3_50-P50FIPC-P50HMG3"]


def mode_map(df, key):
    g = df.groupby([key, "ksic"]).size().reset_index(name="n").sort_values([key, "n", "ksic"], ascending=[True, False, True])
    return g.drop_duplicates(key).set_index(key).ksic


def clusters():
    P = pd.read_parquet(os.path.join(NC8, "cache", "patents.parquet"), columns=["applicantCd", "sc", "ksic", "ky"])
    P = P[P.ky <= 2012]
    return {"ksic_t": mode_map(P, "sc"), "ksic_f": mode_map(P, "applicantCd")}


def off_preds(c):
    a = pd.read_parquet(os.path.join(CACHE, f"pred_c{c}_k0_P50-FREE50-TXT50_p012.parquet"), columns=["P50_p0", "FREE50_p0", "TXT50_p0"])
    zs = [pd.read_parquet(os.path.join(CACHE, f"pred_c{c}_k0_{zt}_p0.parquet")) for zt in ZTAGS]
    d = pd.concat([a] + zs, axis=1); d.columns = [x[:-3] for x in d.columns]
    return d


if __name__ == "__main__":
    clt, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    CM = clusters()
    rows = {c: pd.read_parquet(os.path.join(NC8, "cache", f"A_o{c}.parquet"), columns=["applicantCd", "sc", "first_ky"]) for c in C_ORIG}
    allr = pd.concat(rows.values())
    comps = {"firm": ["firm"], "ksic_t": ["ksic_t"], "ksic_f": ["ksic_f"], "sc": ["sc"], "firm_sc": ["firm", "sc"], "firm_ksic_t": ["firm", "ksic_t"]}[clt]
    def keyser(a, comp):
        if comp == "firm": return a.applicantCd
        if comp == "sc": return a.sc
        if comp == "ksic_t": return a.sc.map(CM["ksic_t"])
        if comp == "ksic_f": return a.applicantCd.map(CM["ksic_f"])
    U = {comp: np.array(sorted(keyser(allr, comp).dropna().unique())) for comp in comps}
    for comp in comps:
        assert keyser(allr, comp).notna().all(), comp
    IDX = {comp: {u: i for i, u in enumerate(U[comp])} for comp in comps}
    PR = {}
    for j, c in enumerate(C_ORIG):
        a = rows[c]; y = a.first_ky.notna().values.astype(np.int8); d = off_preds(c)
        cl = [keyser(a, comp).map(IDX[comp]).values.astype(np.int64) for comp in comps]
        for i, arm in enumerate(ARMS_OFF):
            PR[(i, j)] = AucPrep(y, d[arm].values.astype(np.float64), cl)
    log("prep", clt, {comp: len(U[comp]) for comp in comps})
    AU = np.full((hi - lo, len(ARMS_OFF), 5), np.nan)
    for r in range(lo, hi):
        ws = [np.random.default_rng([20261013, CLT[clt], q, r]).poisson(1.0, len(U[comp])).astype(np.float64) for q, comp in enumerate(comps)]
        for key, A in PR.items():
            AU[(r - lo,) + key] = A.auc(ws)
        if (r - lo) % 100 == 0:
            log("rep", r)
    np.savez_compressed(os.path.join(CACHE, f"boot_{clt}_{lo}_{hi}.npz"), auc=AU, arms=np.array(ARMS_OFF), origins=np.array(C_ORIG),
                        n_clusters=np.array([len(U[comp]) for comp in comps]))
    log("done", clt, lo, hi)
