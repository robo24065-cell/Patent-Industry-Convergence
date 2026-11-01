# -*- coding: utf-8 -*-
r"""N7 R6 재검증 공용. 동결 n7lib/n7feat를 그대로 import하되 캐시 경로만 VERIFY_R6\cache로 돌린다(동결 폴더에는 쓰지 않음).
- 등록원부·app_firm·발명자 파생 캐시는 동결 cache에서 '읽어서' 이 폴더 cache로 복사(내용 동일, 원본 수정 없음).
- 국민연금 집계(nps_firm_*)·시도(nps_sido_*)는 원자료(_shared\ext\firm_nps_wkpl_*)에서 이 폴더에 새로 만든다.
- NPS 별칭 보정(PROTOCOL_R6 §2 = N4 VERIFY_R5 v6 방식): 사업장 정체성 (사업자번호 앞6, 최초적용일)로 2026 상호 키 K와 옛 상호 키를 잇고,
  스냅샷 s에 K가 없으면 같은 정체성의 옛 상호 행을 K로 복사한다."""
import os, sys, shutil, json, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE); sys.path.insert(0, TR)
import numpy as np, pandas as pd
import n7lib
VC = os.path.join(HERE, "cache"); os.makedirs(VC, exist_ok=True)
FROZEN_CACHE = n7lib.CACHE
COPY = ["app_firm.parquet", "reg_ap.parquet", "reg_pt.parquet", "reg_lapnp.parquet", "reg_trf.parquet", "reg_py.parquet", "reg_rec.parquet",
        "reg_own.parquet", "reg_sur.parquet", "reg_inv.parquet", "reg_pubapp.parquet", "foreign_events.parquet"]
for f in COPY:
    if not os.path.exists(os.path.join(VC, f)):
        shutil.copyfile(os.path.join(FROZEN_CACHE, f), os.path.join(VC, f))
n7lib.CACHE = VC
import n7feat
n7feat.CACHE = VC
from n7lib import norm, EXT, log, auc, skill, topk, predict, rank01, cluster_boot
SPEC = json.load(open(os.path.join(TR, "final_spec_r5.json"), encoding="utf-8"))
FE, M0F, SBF, PLF = SPEC["features"], SPEC["baseline_M0"], SPEC["strong_baseline_SB"], SPEC["pilot_champion"]
SEEDS = tuple(SPEC["model"]["seeds"])
# 비특허(외부데이터) 피처 전부 = 국민연금(M0+확장+시도·지역 진입률) + NTIS 참여(M2) + KSIC 정부연구비(M3) + 동종 KSIC 진입률
# (동종 KSIC 진입률은 특허 KSIC 라벨 가중이지만 NTIS 결과 기반 외부 신호로 분류)
EXT_ALL = M0F + ["g_emp2", "log_wkpl", "g_wage1", "d_hire_rate", "nps_gone", "sido", "peer_entry_r", "peer_entry_r_rel"] + \
          ["g_part_any", "g_part_n", "g_part_proj", "g_part_univ", "g_part_gri", "g_part_firm", "g_part_rec", "g_part_share"] + \
          ["k_rnd_g", "k_rnd_sh", "peer_entry_k"]
PATENT_ONLY = [f for f in FE if f not in EXT_ALL]
ALLSN = ["201512", "201612", "201712", "201812", "201912", "202012", "202112", "202212", "202312", "202412", "202512", "202608"]


def alias_maps(mode="final"):
    """스냅샷별 {K(2026 상호 키): 옛 상호 키} — N4 v6와 같은 규칙(정체성 ba가 K 하나에만 대응할 때만)."""
    fc = os.path.join(VC, f"alias_map_{mode}.parquet")
    if os.path.exists(fc):
        d = pd.read_parquet(fc)
        return {s: dict(zip(g.K, g.key)) for s, g in d.groupby("snap")}
    A2K = n7lib.a2key(); Kset = set(A2K.values())
    raw, ident = {}, []
    for s in ALLSN:
        d = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{s}.parquet"), columns=["name", "biz6", "adpt_dt"])
        d["key"] = d.biz6.fillna("") + "|" + d.name.map(norm)
        d["ba"] = d.biz6.fillna("") + "|" + d.adpt_dt.fillna("").astype(str)
        raw[s] = d[["key", "ba"]]
        ident.append(d.loc[d.key.isin(Kset) & d.adpt_dt.notna(), ["ba", "key"]])
    ident = pd.concat(ident).drop_duplicates()
    nK = ident.groupby("ba").key.nunique()
    ident = ident[ident.ba.map(nK) == 1].set_index("ba").key
    rows = []
    for snap in n7lib.snaps_for(mode):
        d = raw[snap.replace("-", "")]
        have = set(d.key)
        c = d[d.ba.isin(ident.index)].copy()
        c["K"] = c.ba.map(ident)
        c = c[(c.key != c.K) & ~c.K.isin(have) & c.key.isin(have)]
        al = c.groupby(["K", "key"]).size().reset_index(name="cnt").sort_values("cnt", ascending=False).drop_duplicates("K")
        al["snap"] = snap
        rows.append(al[["snap", "K", "key"]])
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(fc, index=False)
    log("identity pairs", len(ident), "aliases", out.groupby("snap").size().to_dict())
    return {s: dict(zip(g.K, g.key)) for s, g in out.groupby("snap")}


def install_alias(mode="final"):
    """n7lib.load_nps / n7feat.load_nps / n7feat.nps_sido 를 별칭 보정판으로 바꿔 끼운다(동결 파일은 수정하지 않음)."""
    AM = alias_maps(mode)
    orig_load_nps = n7lib.load_nps
    orig_sido = n7feat.nps_sido
    memo = {}

    def load_nps_alias(m):
        if "nps" in memo:
            return memo["nps"]
        NPS = orig_load_nps(m)
        for snap, agg in NPS.items():
            am = AM.get(snap, {})
            have = set(agg.key)
            pairs = [(K, k) for K, k in am.items() if K not in have and k in have]
            if pairs:
                ix = agg.set_index("key")
                add = ix.loc[[k for _, k in pairs]].reset_index(drop=True)
                add["key"] = [K for K, _ in pairs]
                NPS[snap] = pd.concat([agg, add], ignore_index=True)
        memo["nps"] = NPS
        return NPS

    def nps_sido_alias(snap):
        if ("sido", snap) in memo:
            return memo[("sido", snap)]
        sd = orig_sido(snap)
        am = AM.get(snap, {})
        add = {K: sd[k] for K, k in am.items() if K not in sd.index and k in sd.index}
        if add:
            sd = pd.concat([sd, pd.Series(add, name="sido")])
        memo[("sido", snap)] = sd
        return sd

    n7lib.load_nps = load_nps_alias
    n7feat.load_nps = load_nps_alias
    n7feat.nps_sido = nps_sido_alias
    return AM


def build_final_panel(alias=False):
    mode = "final"
    if alias:
        install_alias(mode)
    D = n7lib.build_panel(mode)
    ctx = n7feat.load_ctx(mode)
    X = n7feat.build_features(D, ctx, SPEC["feature_bundles_needed"])
    return pd.concat([D, X], axis=1)


def auc_rs(y, p):
    from scipy.stats import rankdata
    y = np.asarray(y).astype(int); r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
