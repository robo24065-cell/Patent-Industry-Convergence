# -*- coding: utf-8 -*-
"""s5: NPS biz6|적용일(ba) 별칭 키 유일성 감사. 현행(loose) 규칙 재현 → 엄격(strict) 조건별 탈락 수, 상호 유사도,
N4·N7 별칭 보정 동결 예측(preds_alias_*) 재채점(탈락 별칭 사용 시험 기업 제외), T1·R12(2019-12) 별칭 점검. 재학습 없음."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__)); sys.stdout.reconfigure(encoding="utf-8")
from lib5 import *
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
ALL = ["201512", "201612", "201712", "201812", "201912", "202012", "202112", "202212", "202312", "202412", "202512", "202608"]
lk = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet")); lk = lk[lk.nps_hit]
A2K = dict(zip(lk.applicantCd, lk.biz6 + "|" + lk.nname)); Kset = set(A2K.values())
raw, ident, shared = {}, [], []
for s in ALL:
    d = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{s}.parquet"), columns=["name", "biz6", "adpt_dt", "n_emp", "indnm"])
    d["key"] = d.biz6.fillna("") + "|" + d.name.map(norm)
    d["ba"] = d.biz6.fillna("") + "|" + d.adpt_dt.fillna("").astype(str)
    d["has_adpt"] = d.adpt_dt.notna()
    raw[s] = d[["key", "ba", "has_adpt", "n_emp", "indnm"]]
    ident.append(d.loc[d.key.isin(Kset) & d.adpt_dt.notna(), ["ba", "key"]])
    nk = d[d.has_adpt].groupby("ba").key.nunique()
    shared.append(nk.rename(s))
    log(s, "rows", len(d), "ba 유일 행 비율", round(float((d.ba.map(nk) == 1).mean()), 3))
ident = pd.concat(ident).drop_duplicates()
nK = ident.groupby("ba").key.nunique()
ident = ident[ident.ba.map(nK) == 1].set_index("ba").key
SH = pd.concat(shared, axis=1)            # ba × 스냅샷: 그 스냅샷에서 ba를 쓰는 키 수
ba_max = SH.max(axis=1)                    # 어느 스냅샷에서든 동시에 쓰인 최대 키 수
# ba 유일성 통계(정체성 사전 ba 기준)
R = {"identity_pairs": int(len(ident)),
     "identity_ba_ever_shared_share": float((ident.index.map(ba_max) > 1).mean())}
log("ident", len(ident), "ever shared", R["identity_ba_ever_shared_share"])


def dice(a, b):
    a = a.split("|", 1)[-1]; b = b.split("|", 1)[-1]
    A = {a[i:i + 2] for i in range(len(a) - 1)} or {a}; B = {b[i:i + 2] for i in range(len(b) - 1)} or {b}
    return 2 * len(A & B) / (len(A) + len(B))


def alias_loose_strict(snaps, have_from="raw"):
    rows = []
    for snap in snaps:
        d = raw[snap]
        have = set(d.key)
        c = d[d.ba.isin(ident.index)].copy()
        c["K"] = c.ba.map(ident)
        c = c[(c.key != c.K) & ~c.K.isin(have) & c.key.isin(have)]
        g = c.groupby(["K", "key"]).agg(cnt=("ba", "size"), ba=("ba", "first")).reset_index()
        ncand = g.groupby("K").key.nunique()
        al = g.sort_values("cnt", ascending=False).drop_duplicates("K").copy()   # 현행 규칙(loose)
        al["snap"] = snap
        al["n_cand"] = al.K.map(ncand)
        al["shared_in_s"] = al.ba.map(SH[snap]).fillna(1) > 1
        al["shared_ever"] = al.ba.map(ba_max).fillna(1) > 1
        al["strict"] = (~al.shared_in_s) & (~al.shared_ever) & (al.n_cand == 1)
        al["dice"] = [dice(a, b) for a, b in zip(al.K, al.key)]
        rows.append(al)
    return pd.concat(rows, ignore_index=True)


SN = {s.replace("-", "") for s in []}
A4 = alias_loose_strict(["201512", "201612", "201712", "201812", "201912", "202012", "202112", "202212", "202312"])
A4["snap_d"] = A4.snap.str[:4] + "-" + A4.snap.str[4:]
# 재현 대조: N4 v6_alias_stats.json, N7 alias_map_final.parquet
v6 = json.load(open(os.path.join(N4, "VERIFY_R5", "v6_alias_stats.json")))
R["N4_repro_counts"] = {k: [int((A4.snap_d == k).sum()), v] for k, v in v6.items()}
am7 = pd.read_parquet(os.path.join(N7, "VERIFY_R6", "cache", "alias_map_final.parquet"))
mine7 = A4[A4.snap <= "202212"][["snap_d", "K", "key"]].rename(columns={"snap_d": "snap"})
mm = mine7.merge(am7, on=["snap", "K"], how="outer", suffixes=("_mine", "_n7"), indicator=True)
R["N7_repro"] = dict(mine=len(mine7), n7=len(am7), both=int((mm._merge == "both").sum()),
                     same_key=int(((mm._merge == "both") & (mm.key_mine == mm.key_n7)).sum()))
log("repro", R["N4_repro_counts"], R["N7_repro"])
# 스냅샷별 loose/strict
tab = A4.groupby("snap").agg(loose=("K", "size"), strict=("strict", "sum"), shared_in_s=("shared_in_s", "sum"),
                             shared_ever=("shared_ever", "sum"), multi_cand=("n_cand", lambda s: int((s > 1).sum())),
                             dice_med=("dice", "median")).reset_index()
R["by_snap"] = tab.to_dict("records")
R["dice_strict_median"] = float(A4[A4.strict].dice.median()); R["dice_dropped_median"] = float(A4[~A4.strict].dice.median())
R["dice_strict_lt02"] = float((A4[A4.strict].dice < 0.2).mean()); R["dice_dropped_lt02"] = float((A4[~A4.strict].dice < 0.2).mean())
R["dropped_total"] = int((~A4.strict).sum()); R["loose_total"] = int(len(A4))
log("by snap\n", tab.to_string())
A4.to_parquet(os.path.join(CACHE, "alias_loose_strict.parquet"), index=False)
# 공식 패널이 별칭을 쓰는가: 공식 final_test_pred의 nps_present와 동결 캐시 키 비교는 하지 않고 코드로 확인(별칭 코드는 VERIFY 폴더에만 있음)
# ---- 별칭 보정 동결 예측 재채점
RS = []
K2A = {}
for a, k_ in A2K.items():
    K2A.setdefault(k_, []).append(a)
def bad_firms(t0, snaps_used):
    bad = A4[(~A4.strict) & A4.snap.isin(snaps_used)]
    return {a for K in bad.K for a in K2A.get(K, [])}
def used_firms(t0, snaps_used):
    al = A4[A4.snap.isin(snaps_used)]
    return {a for K in al.K for a in K2A.get(K, [])}
cases = [("N4", os.path.join(N4, "VERIFY_R5", "cache", "preds_alias_main.parquet"), "p_champ", "p_M0", 2023, 0.140),
         ("N7", os.path.join(N7, "VERIFY_R6", "cache", "preds_alias.parquet"), "p_champ", "p_M0", 2022, 0.251)]
for tr, fp, pc, p0, t0, rep in cases:
    D = pd.read_parquet(fp); D["applicantCd"] = D.applicantCd.astype(str)
    if "bizno" not in D:
        bz = pd.read_parquet(os.path.join(N4, "final_test_pred_r4.parquet"), columns=["applicantCd", "bizno"]); bz["applicantCd"] = bz.applicantCd.astype(str)
        D = D.merge(bz, on="applicantCd", how="left")
    s_t = [f"{t0}12", f"{t0 - 1}12"]; s_any = [s for s in ALL if s <= f"{t0}12"]
    base = score(D.y.values, D[pc].values, D[p0].values, D.bizno.values, None, NB)
    base.update(track=tr, stratum=f"별칭 보정 동결 예측(재현, 원고 {rep})", d_skill=0.0); RS.append(base)
    for lab, sn in [("t0·t0−1 스냅샷", s_t), ("t0 이전 모든 스냅샷(민감도)", s_any)]:
        bad = bad_firms(t0, sn); used = used_firms(t0, sn)
        m = ~D.applicantCd.isin(bad)
        r = score(D.y.values[m], D[pc].values[m], D[p0].values[m], D.bizno.values[m], None, NB)
        r.update(track=tr, stratum=f"엄격 규칙 탈락 별칭 사용 기업 제외({lab})", d_skill=r["skill"] - base["skill"],
                 note=f"별칭 사용 시험 기업 {int(D.applicantCd.isin(used).sum())}, 탈락 별칭 사용 {int((~m).sum())}(양성 {int(D.y[~m].sum())})")
        RS.append(r)
        log(tr, lab, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("n", "pos", "skill", "skill_ci", "d_skill", "note")})
    R[f"{tr}_alias_repro_skill"] = base["skill"]
# ---- T1 TXT_R9 s1_firm / R12 r12_reg (2019-12) 규칙: K 없고 같은 ba면 모든 행 추가 후 가입자 최대 행 선택
W = raw["201912"]; have = set(W.key)
al = W[W.ba.isin(ident.index)].copy(); al["K"] = al.ba.map(ident)
al = al[(al.key != al.K) & ~al.K.isin(have)]
ncand = al.groupby("K").key.nunique()
ch = al.sort_values("n_emp", ascending=False).drop_duplicates("K").copy()
ch["n_cand"] = ch.K.map(ncand); ch["shared_in_s"] = ch.ba.map(SH["201912"]).fillna(1) > 1; ch["shared_ever"] = ch.ba.map(ba_max).fillna(1) > 1
ch["strict"] = (~ch.shared_in_s) & (~ch.shared_ever) & (ch.n_cand == 1)
ch[["K", "key", "ba", "n_cand", "shared_in_s", "shared_ever", "strict"]].to_parquet(os.path.join(CACHE, "r12_alias_2019.parquet"), index=False)
R["T1R12_2019_alias_K"] = int(len(ch)); R["T1R12_2019_alias_rows"] = int(len(al)); R["T1R12_2019_dropped"] = int((~ch.strict).sum())
R["T1R12_2019_multi_cand"] = int((ch.n_cand > 1).sum())
# 다른 키를 쓰는 사업장이 끼어들어 업종명이 달라지는 경우(모호 후보끼리 업종명 불일치)
amb = al[al.K.isin(ch[ch.n_cand > 1].K)]
R["T1R12_2019_multi_cand_indnm_conflict"] = int(amb.groupby("K").indnm.nunique().gt(1).sum())
reg12 = pd.read_parquet(os.path.join(MC, r"_shared_report\label_origin_R10\confirm_R12\cache\reg12.parquet"))
reg12["applicantCd"] = reg12.applicantCd.astype(str)
npsfirms = reg12[reg12.nps.notna()]
aliasK_bad = set(ch[~ch.strict].K); aliasK = set(ch.K)
R["R12_nps_firms"] = int(len(npsfirms))
R["R12_nps_firms_via_alias"] = int(npsfirms.applicantCd.map(A2K).isin(aliasK).sum())
R["R12_nps_firms_via_dropped_alias"] = int(npsfirms.applicantCd.map(A2K).isin(aliasK_bad).sum())
log("T1/R12", {k: v for k, v in R.items() if k.startswith(("T1R12", "R12"))})
pd.DataFrame(RS).to_csv(os.path.join(CACHE, "rescore_alias.csv"), index=False, encoding="utf-8-sig")
json.dump(R, open(os.path.join(CACHE, "s5_alias.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
log("done")
