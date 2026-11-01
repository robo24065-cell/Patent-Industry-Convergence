# -*- coding: utf-8 -*-
"""1) KR 특허(제공) KSIC 라벨 vs 공식 IPC-KSIC 연계표 배정(메인 IPC, 차수 맞춤) 일치율 — KSIC별."""
import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lv_common import *
import duckdb, json

cc = load_crosswalk()
log("연계표 쌍", len(cc), "그룹", cc.ind.nunique(), "출원 합", int(cc.cnt.sum()))
TC = make_lookup(cc, True)
TU = make_lookup(cc, False)

kr = duckdb.sql(f"select appl_no, appl_year, rgst_no, main_ipc, ksic from read_parquet('{META}') where cc='KR'").df()
kr["reg"] = kr.rgst_no.notna() & (kr.rgst_no.astype(str).str.len() > 0)
kr["grp"] = kr.ksic.map(ksic_to_group)
assert kr.grp.notna().all()
log("KR", len(kr), "등록", int(kr.reg.sum()))
a = assign(kr, TC)
u = assign(kr, TU)
kr["cw"] = a.cw_group; kr["conf"] = a.cw_conf; kr["lvl"] = a.cw_level
kr["cwu"] = u.cw_group
kr["k3"] = kr.ksic.str[:3]; kr["k2"] = kr.ksic.str[:2]
kr["cw3"] = kr.cw.map(ksic9_crosswalk_3digit); kr["cw2"] = kr.cw3.str[:2]
kr["ag_g"] = (kr.cw == kr.grp)
kr["ag_u"] = (kr.cwu == kr.grp)
kr["ag3"] = (kr.cw3 == kr.k3)
kr["ag2"] = (kr.cw2 == kr.k2)
# 5자리: 라벨공간 안에서 그룹 내 후보 5자리가 1개인 그룹만 연계표가 5자리를 지목할 수 있다
g2k = kr.groupby("grp").ksic.unique()
uniq = {g: v[0] for g, v in g2k.items() if len(v) == 1}
kr["cw5"] = kr.cw.map(uniq)
kr["ag5"] = (kr.cw5 == kr.ksic)
res = {"n_kr": int(len(kr)), "n_reg": int(kr.reg.sum()), "ipc_hit": float(kr.cw.notna().mean()),
       "ipc_level": kr.lvl.value_counts(normalize=True).round(4).to_dict(),
       "groups_unique5": sorted(uniq.keys()), "n_unique5_ksic": len(uniq)}
log("IPC 매칭률", res["ipc_hit"], res["ipc_level"])

rows = []
for nm, s in (("전체 KR 출원", kr), ("KR 등록(이전 측정과 같은 모집단)", kr[kr.reg]), ("출원 2005–2017(개발 기간)", kr[(kr.appl_year >= 2005) & (kr.appl_year <= DEV_END)])):
    s = s[s.cw.notna()]
    for lab, a_, b_ in (("연계표그룹(27, 라벨공간 제약)", "cw", "grp"), ("연계표그룹(61, 제약 없음)", "cwu", "grp"),
                        ("3자리(25)", "cw3", "k3"), ("2자리(13)", "cw2", "k2")):
        k, po, pe, n = kappa(s[a_], s[b_])
        rows.append({"모집단": nm, "해상도": lab, "n": n, "일치율": round(po, 4), "우연기대": round(pe, 4), "kappa": round(k, 4)})
    t = s[s.cw5.notna() | s.grp.isin(uniq)]
    t = t[t.grp.isin(uniq)]
    rows.append({"모집단": nm, "해상도": f"5자리(연계표가 5자리를 지목 가능한 {len(uniq)}개 KSIC 한정)", "n": len(t),
                 "일치율": round(float(t.ag5.mean()), 4), "우연기대": None, "kappa": None})
    rows.append({"모집단": nm, "해상도": "5자리(50개 전체, 그룹 내 후보 여럿이면 연계표는 지목 불가=불일치)", "n": len(s),
                 "일치율": round(float(s.ag5.mean()), 4), "우연기대": None, "kappa": None})
O = pd.DataFrame(rows)
O.to_csv(os.path.join(OUT, "lv1_overall_agreement.csv"), index=False, encoding="utf-8-sig")
print(O.to_string())

# 확신도 구간
s = kr[kr.cw.notna()].copy()
s["conf_bin"] = pd.cut(s.conf, [0, .25, .5, .75, .9, 1.0001], labels=["~25%", "25–50%", "50–75%", "75–90%", "90%+"], right=False)
C = s.groupby("conf_bin", observed=True).agg(n=("ag_g", "size"), 일치율=("ag_g", "mean")).reset_index()
C.to_csv(os.path.join(OUT, "lv1_by_confidence.csv"), index=False, encoding="utf-8-sig")
print(C.to_string())

# KSIC별 (재현율: 그 KSIC 특허 중 연계표가 같은 그룹을 지목한 비율) + 그룹별 정밀도
defs = json.load(open(os.path.join(BASE, r"공모전데이터\외부데이터\KSIC해설\ksic50_definitions.json"), encoding="utf-8"))
NM = {r["code"]: r["codenmKr"] for r in defs}
def nm_of(k):
    return NM.get(k, "")
prec = s.groupby("cw").ag_g.mean()
rows = []
for k, g in s.groupby("ksic"):
    wrong = g[~g.ag_g].cwu.fillna("미매칭").value_counts()
    rows.append({"ksic": k, "산업명": nm_of(k), "연계표그룹": g.grp.iloc[0], "그룹내_라벨후보수": len(g2k[g.grp.iloc[0]]),
                 "n_KR": len(g), "n_등록": int(g.reg.sum()),
                 "그룹일치율": round(float(g.ag_g.mean()), 4), "그룹일치율_제약없음": round(float(g.ag_u.mean()), 4),
                 "3자리일치율": round(float(g.ag3.mean()), 4), "2자리일치율": round(float(g.ag2.mean()), 4),
                 "5자리일치율(지목가능시)": round(float(g.ag5.mean()), 4) if g.grp.iloc[0] in uniq else None,
                 "그룹정밀도(연계표가_이그룹지목→실제라벨일치)": round(float(prec.get(g.grp.iloc[0], np.nan)), 4),
                 "평균확신도": round(float(g.conf.mean()), 4),
                 "주요_오배정그룹": "; ".join(f"{i}:{c}" for i, c in wrong.head(3).items())})
K = pd.DataFrame(rows).sort_values("그룹일치율")
K.to_csv(os.path.join(OUT, "lv1_agreement_by_ksic.csv"), index=False, encoding="utf-8-sig")
print(K[["ksic", "연계표그룹", "n_KR", "그룹일치율", "그룹일치율_제약없음", "3자리일치율", "5자리일치율(지목가능시)", "주요_오배정그룹"]].to_string())
res["ksic_group_agree_min_med_max"] = [float(K.그룹일치율.min()), float(K.그룹일치율.median()), float(K.그룹일치율.max())]
json.dump(res, open(os.path.join(OUT, "lv1_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
# 다음 단계용 캐시(특허별 연계표 배정)
kr[["appl_no", "appl_year", "reg", "ksic", "grp", "cw", "cwu", "conf"]].to_parquet(os.path.join(OUT, "cache_kr_cw.parquet"), index=False)
log("done")
