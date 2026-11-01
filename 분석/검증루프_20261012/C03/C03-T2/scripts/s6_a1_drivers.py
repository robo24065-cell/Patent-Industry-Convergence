# -*- coding: utf-8 -*-
"""s6 (사후 진단, 판정 외): A1 변화의 원인 분해 — 어떤 구 코드가 어느 KSIC로 옮겨졌나, n:n 균등 분할의 '포괄 코드 번짐' 여부,
그리고 사후 변형 A1c(코드 연속성 우선: 같은 코드가 HS2017에 이어지면 그 코드로 100%, 폐지 코드만 대상에 균등 분할)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s5_sensitivity as S5  # 재사용(실행 시 s5 결과 재생성됨 — 동일 시드라 같은 값)
from lib import *

c12, c07, codes = S5.c12, S5.c07, S5.codes


def old_to_new_cont(corr):
    t = corr.copy(); t["nT"] = t.groupby("old").new.transform("nunique")
    t["cont"] = t.groupby("old").apply(lambda g: g.name in set(g.new), include_groups=False).reindex(t.old).values
    a = t[t.cont & (t.new == t.old)].assign(f=1.0)
    b = t[~t.cont].assign(f=lambda d: 1 / d.nT)
    z = pd.concat([a, b]); z = z[z.new.isin(codes)]
    return z[["old", "new", "f"]]


orig_o2n = S5.old_to_new


def o2n_patch(corr, mode):
    return old_to_new_cont(corr) if mode == "cont" else orig_o2n(corr, mode)


S5.old_to_new = o2n_patch
o, q, p2, ek = S5.run("A1c_hs_align_continuity(사후)", "cont", "equal", False)
o3, q3, p23, ek3 = S5.run("A1c+A3(사후)", "cont", "equal", True)
print({k: o[k] for k in ["SS", "SW", "WS", "WW", "rho", "rho_lo", "rho_hi", "rho_chg", "rho_chg_lo", "rho_chg_hi", "list42", "SW_list"]})
print({k: o3[k] for k in ["SS", "SW", "WS", "WW", "rho", "rho_lo", "rho_hi", "rho_chg", "list42", "SW_list"]})
# 원인 분해: KR·5개국 P2 KSIC 값 원본 vs A1 vs A1c
b = S5.EK["B_original"]; a1 = S5.EK["A1_hs_align_equal"]
def p2v(e):
    return e[e.year.isin([2015, 2016, 2017])].groupby(["k5", "rep"]).v.sum().unstack()
vb, va, vc = p2v(b), p2v(a1), p2v(ek)
rb, ra, rc = S5.P2["B_original"], S5.P2["A1_hs_align_equal"], p2
D = pd.DataFrame({"rca_orig": rb, "rca_A1": ra, "rca_A1c": rc, "KR_ratio_A1": va.KR / vb.KR, "KR_ratio_A1c": vc.KR / vb.KR,
                  "others_ratio_A1": va.drop(columns="KR").sum(1) / vb.drop(columns="KR").sum(1),
                  "others_ratio_A1c": vc.drop(columns="KR").sum(1) / vb.drop(columns="KR").sum(1)})
D["rank_orig"] = rb.rank(); D["rank_A1"] = ra.rank(); D["rank_A1c"] = rc.rank(); D["drank_A1"] = D.rank_A1 - D.rank_orig
D["rta"] = S5.rta["P2"]
D["q_orig"] = S5.Q["B_original"]; D["q_A1"] = S5.Q["A1_hs_align_equal"]; D["q_A1c"] = q
D = D.round(4)
D.to_csv(os.path.join(CACHE, "s6_a1_drivers.csv"), encoding="utf-8-sig")
print(D.reindex(D.drank_A1.abs().sort_values(ascending=False).index).head(15).to_string())
# 반도체 IC 코드 n:n 진단
for tcode in ["854231", "854232", "854233", "854239", "851712", "851770", "852852", "870193", "300490"]:
    g = c12[c12.new == tcode]
    print(tcode, g[["old", "rel", "exflag"]].values.tolist(), "| old의 대상 수:", {o: int((c12.old == o).sum()) for o in g.old})
# 보충 수집 구 코드별 KR 2015–16 금액 상위와 대상
dl = S5.dl
add = dl[dl.key.str.startswith("OLDH4") & (dl.rep == "KR")].groupby("hs6").v.sum().sort_values(ascending=False).head(15)
print("KR 보충 H4 코드 상위:", [(h, round(v / 1e9, 2), sorted(c12[c12.old == h].new)) for h, v in add.items()])
# 원 수집 H4 연도 코드 중 n:n·n:1 포괄 코드의 KR 금액(번짐 후보)
ex = S5.ex
e = ex[(ex.year.isin([2015, 2016])) & (ex.rep == "KR")]
nT = c12.groupby("old").new.nunique()
e = e.assign(nT=e.hs6.map(nT)).sort_values("export_usd", ascending=False)
print("KR H4 원 수집 중 대상 2개 이상 구 코드 상위:", e[e.nT > 1].groupby("hs6").export_usd.sum().sort_values(ascending=False).head(10).round(-6).to_dict())
res = dict(A1c=o, A1c_A3=o3)
json.dump(res, open(os.path.join(CACHE, "s6_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
