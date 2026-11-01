# -*- coding: utf-8 -*-
"""사후 점검(INFO — 채택 판단에 쓰지 않음, 동결 스펙을 바꾸지 않음). dev 원점 2014–2019만.
출력: robust_checks.csv, robust_desc.csv"""
import json, warnings
import numpy as np, pandas as pd
from n6_lib import *
from n6_exp import M0, M1, M2, BUNDLES, load_panel, score, DEV
warnings.filterwarnings("ignore")

D = load_panel()
st = json.load(open(f"{CACHE}/champion_state.json", encoding="utf-8"))
CF, CS = st["feats"], st["spec"]
champ, pc = score(D, CF, CS)
m0, pm0 = score(D, M0)
rows = []


def add(name, a, b, note=""):
    sk = [skill(a[v], b[v]) for v in DEV]
    rows.append(dict(check=name, **{f"auc_{v}": round(a[v], 4) for v in DEV}, **{f"ref_{v}": round(b[v], 4) for v in DEV},
                     mean_auc=round(float(np.nanmean([a[v] for v in DEV])), 4), mean_ref=round(float(np.nanmean([b[v] for v in DEV])), 4),
                     mean_skill=round(float(np.nanmean(sk)), 4), wins=f"{sum(s > 0 for s in sk)}/{int(np.sum(~np.isnan(sk)))}", note=note))
    log(name, round(float(np.nanmean(sk)), 4))


# 1 강화 기준선
m0e, pm0e = score(D, M0, {"model": "ens", "seeds": [0], "w_xgb": 0.5})
add("챔피언 vs 강화 기준선 M0-앙상블(XGB+로지스틱, 같은 M0 피처)", champ, m0e)
m0l, _ = score(D, M0, {"model": "logit"})
add("챔피언 vs 로지스틱 M0", champ, m0l)
# 2 모형족 교차
cl, _ = score(D, CF, {"model": "logit"}); add("로지스틱 단독(챔피언 피처) vs 로지스틱 M0", cl, m0l)
cx, _ = score(D, CF, {"model": "xgb"}); add("XGB 단독(챔피언 피처) vs XGB M0", cx, m0)
# 3 묶음 하나씩 빼기 (챔피언 대비 손실 = 양수면 그 묶음이 기여)
for nm, bf in [("M1 과제 내용", M1), ("M2 국민연금", M2), ("C02 특허 제목 대응", BUNDLES["C02"][1]), ("C05 NTIS 성과 IP", BUNDLES["C05"][1]),
               ("C10 6T·수명주기", BUNDLES["C10"][1])]:
    a, _ = score(D, [f for f in CF if f not in bf], CS)
    add(f"챔피언 vs '{nm}' 제외", champ, a)
# 4 위약: C02+C05 피처를 원점 안에서 섞기(학습·평가 모두)
rng = np.random.default_rng(7)
Dp = D.copy()
for c in BUNDLES["C02"][1] + BUNDLES["C05"][1]:
    Dp[c] = Dp.groupby("t0")[c].transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
a, _ = score(Dp, CF, CS); add("위약: C02·C05 원점 내 섞기 vs 챔피언(음수여야 정상)", a, champ)
# 5 하위집단(학습은 전체, 평가만 부분)
def sub(mask_fn, name):
    a, b = {}, {}
    for v in DEV:
        te = D[D.t0 == v]; m = mask_fn(te).values
        if m.sum() == 0:
            a[v] = b[v] = np.nan; continue
        a[v] = auc(te.y.values[m], pc[v][m]); b[v] = auc(te.y.values[m], pm0[v][m])
    add(name, a, b, f"평가 표본 비율(유효 원점) {np.nanmean([mask_fn(D[D.t0 == v]).mean() if mask_fn(D[D.t0 == v]).sum() else np.nan for v in DEV]):.3f}")
sub(lambda te: te.linked == 1, "출원인코드 연결 기업만: 챔피언 vs M0")
sub(lambda te: te.o_pat == 0, "NTIS 성과 특허 없는 기업만: 챔피언 vs M0")
sub(lambda te: te.nps_present == 1, "NPS 가입(2015+ 원점) 기업만: 챔피언 vs M0")
sub(lambda te: te.nps_present == 0, "NPS 미가입(2015+ 원점) 기업만: 챔피언 vs M0")
# 6 시드
a, _ = score(D, CF, {**CS, "seeds": [1]}); b, _ = score(D, M0, {"seeds": [1]}); add("시드 1: 챔피언 vs M0", a, b)
# 7 12월 결손 모의: 둘째 해 12월 출원을 레이블에서 빼고 채점(최종시험 2023-12 약 10% 미공개 대비)
nt = load_ntis(2008, 2019); firm = firm_link(nt); af = load_apps(firm, "dev")
yd = {}
for v in DEV:
    lo = (v + 1) * 10000 + 101; hi = (v + 2) * 10000 + 1130
    yd[v] = set(af[(af.appl_date >= lo) & (af.appl_date <= hi)].bz)
a, b = {}, {}
for v in DEV:
    te = D[D.t0 == v]; y2 = te.bz.isin(yd[v]).astype(int).values
    a[v] = auc(y2, pc[v]); b[v] = auc(y2, pm0[v])
add("레이블에서 둘째 해 12월 출원 제외: 챔피언 vs M0", a, b,
    "양성 유지율 " + "/".join(f"{np.mean(D[D.t0 == v].bz.isin(yd[v]))/D[D.t0 == v].y.mean():.3f}" for v in DEV))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/robust_checks.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(R[["check", "mean_auc", "mean_ref", "mean_skill", "wins", "note"]].to_string(index=False))
# 기술통계: 챔피언 점수 10분위별 첫 특허율, 신호별 전환율
desc = []
for v in DEV:
    te = D[D.t0 == v].copy(); te["p"] = pc[v]; te["dec"] = pd.qcut(te.p.rank(method="first"), 10, labels=False) + 1
    for d_, s in te.groupby("dec"):
        desc.append(dict(kind="champ_decile", t0=v, grp=int(d_), n=len(s), rate=round(s.y.mean(), 4)))
    for nm, m in [("o_pat>0", te.o_pat > 0), ("pt_max 상위 20%", te.pt_max >= te.pt_max.quantile(0.8)), ("k50_assigned>0", te.k50_assigned > 0)]:
        desc.append(dict(kind=nm, t0=v, grp=1, n=int(m.sum()), rate=round(te.y[m].mean(), 4)))
        desc.append(dict(kind=nm, t0=v, grp=0, n=int((~m).sum()), rate=round(te.y[~m].mean(), 4)))
E = pd.DataFrame(desc); E.to_csv(f"{OUT}/robust_desc.csv", index=False, encoding="utf-8-sig")
print(E[E.kind != "champ_decile"].pivot_table(index=["kind", "grp"], columns="t0", values="rate").round(3).to_string())
print(E[E.kind == "champ_decile"].pivot_table(index="grp", columns="t0", values="rate").round(3).to_string())
