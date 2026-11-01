# -*- coding: utf-8 -*-
"""사후 점검(채택 판단에 쓰지 않음, dev 원점만). 최종 챔피언 R03 기준.
1) 원점별 AUC·skill·군집 부트스트랩 CI·상위 k 적중률(M0·SB·챔피언)  2) 같은 모형족 기준선(M0-ens, SB-ens)
3) 모형족 분해(XGB만, 로지스틱만)  4) 묶음 하나씩 빼기  5) 다른 시드(3,4,5)  6) 1년 결과창(사전등록 보조)
7) NPS 미가입 하위집단  8) 위약: 학습 레이블을 원점 안에서 섞기"""
import os, json, numpy as np, pandas as pd, warnings
from n7lib import *
from n7feat import BUNDLES
warnings.filterwarnings("ignore")
D = pd.read_parquet(os.path.join(CACHE, "dev_panel.parquet"))
D = pd.concat([D, pd.read_parquet(os.path.join(CACHE, "feat_dev.parquet"))], axis=1)
assert D.t0.max() + 2 < F_YEAR
V = DEV_ORIGINS; S3 = (0, 1, 2)
st = json.load(open(os.path.join(OUT, "champ.json"), encoding="utf-8"))
CF = st["champ"]["feats"]; CM = st["champ"]["model"]
PILOT = M0 + M1_ADD + M2_ADD + M3_ADD


def sc(feats, model, seeds=S3, ycol="y", mask=None, perm=False):
    P = {}
    for v in V:
        tr = D[(D.t0 >= TRAIN_MIN_T0) & (D.t0 <= v - 2)]; te = D[D.t0 == v]
        ytr = tr[ycol].values.copy()
        if perm:
            rng = np.random.default_rng(v)
            for t in tr.t0.unique():
                m = (tr.t0 == t).values; ytr[m] = rng.permutation(ytr[m])
        P[v] = predict(tr[feats].astype(float), ytr, te[feats].astype(float), model=model, seeds=seeds)
    return P


def aucs(P, ycol="y", mask=None):
    out = {}
    for v in V:
        te = D[D.t0 == v]; m = np.ones(len(te), bool) if mask is None else mask(te).values
        out[v] = auc(te[ycol].values[m], P[v][m])
    return out


def cmp(name, a, b):
    sk = [skill(a[v], b[v]) for v in V]
    return dict(check=name, **{f"auc_{v}": round(a[v], 4) for v in V}, **{f"ref_{v}": round(b[v], 4) for v in V},
                mean_skill=round(float(np.mean(sk)), 4), wins=f"{sum(s > 0 for s in sk)}/{len(V)}")


Pc = sc(CF, CM); P0 = sc(M0, "xgb"); Psb = sc(STRONG, "xgb"); Pp = sc(PILOT, "xgb")
P0e = sc(M0, "ens"); Psbe = sc(STRONG, "ens"); Ppe = sc(PILOT, "ens")
ac, a0, asb, ap = aucs(Pc), aucs(P0), aucs(Psb), aucs(Pp)
a0e, asbe, ape = aucs(P0e), aucs(Psbe), aucs(Ppe)
# 1) 원점별 표
rows = []
for v in V:
    te = D[D.t0 == v]; y = te.y.values
    ci = cluster_boot(y, Pc[v], P0[v], te.bizno.values, n=300, seed=v)
    cs = cluster_boot(y, Pc[v], Psb[v], te.bizno.values, n=300, seed=v + 1)
    r = dict(t0=v, n=len(te), pos=int(y.sum()), base_rate=round(y.mean(), 4), auc_M0=round(a0[v], 4), auc_SB=round(asb[v], 4),
             auc_pilot=round(ap[v], 4), auc_champ=round(ac[v], 4), skill_vs_M0=round(skill(ac[v], a0[v]), 4),
             ci_lo=round(ci["skill_ci"][0], 4), ci_hi=round(ci["skill_ci"][1], 4), skill_vs_SB=round(skill(ac[v], asb[v]), 4),
             sb_ci_lo=round(cs["skill_ci"][0], 4), sb_ci_hi=round(cs["skill_ci"][1], 4),
             auc_champ_ci=f"[{ci['auc_model_ci'][0]:.3f}, {ci['auc_model_ci'][1]:.3f}]")
    for fr, nm in [(0.01, "top1"), (0.05, "top5"), (0.10, "top10")]:
        r[f"{nm}_champ"] = round(topk(y, Pc[v], fr)[0], 4); r[f"{nm}_M0"] = round(topk(y, P0[v], fr)[0], 4); r[f"{nm}_SB"] = round(topk(y, Psb[v], fr)[0], 4)
    r["rec10_champ"] = round(topk(y, Pc[v], 0.10)[1], 4); r["rec10_M0"] = round(topk(y, P0[v], 0.10)[1], 4)
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "posthoc_champ_by_origin.csv"), index=False, encoding="utf-8-sig")
print(T.to_string(index=False))
out = [cmp("챔피언 vs M0(XGB) — 주 지표", ac, a0), cmp("챔피언 vs 강한 기준선 SB(XGB)", ac, asb), cmp("챔피언 vs 파일럿 챔피언(XGB 3시드)", ac, ap),
       cmp("챔피언 vs M0-ens(같은 모형족)", ac, a0e), cmp("챔피언 vs SB-ens(같은 모형족)", ac, asbe), cmp("챔피언 vs 파일럿 피처-ens", ac, ape),
       cmp("M0-ens vs M0(XGB)", a0e, a0), cmp("SB-ens vs SB(XGB)", asbe, asb)]
# 3) 모형족 분해
ax = aucs(sc(CF, "xgb")); al = aucs(sc(CF, "logit"))
out += [cmp("챔피언 피처 XGB만 vs M0", ax, a0), cmp("챔피언 피처 로지스틱만 vs M0", al, a0), cmp("챔피언(ens) vs 로지스틱만", ac, al), cmp("챔피언(ens) vs XGB만", ac, ax)]
# 4) 묶음 하나씩 빼기
groups = {"M1 특허 행태": M1_ADD, "M2 NTIS 참여": M2_ADD, "M3 KSIC 연구비": M3_ADD, "V06 동종 진입률": ["peer_entry_k"],
          "C10 국민연금 확장": BUNDLES["C10_nps_ext"], "D01 발명자 규모": BUNDLES["D01_inventor_size"],
          "R01 지역 진입률": BUNDLES["C09_peer_entry_region"], "R03 납부 지연": BUNDLES["C01_pay_delay"]}
for g, fs in groups.items():
    keep = [f for f in CF if f not in fs]
    out.append(cmp(f"챔피언 vs {g} 제거", ac, aucs(sc(keep, CM))))
# 5) 다른 시드
out.append(cmp("시드 3·4·5: 챔피언 vs 파일럿 챔피언(XGB)", aucs(sc(CF, CM, seeds=(3, 4, 5))), aucs(sc(PILOT, "xgb", seeds=(3, 4, 5)))))
out.append(cmp("시드 3·4·5: 챔피언 vs M0(XGB)", aucs(sc(CF, CM, seeds=(3, 4, 5))), aucs(sc(M0, "xgb", seeds=(3, 4, 5)))))
# 6) 1년 결과창(t0+1만)
a1c = aucs(sc(CF, CM, ycol="y1"), ycol="y1"); a10 = aucs(sc(M0, "xgb", ycol="y1"), ycol="y1"); a1s = aucs(sc(STRONG, "xgb", ycol="y1"), ycol="y1")
out += [cmp("1년 결과창: 챔피언 vs M0", a1c, a10), cmp("1년 결과창: 챔피언 vs SB", a1c, a1s)]
# 7) NPS 미가입
m = lambda te: te.nps_present == 0
out.append(cmp("NPS 미가입 기업만: 챔피언 vs M0", aucs(Pc, mask=m), aucs(P0, mask=m)))
# 8) 위약
out.append(cmp("위약: 학습 레이블 원점 안 섞기(챔피언 명세) vs M0", aucs(sc(CF, CM, perm=True)), a0))
O = pd.DataFrame(out); O.to_csv(os.path.join(OUT, "posthoc_checks.csv"), index=False, encoding="utf-8-sig")
print(O.to_string(index=False))
