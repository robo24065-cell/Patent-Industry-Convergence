# -*- coding: utf-8 -*-
"""최종시험 위약(final 실행 이후, 모형·점수는 바꾸지 않음).
  python v02_final_placebo.py test 1000   : 시험 레이블을 KSIC 층 안에서 섞고(동결 점수 그대로) 챔피언·M0·SB AUC와 skill을 다시 잰다
                                            → v02_placebo_test.csv. skill 분포가 0 근처여야 하고, 실제 skill이 분포 꼬리에 있어야 한다.
  python v02_final_placebo.py feats       : t0=2022 시험 피처를 봉인 추출본에서 다시 만든다(레이블 없이) → cache/final_test_feats.parquet
  python v02_final_placebo.py train a b   : 학습 레이블을 t0×KSIC 층 안에서 섞고 챔피언 사양으로 재학습, 실제 시험 레이블로 채점
                                            (dev S6과 같은 방식) → v02_placebo_train.csv"""
import sys, os, json
sys.dont_write_bytecode = True
from n8lib import *

SPEC = json.load(open(os.path.join(HERE, "final_spec_r6.json"), encoding="utf-8"))
CH = SPEC["champion"]
P = pd.read_parquet(os.path.join(HERE, "final_preds_r6.parquet"))
y = P.y.values.astype(int)


def sk(a, b):
    return (a - b) / (1 - b)


mode = sys.argv[1]
if mode == "test":
    n = int(sys.argv[2])
    idx = list(P.groupby("ksic").indices.values())
    pc, p0, ps = P.p_champ.values, P.p_M0.values, P.p_SB.values
    rows = []
    for r in range(n):
        rng = np.random.default_rng(5000 + r)
        ys = y.copy()
        for ix in idx:
            ys[ix] = rng.permutation(ys[ix])
        a, b, s = auc(ys, pc), auc(ys, p0), auc(ys, ps)
        rows.append(dict(rep=r, auc_champ=a, auc_M0=b, auc_SB=s, skill_vs_M0=sk(a, b), skill_vs_SB=sk(a, s),
                         top1pct_hits=topk(ys, pc, 0.01)[1]))
    D = pd.DataFrame(rows); D.to_csv(os.path.join(HERE, "v02_placebo_test.csv"), index=False, encoding="utf-8-sig")
    log(D.describe().round(4).to_string())
elif mode == "feats":
    R = extract(SEAL); X = prep(R); del R
    FT = family_tokens(X["pt"]); EXTU = external_unit(); KI = ksic_industry()
    TE = to32(build_t0(2022, X, FT, EXTU, KI, no_label=True)).reset_index(drop=True)
    TE = TE[["rgst_no", "ksic"] + CH["feats"]]
    assert (TE.rgst_no.values == P.rgst_no.values).all()
    TE.to_parquet(f"{CACHE}/final_test_feats.parquet", index=False)
    log("feats", TE.shape)
elif mode == "train":
    a, b = int(sys.argv[2]), int(sys.argv[3])
    TE = pd.read_parquet(f"{CACHE}/final_test_feats.parquet")
    assert (TE.rgst_no.values == P.rgst_no.values).all()
    TR = pd.read_parquet(f"{CACHE}/panel_dev.parquet", columns=["t0", "y", "ksic"] + CH["feats"])
    TR = TR[(TR.t0 >= 2008) & (TR.t0 <= 2020)].reset_index(drop=True)
    gidx = list(TR.groupby(["t0", "ksic"]).indices.values())
    a0, aS = auc(y, P.p_M0.values), auc(y, P.p_SB.values)
    f = os.path.join(HERE, "v02_placebo_train.csv")
    for r in range(a, b):
        rng = np.random.default_rng(9000 + r)
        ys = TR.y.values.astype(int).copy()
        for ix in gidx:
            ys[ix] = rng.permutation(ys[ix])
        rs = np.random.default_rng(CH["neg_seed"])               # 동결 사양과 같은 음성 5% 표본 규칙
        keep = (ys == 1) | (rs.random(len(TR)) < CH["neg_frac"])
        w = np.where(ys == 1, 1.0, 1.0 / CH["neg_frac"])[keep]
        p = fit_pred(TR.loc[keep, CH["feats"]].astype(float), ys[keep], TE[CH["feats"]].astype(float), "xgb", w=w)
        ac = auc(y, p)
        pd.DataFrame([dict(rep=r, auc_champ=ac, skill_vs_M0=sk(ac, a0), skill_vs_SB=sk(ac, aS), top1pct_hits=topk(y, p, 0.01)[1])]).to_csv(
            f, mode="a", header=not os.path.exists(f), index=False, encoding="utf-8-sig")
        log(r, round(ac, 4), round(sk(ac, a0), 4))
