# -*- coding: utf-8 -*-
"""N1 실험 엔진: 원점 내 백분위 순위 피처 → Ridge, 전진 검증(평가 v, 학습 t0 ≤ v−3), 원점별 Spearman ρ, 상위1/3 AUC,
KSIC 군집 부트스트랩, R2/experiments.csv upsert 기록. 채택 규칙: 평균 Δρ ≥ +0.003 & 원점 승률 ≥ 60% (PROTOCOL_R2 §3)."""
import numpy as np, pandas as pd, os, json
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from n1lib import *

P_ALL = pd.read_parquet(f"{CACHE}/panel_n1.parquet")
P = P_ALL[P_ALL.t0.between(2008, 2018) & (P_ALL.t0 != 2010)].copy()     # 주 타깃 개발 원점(결과 ≤2021 < F=2022)
VS = list(range(2011, 2019))
TG = "y_emp"
PATENT_PREFIX = ("mom_", "lvl_", "sh_", "dsh_", "gap_", "frgn_", "krcn", "dkrcn", "krjp", "dkrjp", "lead_", "kab", "cab", "ffam", "dffam",
                 "dom", "ddom", "grant", "dgrant", "cninv", "apmom", "newmom")
def is_patent(c): return c.startswith(PATENT_PREFIX) and not c.startswith("lv_")

def preds(F, alpha=5.0, tg=TG, data=None, train_tg=None, wfun=None, vs=VS, gap=3, model=None, aug=None, fe=None, xf=None, bag=False):
    """F: 피처 목록. train_tg: 학습 타깃(없으면 tg). 학습 원점 t0 ≤ v−gap(결과가 v 이전 완결)."""
    data = P if data is None else data
    trt = train_tg or tg
    if trt not in data.columns and "+" in trt:   # 복합 학습 타깃: 구성 타깃의 원점 내 순위 평균(모두 결측이 아닐 때)
        data = data.copy(); parts = trt.split("+")
        data[trt] = pd.concat([data.groupby("t0")[c].rank(pct=True) for c in parts], axis=1).mean(1, skipna=False)
    need = list(dict.fromkeys(F + [tg, trt]))
    D = data.dropna(subset=[tg]) if trt == tg else data
    Q = rank_within(D, need)
    if xf == "normal":   # 순위 → 정규점수(꼬리 강조)
        from scipy.stats import norm
        for c in need: Q[c] = norm.ppf(Q[c].clip(0.02, 0.98))
    out = {}
    for v in vs:
        te = Q[(Q.t0 == v) & Q.index.isin(data.dropna(subset=[tg]).index)]
        trs = D[(D.t0 <= v-gap)] if trt == tg else data[(data.t0 <= v-gap)]
        trs = trs.dropna(subset=[trt]) if isinstance(trt, str) else trs
        tr = Q.loc[trs.index]
        w = None if wfun is None else wfun(tr.t0.values, v)
        if aug:   # 짧은 시계 레이블 보강: (타깃열, 시계 h) → 결과가 v 이전에 끝나는 원점 t0 = v−h 의 행 추가(학습 타깃 순위로 정렬)
            add_ = []
            for col, h in aug:
                A = data[(data.t0 == v-h)].dropna(subset=[col])
                if len(A) == 0: continue
                QA = rank_within(A, list(dict.fromkeys(F + [col]))); QA[trt] = QA[col]; add_.append(QA)
            if add_:
                tr = pd.concat([tr] + add_, ignore_index=True)
                w = None if wfun is None else wfun(tr.t0.values, v)
        if model is None:
            m = Ridge(alpha=alpha).fit(tr[F], tr[trt], sample_weight=w)
        else:
            m = model().fit(tr[F], tr[trt])
        p = m.predict(te[F])
        if bag and tr.t0.nunique() >= 3:   # 학습 원점 1개씩 빼고 적합한 모형들의 평균(원점 잡음 완화)
            ps = [p]
            for t in tr.t0.unique():
                tt = tr[tr.t0 != t]
                ps.append(Ridge(alpha=alpha).fit(tt[F], tt[trt]).predict(te[F]) if model is None else model().fit(tt[F], tt[trt]).predict(te[F]))
            p = np.mean(ps[1:], axis=0)
        if fe:    # 산업 고정효과(학습 잔차의 KSIC 평균, 수축 n/(n+k)) × λ — 학습 레이블(결과 ≤ v)만 사용
            lam, k = fe
            rs = pd.Series(np.asarray(tr[trt]) - m.predict(tr[F]), index=tr.index)
            g = pd.DataFrame({"ksic": tr.ksic.values, "r": rs.values}).groupby("ksic").r.agg(["mean", "count"])
            adj = (g["mean"] * g["count"] / (g["count"] + k)).reindex(te.ksic.values).fillna(0).values
            p = p + lam * adj
        out[v] =pd.DataFrame({"ksic": te.ksic.values, "p": p, "y": data.loc[te.index, tg].values})
    return out

def b0(tg=TG, mom="g3_emp", vs=VS, data=None):
    data = P if data is None else data
    out = {}
    for v in vs:
        g = data[(data.t0 == v)].dropna(subset=[tg])
        out[v] = pd.DataFrame({"ksic": g.ksic.values, "p": g[mom].rank(pct=True).fillna(0.5).values, "y": g[tg].values})
    return out

def score(o): return pd.Series({v: spearmanr(d.y, d.p)[0] for v, d in o.items()})
def auc(o): return pd.Series({v: roc_auc_score((d.y.rank(pct=True) >= 2/3).astype(int), d.p) for v, d in o.items()})

def boot(o, ref=None, n=1000, seed=0):
    """KSIC 군집 부트스트랩: 원점별 ρ 평균의 CI, ref 가 있으면 Δ(모형−ref) CI."""
    rng = np.random.default_rng(seed); ks = np.array(sorted(set().union(*[set(d.ksic) for d in o.values()])))
    X = {v: d.set_index("ksic") for v, d in o.items()}; R = {v: d.set_index("ksic") for v, d in ref.items()} if ref else None
    bm, bd = [], []
    for _ in range(n):
        s = rng.choice(ks, len(ks), replace=True); a, c = [], []
        for v in o:
            kk = [k for k in s if k in X[v].index]
            x = X[v].loc[kk]; a.append(spearmanr(x.y, x.p)[0])
            if R is not None:
                kr = [k for k in kk if k in R[v].index]; z = R[v].loc[kr]; c.append(spearmanr(z.y, z.p)[0])
        bm.append(np.nanmean(a))
        if R is not None: bd.append(np.nanmean(a) - np.nanmean(c))
    r = {"ci": np.percentile(bm, [2.5, 97.5]).round(3).tolist()}
    if R is not None: r["dci"] = np.percentile(bd, [2.5, 97.5]).round(3).tolist(); r["p_le0"] = float(np.mean(np.array(bd) <= 0))
    return r

_B0 = None
def base_scores():
    global _B0
    if _B0 is None:
        o = b0(); _B0 = (score(o), auc(o), o)
    return _B0

EXP = f"{R2}/experiments.csv"
def log_exp(eid, parent, change, o, champ=None, decision=None, note="", file=EXP, base=None):
    """champ: 챔피언 원점별 ρ(Series). decision 없으면 규칙으로 판정."""
    s, a = score(o), auc(o)
    sb, ab, _ = base if base is not None else base_scores()
    row = {"exp_id": eid, "parent": parent, "change": change}
    for v in s.index: row[f"rho_{v}"] = round(float(s[v]), 4)
    row["mean_rho"] = round(float(s.mean()), 4); row["skill_vs_B0"] = round(float(s.mean() - sb.mean()), 4)
    row["auc_top3"] = round(float(a.mean()), 4); row["auc_skill"] = round(float((a.mean() - ab.mean())/(1 - ab.mean())), 4)
    row["pos_origins"] = int((s > 0).sum())
    if champ is not None:
        d = s - champ.reindex(s.index); row["d_vs_champ"] = round(float(d.mean()), 4); row["win"] = round(float((d > 0).mean()), 3)
        acc = d.mean() >= 0.003 and (d > 0).mean() >= 0.6
        row["decision"] = decision or ("ACCEPT" if acc else "ROLLBACK")
    else:
        row["d_vs_champ"] = np.nan; row["win"] = np.nan; row["decision"] = decision or "BASE"
    row["note"] = note
    L = pd.read_csv(file, encoding="utf-8-sig") if os.path.exists(file) else pd.DataFrame()
    if len(L): L = L[L.exp_id != eid]
    L = pd.concat([L, pd.DataFrame([row])], ignore_index=True); L.to_csv(file, index=False, encoding="utf-8-sig")
    print(f"{eid:6s} {change[:48]:48s} ρ {s.mean():.3f} skill {row['skill_vs_B0']:+.3f} AUC {a.mean():.3f} Δ {row['d_vs_champ']:+.4f} 승 {row['win']} {row['decision']}")
    return row["decision"].startswith("ACCEPT"), s

def champ_state(path=f"{R2}/champion.json"):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None
def save_champ(st, path=f"{R2}/champion.json"):
    json.dump(st, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
