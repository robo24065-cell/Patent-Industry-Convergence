# -*- coding: utf-8 -*-
"""N2 공용 라이브러리: 특허 KSIC 브리지 배분 성장(gd) 생성, 전진검증 평가, 챔피언/도전자 기록.
개발 원점 t=2016..2023, 학습 2014 ≤ t' ≤ t−1. 특허 가중은 T=t−1년 말, DART는 회계연도 t 보고서(실시간판) 이하.
🔒 2024 조사값은 이 라이브러리에서 절대 로드하지 않는다(panel_base.parquet 는 ≤2023으로 생성)."""
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge, QuantileRegressor
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "cache"); R2 = os.path.join(HERE, "R2")
os.makedirs(R2, exist_ok=True)
DEV = list(range(2016, 2024))
TRAIN_START = 2014
_cache = {}
PANEL_OVERRIDE = None     # final_run_r2.py(final/future) 전용


def load(name):
    if name not in _cache:
        _cache[name] = pd.read_parquet(os.path.join(C, name))
    return _cache[name]


def panel():
    if PANEL_OVERRIDE is not None:
        return PANEL_OVERRIDE.copy()
    Y = load("panel_base.parquet").copy()
    assert Y.t.max() <= 2023
    return Y


def firm_growth(src="rt", clip=1.0):
    """기업×연도 매출 성장. src='rt' 실시간판(t년 보고서의 당기/전기), 'first' 파일럿판(최초 보고값 연결)."""
    if src in ("rt", "rt2"):
        F = load("fin_rt.parquet" if src == "rt" else "fin_rt2.parquet")
        fr = F[["corp_code", "year", "rev", "rev_prev", "assets", "as_prev", "op", "op_prev"]].copy()
    else:
        F = load("fin_first.parquet")[["corp_code", "year", "rev", "assets", "op"]]
        fr = F.merge(F.assign(year=F.year + 1).rename(columns={"rev": "rev_prev", "assets": "as_prev", "op": "op_prev"}),
                     on=["corp_code", "year"])
    fr = fr[(fr.rev > 0) & (fr.rev_prev > 0)].copy()
    fr["gi"] = np.log(fr.rev / fr.rev_prev).clip(-clip, clip)
    return fr.rename(columns={"year": "t"})


def weights(kind="pit"):
    if kind in ("pit", "pit_share"):
        W = load("alloc_pit.parquet")
        W = W.assign(w=W.w_pit if kind == "pit" else W.w_pit_share)
    elif kind in ("pit_rec", "pit_rec3", "pit_rec8"):
        W = load("alloc_pit.parquet"); W = W.assign(w=W["w_" + kind])
    elif kind in ("snap", "snap_share"):
        W = load("alloc_snap.parquet")
        W = W.assign(w=W.w_snap if kind == "snap" else W.w_snap_share)
    elif kind == "pilot":                   # 파일럿 f30 산출물 그대로(스냅샷 분모 N_full) — 재현 확인용
        W = load("alloc_pilot_snapshot.parquet")
        W = W.assign(w=W.w_full)
    else:
        raise ValueError(kind)
    if "n_k_rec8" in W.columns:
        return W[["corp_code", "t0", "ksic", "w", "n_k", "n_k_rec8"]]
    return W[["corp_code", "t0", "ksic", "w", "n_k"]]


def gd_table(spec, tag="gd"):
    """배분 성장 gd(k,t) = log Σ a_i exp(gi) / Σ a_i, a_i = w(i,k,t−1)·rev_prev_i."""
    fr = firm_growth(spec.get("fin", "rt"), spec.get("clip_gi", 1.0))
    W = weights(spec.get("w", "pit"))
    M = fr.merge(W, left_on=["corp_code", fr.t - 1], right_on=["corp_code", "t0"])
    w = M.w.values.copy()
    c = spec.get("conf_c")
    if c:                                    # 라벨 신뢰도 수축: 해당 KSIC 보유 건수가 적으면 가중 축소
        nk_ = M[spec.get("conf_on", "n_k")]
        w = w * nk_ / (nk_ + c)
    M["a"] = w * M.rev_prev
    if spec.get("size_pow") is not None:     # 규모 완화: a = w·rev_prev^p
        M["a"] = w * M.rev_prev ** spec["size_pow"]
    cap = spec.get("cap")
    if cap:                                  # 한 기업이 KSIC-연도 배분 기반의 cap 비중을 넘지 못하게(대기업 번들 완화)
        for _ in range(3):
            tot = M.groupby(["ksic", "t"]).a.transform("sum")
            M["a"] = np.minimum(M.a, cap * tot)
    M["ae"] = M.a * np.exp(M.gi)
    G = M.groupby(["ksic", "t"])
    D = pd.DataFrame({tag: np.log(G.ae.sum() / G.a.sum()), f"nf_{tag}": G.size(), f"cov_{tag}": G.a.sum()})
    if spec.get("assets"):
        Ma = M[(M.assets > 0) & (M.as_prev > 0)].copy()
        wa = Ma.w * (Ma.n_k / (Ma.n_k + c) if c else 1)
        Ma["x1"] = wa * Ma.assets; Ma["x0"] = wa * Ma.as_prev
        G2 = Ma.groupby(["ksic", "t"])
        D[f"{tag}_assets"] = np.log(G2.x1.sum() / G2.x0.sum())
    return D.reset_index()


def induty_table(spec):
    """대조(E11): DART 업종코드(스냅샷) 5자리 = KSIC 인 기업 매출을 전액 그 산업에 배분."""
    fr = firm_growth(spec.get("fin", "rt"), spec.get("clip_gi", 1.0))
    dart = load("dart_firms.parquet")
    fi = fr.merge(dart[["corp_code", "induty_code"]], on="corp_code")
    fi = fi[fi.induty_code.str.len() == 5].copy()
    fi["ae"] = fi.rev_prev * np.exp(fi.gi)
    G = fi.groupby(["induty_code", "t"])
    D = pd.DataFrame({"gd_induty": np.log(G.ae.sum() / G.rev_prev.sum()), "nf_induty": G.size()}).reset_index()
    return D.rename(columns={"induty_code": "ksic"})


def build(spec):
    Y = panel()
    D = gd_table(spec)
    Y = Y.merge(D, on=["ksic", "t"], how="left")
    Y["nf_gd"] = Y.nf_gd.fillna(0)
    fb = spec.get("fill", "g_idx")
    Y["gd_f"] = Y.gd.fillna(Y[fb]) if fb else Y.gd
    if spec.get("assets"):
        Y["gd_assets_f"] = Y.gd_assets.fillna(Y[fb])
    if spec.get("shrink_k"):
        k = spec["shrink_k"]
        Y["gd_f"] = (Y.gd_f * Y.nf_gd + Y[fb] * k) / (Y.nf_gd + k)
    if spec.get("gd2"):                       # 두 번째 배분 성장(다른 가중)
        D2 = gd_table({**spec, "w": spec["gd2"], "assets": False}, tag="gd2")
        Y = Y.merge(D2[["ksic", "t", "gd2"]], on=["ksic", "t"], how="left")
        Y["gd2_f"] = Y.gd2.fillna(Y[fb])
    if spec.get("induty"):
        Y = Y.merge(induty_table(spec), on=["ksic", "t"], how="left")
        Y["gd_induty_f"] = Y.gd_induty.fillna(Y[fb])
    Y["cov_ratio"] = Y.cov_gd / (Y.ship_prev * 1e6)
    Y["d11"] = (Y.t >= 2021).astype(float)      # 11차(등록기반) 조사 체계 더미
    for f in spec.get("derived", []):
        if f == "gd_minus_idx":
            Y[f] = Y.gd_f - Y.g_idx
        if f == "ind_bias":     # 과거(≤t−1) 산업별 평균 잔차(g − g_idx), 학습창 안에서 계산 → run 에서 처리
            pass
    return Y


class _Avg:
    def __init__(self, ms): self.ms = ms
    def fit(self, X, y, sample_weight=None):
        for m in self.ms:
            m.fit(X, y) if sample_weight is None else m.fit(X, y, sample_weight=sample_weight)
        return self
    def predict(self, X): return np.mean([m.predict(X) for m in self.ms], axis=0)


def _model(name, alpha=1.0, q=0.5):
    if name == "ols":
        return LinearRegression()
    if name == "huber":
        return HuberRegressor(epsilon=1.35, max_iter=1000)
    if name == "ridge":
        return Ridge(alpha=alpha)
    if name == "qr":
        return QuantileRegressor(quantile=q, alpha=alpha, solver="highs")
    if name == "qr_huber":
        return _Avg([QuantileRegressor(quantile=q, alpha=alpha, solver="highs"), HuberRegressor(epsilon=1.35, max_iter=1000)])
    raise ValueError(name)


def run(Y, spec, origins=None):
    """원점별 예측 dict {t: DataFrame(ksic, y, p)}."""
    feats = list(spec["feats"])
    P = {}
    for t in (origins or DEV):
        tr = Y[(Y.t <= t - 1) & (Y.t >= spec.get("train_start", TRAIN_START))].dropna(subset=["g"]).copy()
        te = Y[Y.t == t].copy()
        if origins is None:
            te = te.dropna(subset=["g"])
        if spec.get("ind_bias"):        # 산업 고정효과(수축): 학습창 안의 산업별 평균 잔차 (g − 기준 예측)
            lam = spec["ind_bias"]
            b0 = LinearRegression().fit(tr[["g_idx"]], tr.g)
            r = tr.g - b0.predict(tr[["g_idx"]])
            gm = r.groupby(tr.ksic).agg(["sum", "count"])
            bias = gm["sum"] / (gm["count"] + lam)
            tr["ind_bias"] = tr.ksic.map(bias).fillna(0)   # (주의) 학습행에는 자기 포함 — 수축으로 완화
            te["ind_bias"] = te.ksic.map(bias).fillna(0)
            if "ind_bias" not in feats:
                feats = feats + ["ind_bias"]
        ytr = tr.g.clip(-spec["clip_y"], spec["clip_y"]) if spec.get("clip_y") else tr.g
        med = tr[feats].median()
        Xtr, Xte = tr[feats].fillna(med), te[feats].fillna(med)
        m = _model(spec.get("model", "ols"), spec.get("alpha", 1.0), spec.get("q", 0.5))
        sw = None
        if spec.get("decay"):
            sw = spec["decay"] ** (t - 1 - tr.t.values)
        if sw is not None and spec.get("model", "ols") not in ("huber", "qr_huber"):
            m.fit(Xtr, ytr, sample_weight=sw)
        else:
            m.fit(Xtr, ytr)
        p = m.predict(Xte)
        if spec.get("post") == "hier":      # 3자리 계층 정합: 같은 3자리 그룹 예측의 평균을 3자리 지수 기반 예측 쪽으로 부분 이동
            lam = spec.get("hier_lam", 0.5)
            b0 = LinearRegression().fit(tr[["g_idx"]], tr.g)
            pb = b0.predict(te[["g_idx"]])
            df = pd.DataFrame({"k3": te.k3.values, "p": p, "pb": pb})
            gp = df.groupby("k3").p.transform("mean"); gb = df.groupby("k3").pb.transform("mean")
            p = p + lam * (gb - gp).values
        if spec.get("post") == "blend_zero":
            p = p * spec.get("blend", 0.8) + (1 - spec.get("blend", 0.8)) * np.median(ytr)
        P[t] = pd.DataFrame({"ksic": te.ksic.values, "y": te.g.values, "p": p})
    return P


def mae(P):
    return {t: float(np.mean(np.abs(P[t].y - P[t].p))) for t in P}


def metrics(P, base):
    m, b = mae(P), mae(base)
    sk = {t: 1 - m[t] / b[t] for t in DEV}
    return {"mae": m, "skill": sk, "mean_mae": float(np.mean(list(m.values()))), "mean_skill": float(np.mean(list(sk.values()))),
            "medae": float(np.mean([np.median(np.abs(P[t].y - P[t].p)) for t in DEV])),
            "dir": float(np.mean([np.mean(np.sign(P[t].y - P[t].y.median()) == np.sign(P[t].p - P[t].p.median())) for t in DEV])),
            "rho": float(np.mean([spearmanr(P[t].y, P[t].p).statistic for t in DEV]))}


BASE_SPEC = {"feats": ["g_idx"], "model": "ols"}


def baseline():
    Y = panel()
    return run(Y, BASE_SPEC)


def boot_ci(P, base, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    e1 = pd.concat([P[t].assign(t=t) for t in DEV]); e0 = pd.concat([base[t].assign(t=t) for t in DEV])
    e1["e"] = (e1.y - e1.p).abs(); e0["e"] = (e0.y - e0.p).abs()
    E1 = e1.groupby(["ksic", "t"]).e.sum().unstack(); E0 = e0.groupby(["ksic", "t"]).e.sum().unstack()
    ks = E1.index.values; out = []
    for _ in range(n):
        s = rng.choice(ks, len(ks), replace=True)
        out.append(np.mean(1 - E1.loc[s].sum() / E0.loc[s].sum()))
    return np.percentile(out, [2.5, 50, 97.5]), float(np.mean(np.array(out) > 0))


# ---------------- 챔피언/도전자 기록 ----------------
LOG = os.path.join(R2, "experiments.csv")
STATE = os.path.join(R2, "champion.json")


def log_row(exp_id, parent, change, met, decision, note="", champ_met=None):
    row = {"exp_id": exp_id, "parent": parent, "change": change}
    for t in DEV:
        row[f"mae_{t}"] = round(met["mae"][t], 5)
    for t in DEV:
        row[f"skill_{t}"] = round(met["skill"][t], 4)
    row.update({"mean_mae": round(met["mean_mae"], 5), "mean_skill": round(met["mean_skill"], 4),
                "rho": round(met["rho"], 3), "dir": round(met["dir"], 3), "medae": round(met["medae"], 5)})
    if champ_met is not None:
        row["d_skill_vs_champ"] = round(met["mean_skill"] - champ_met["mean_skill"], 4)
        row["win_vs_champ"] = round(float(np.mean([met["mae"][t] < champ_met["mae"][t] for t in DEV])), 3)
    else:
        row["d_skill_vs_champ"] = ""; row["win_vs_champ"] = ""
    row.update({"decision": decision, "note": note})
    df = pd.DataFrame([row])
    if os.path.exists(LOG):
        old = pd.read_csv(LOG, encoding="utf-8-sig")
        old = old[old.exp_id != exp_id]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(LOG, index=False, encoding="utf-8-sig")
    return row


def save_state(exp_id, spec):
    json.dump({"exp_id": exp_id, "spec": spec}, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def load_state():
    return json.load(open(STATE, encoding="utf-8"))


def evaluate(spec):
    Y = build(spec)
    return run(Y, spec)


class Arena:
    def __init__(self):
        self.base = baseline()
        st = load_state()
        self.champ_id, self.champ = st["exp_id"], st["spec"]
        self.cP = evaluate(self.champ)
        self.cm = metrics(self.cP, self.base)
        print(f"[champion {self.champ_id}] skill {self.cm['mean_skill']:.4f} mae {self.cm['mean_mae']:.4f}", flush=True)

    def trial(self, exp_id, change, note="", **upd):
        spec = json.loads(json.dumps(self.champ)); spec.update(upd)
        spec = {k: v for k, v in spec.items() if v is not None}
        P = evaluate(spec); m = metrics(P, self.base)
        d = m["mean_skill"] - self.cm["mean_skill"]
        win = float(np.mean([m["mae"][t] < self.cm["mae"][t] for t in DEV]))
        dec = "ACCEPT" if (d >= 0.003 and win >= 0.6) else "ROLLBACK"
        if dec == "ACCEPT" and m["medae"] > self.cm["medae"]:
            note = (note + " | ⚠중앙절대오차 악화 %.4f→%.4f" % (self.cm["medae"], m["medae"])).strip(" |")
        log_row(exp_id, self.champ_id, change, m, dec, note, self.cm)
        print(f"{exp_id} {change}: skill {m['mean_skill']:.4f} (d {d:+.4f}, win {win:.2f}) rho {m['rho']:.3f} dir {m['dir']:.3f} medae {m['medae']:.4f} -> {dec}", flush=True)
        if dec == "ACCEPT":
            self.champ_id, self.champ, self.cP, self.cm = exp_id, spec, P, m
            save_state(exp_id, spec)
        return m, dec, P


def _control(self, exp_id, change, note="", **upd):
    """대조 실험(보고서 핵심 표용): 챔피언에서 한 요소만 바꿔 채점, 챔피언은 유지. decision=CONTROL."""
    spec = json.loads(json.dumps(self.champ)); spec.update(upd)
    spec = {k: v for k, v in spec.items() if v is not None}
    P = evaluate(spec); m = metrics(P, self.base)
    log_row(exp_id, self.champ_id, change, m, "CONTROL", note, self.cm)
    print(f"{exp_id} CONTROL {change}: skill {m['mean_skill']:.4f} (d {m['mean_skill'] - self.cm['mean_skill']:+.4f}) rho {m['rho']:.3f} dir {m['dir']:.3f} medae {m['medae']:.4f}", flush=True)
    return m, P


Arena.control = _control


def _rebase(self, exp_id, change, note="", **upd):
    """누수 보완 재기준: 규칙상 점수와 무관하게 챔피언 정의를 교체(원점 시점 정보만 쓰도록). 기록은 decision=REBASE."""
    spec = json.loads(json.dumps(self.champ)); spec.update(upd)
    P = evaluate(spec); m = metrics(P, self.base)
    d = m["mean_skill"] - self.cm["mean_skill"]
    win = float(np.mean([m["mae"][t] < self.cm["mae"][t] for t in DEV]))
    log_row(exp_id, self.champ_id, change, m, "REBASE(누수보완)", note, self.cm)
    print(f"{exp_id} REBASE {change}: skill {m['mean_skill']:.4f} (d {d:+.4f}, win {win:.2f})", flush=True)
    self.champ_id, self.champ, self.cP, self.cm = exp_id, spec, P, m
    save_state(exp_id, spec)
    return m


Arena.rebase = _rebase
