# -*- coding: utf-8 -*-
"""C04-T1 s3: 동결 예측(final_test_pred_r4.parquet) 재채점 — 재학습·튜닝 없음.
R0 공식값 재현 → 시나리오 ①~④. 사업자번호 군집 부트스트랩 1000회(시드 0)."""
import os, json
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
N4 = os.path.join(BASE, r"분석\모델경쟁_20260927\N4_특허행태_기업폐업조기경보")
HERE = os.path.join(BASE, r"분석\검증루프_20261012\C04\C04-T1")
OUT = os.path.join(HERE, "cache")
W_SKILL = 0.20857437535030798 - 0.14745614490884804
W_AUC = 0.8164203060001841 - 0.7898194648353767

def wauc(y, s, w):
    u, inv = np.unique(s, return_inverse=True)
    wp = np.bincount(inv, w * y, minlength=len(u)); wn = np.bincount(inv, w * (1 - y), minlength=len(u))
    cn = np.cumsum(wn) - wn
    return float(np.sum(wp * (cn + 0.5 * wn)) / (wp.sum() * wn.sum()))

def prep(s):
    u, inv = np.unique(s, return_inverse=True); return inv, len(u)

def wauc_inv(y, inv, k, w):
    wp = np.bincount(inv, w * y, minlength=k); wn = np.bincount(inv, w * (1 - y), minlength=k)
    cn = np.cumsum(wn) - wn
    return np.sum(wp * (cn + 0.5 * wn)) / (wp.sum() * wn.sum())

def topk_prec(y, s, frac=0.01):
    k = int(len(y) * frac); o = np.argsort(-s, kind="stable")[:k]; return float(y[o].mean()), k

def score(name, d, wcol=None, nboot=1000, seed=0, note=""):
    y = d.y.values.astype(float); pc = d.p_champ.values.astype(float); pm = d.p_M0.values.astype(float)
    w0 = d[wcol].values.astype(float) if wcol else np.ones(len(d))
    a1, a0 = wauc(y, pc, w0), wauc(y, pm, w0)
    sk = (a1 - a0) / (1 - a0)
    t1, k = topk_prec(y, pc); t0, _ = topk_prec(y, pm)
    cl = pd.factorize(d.cl.astype(str))[0]; ncl = cl.max() + 1
    ic, kc = prep(pc); im, km = prep(pm)
    rng = np.random.default_rng(seed); A1 = []; A0 = []
    for _ in range(nboot):
        cnt = np.bincount(rng.integers(0, ncl, ncl), minlength=ncl)
        w = w0 * cnt[cl]
        A1.append(wauc_inv(y, ic, kc, w)); A0.append(wauc_inv(y, im, km, w))
    A1 = np.array(A1); A0 = np.array(A0); S = (A1 - A0) / (1 - A0)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(scenario=name, n=len(d), pos=float((y * w0).sum()), base_rate=float((y * w0).sum() / w0.sum()), auc_champ=a1, auc_M0=a0, skill=sk,
                auc_champ_ci=q(A1), skill_ci=q(S), top1pct_k=k, top1pct_prec_champ=t1, top1pct_prec_M0=t0, note=note)

pr = pd.read_parquet(os.path.join(N4, "final_test_pred_r4.parquet"))
pr["cl"] = pr.bizno.astype(str)
att = pd.read_parquet(os.path.join(OUT, "s2_testpop_attrs.parquet"))
d = pr.join(att.drop(columns=["y", "nps_present"]), on="applicantCd")
assert len(d) == 24840 and int(d.y.sum()) == 1109

res = []
# ---------- R0 재현 ----------
r0 = score("S0 공식(재현)", d)
fin = json.load(open(os.path.join(N4, "final_result_r4.json"), encoding="utf-8"))["results"][0]
r0["reproduce_ok"] = bool(abs(r0["auc_champ"] - fin["auc"]["champ"]) < 1e-5 and abs(r0["auc_M0"] - fin["auc"]["M0_xgb"]) < 1e-5
                         and abs(r0["skill"] - fin["skill_vs_M0"]) < 2e-5 and abs(r0["top1pct_prec_champ"] - fin["top1pct_precision"]["champ"]) < 1e-9)
r0["official"] = dict(auc=fin["auc"]["champ"], m0=fin["auc"]["M0_xgb"], skill=fin["skill_vs_M0"], skill_ci=fin["skill_vs_M0_ci95"], top1=fin["top1pct_precision"]["champ"])
res.append(r0)
print(json.dumps(r0, ensure_ascii=False, indent=1))
if not r0["reproduce_ok"]:
    raise SystemExit("R0 재현 실패 — 중단")

S2 = json.load(open(os.path.join(OUT, "s2_population.json"), encoding="utf-8"))
S1 = json.load(open(os.path.join(OUT, "s1_profile.json"), encoding="utf-8"))

# ---------- ① 조회 실패·미등록·미조회 추가 (전부 폐업 간주, 점수 대체 3종) ----------
def add_rows(nadd, rule, tag):
    if rule == "max": vc, vm = d.p_champ.max(), d.p_M0.max()
    elif rule == "median": vc, vm = d.p_champ.median(), d.p_M0.median()
    else: vc, vm = d.p_champ.min(), d.p_M0.min()
    a = pd.DataFrame({"y": 1, "p_champ": vc, "p_M0": vm, "cl": [f"{tag}{i}" for i in range(nadd)]})
    return pd.concat([d[["y", "p_champ", "p_M0", "cl"]], a], ignore_index=True)
n_unreg = S2["pool_grp_excl_closed_by_t0"].get("unregistered", 0)
n_nob = S2["pool_grp_excl_closed_by_t0"].get("no_bizno", 0)
for rule in ["max", "median", "min"]:
    res.append(score(f"S1a 미등록 {n_unreg}개 추가=폐업, 점수 {rule}", add_rows(n_unreg, rule, "U"), note="registered=False (국세청 조회했으나 미등록)"))
for rule in ["max", "median", "min"]:
    res.append(score(f"S1b 사업자번호 없음 {n_nob}개 추가=폐업, 점수 {rule}", add_rows(n_nob, rule, "N"), nboot=300, note="극단 상한(비현실): 미조회 전원 폐업 간주"))
# RC:001(권리목록 조회 실패) — 참고: 제외 시
res.append(score("S1c 권리목록 RC:001 출원인 제외(참고)", d[d.rl_query != "RC:001"], note="RC:001은 KIPRIS 권리목록 조회 상태(2026)이며 국세청·N4 모집단과 무관"))

# ---------- ② 2025 하반기 결손 보정 가중 ----------
d["endd"] = pd.to_numeric(d.end, errors="coerce")
h2 = (d.y == 1) & (d.endd >= 20250701)
for nm, defi in [("전체 r3nts 법인 기준", S1["P3_deficit_2025H2_corp"])]:
    d["w2"] = np.where(h2, 1 / (1 - defi), 1.0)
    res.append(score(f"S2 2025-H2 양성 가중 1/(1-{defi:.3f}) ({nm})", d, wcol="w2"))
# 시험 모집단 자체 기준(보조): 시험 양성의 H2/H1 비 vs 2017–24 중앙값
pos_m = S2["pos_end_month"]
h1_25 = sum(v for k, v in pos_m.items() if k.startswith("2025") and int(k[5:]) <= 6); h2_25 = sum(v for k, v in pos_m.items() if k.startswith("2025") and int(k[5:]) > 6)
defi_t = 1 - (h2_25 / h1_25) / S1["P3_med_h2h1_2017_24_corp"]
d["w2t"] = np.where(h2, 1 / (1 - max(defi_t, 0)), 1.0)
res.append(score(f"S2t 2025-H2 양성 가중 1/(1-{defi_t:.3f}) (시험 양성 자체 H2/H1 기준, 보조)", d, wcol="w2t"))

# ---------- ③ 휴업·합병형·폐업 후 활동 ----------
hy = d.b_stt_cd == "02"
res.append(score(f"S3a 휴업 {int(hy.sum())}개 제외", d[~hy]))
dd = d.copy(); dd.loc[hy, "y"] = 1
res.append(score(f"S3a' 휴업 {int(hy.sum())}개 폐업 간주", dd))
flag = set(pd.read_csv(os.path.join(OUT, "s2_pos_flag_merge_or_postpay.csv"), dtype=str).applicantCd)
res.append(score(f"S3b 합병형·폐업 후 납부 양성 {len(flag)}개 제외", d[~d.applicantCd.isin(flag)]))
dd = d.copy(); dd.loc[dd.applicantCd.isin(flag), "y"] = 0
res.append(score(f"S3c 합병형·폐업 후 납부 양성 {len(flag)}개 존속 간주", dd))
S2j = json.load(open(os.path.join(OUT, "s2_population.json"), encoding="utf-8"))
mg = set(pd.read_csv(os.path.join(OUT, "s2_pos_flag_merge.csv"), dtype=str).applicantCd)
res.append(score(f"S3b-m 합병형(±1년 합병 이전) 양성 {len(mg)}개만 제외", d[~d.applicantCd.isin(mg)]))
res.append(score(f"S3b-p 폐업 후 납부 양성 {len(flag - mg)}개만 제외(합병형 제외분)", d[~d.applicantCd.isin(flag - mg)]))
res.append(score("S3d 휴업 제외 + 합병형·폐업후납부 제외", d[~hy & ~d.applicantCd.isin(flag)]))

# ---------- ④ NPS 연결 하위집단 (as-of 2023-12) ----------
res.append(score("S4a NPS 연결(as-of t0)", d[d.nps_present == 1]))
res.append(score("S4b NPS 비연결(as-of t0)", d[d.nps_present == 0]))

out = pd.DataFrame(res)
for c_ in ["auc_champ", "skill", "top1pct_prec_champ"]:
    out["d_" + c_] = out[c_] - r0[c_]
out["skill_ci_lo"] = out.skill_ci.map(lambda v: v[0]); out["skill_ci_hi"] = out.skill_ci.map(lambda v: v[1])
out["auc_ci_lo"] = out.auc_champ_ci.map(lambda v: v[0]); out["auc_ci_hi"] = out.auc_champ_ci.map(lambda v: v[1])
out["flag_dskill_gt_halfW"] = out.d_skill.abs() > W_SKILL / 2
out["flag_dauc_gt_halfW"] = out.d_auc_champ.abs() > W_AUC / 2
cols = ["scenario", "n", "pos", "base_rate", "auc_champ", "auc_ci_lo", "auc_ci_hi", "auc_M0", "skill", "skill_ci_lo", "skill_ci_hi", "top1pct_k",
        "top1pct_prec_champ", "top1pct_prec_M0", "d_auc_champ", "d_skill", "d_top1pct_prec_champ", "flag_dskill_gt_halfW", "flag_dauc_gt_halfW", "note"]
out[cols].to_csv(os.path.join(HERE, "scenario_rescore.csv"), index=False, encoding="utf-8-sig")
json.dump(dict(W_skill_half=W_SKILL / 2, W_auc_half=W_AUC / 2, defi_testpos=defi_t, h1_25=h1_25, h2_25=h2_25), open(os.path.join(OUT, "s3_meta.json"), "w"), indent=1)
pd.set_option("display.width", 250)
print(out[["scenario", "n", "pos", "auc_champ", "auc_M0", "skill", "skill_ci_lo", "skill_ci_hi", "top1pct_prec_champ", "d_skill", "flag_dskill_gt_halfW", "flag_dauc_gt_halfW"]].round(4).to_string())
