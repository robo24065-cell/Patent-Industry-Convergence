# -*- coding: utf-8 -*-
"""verify_0 독립 점검(작업 코드·트랙 라이브러리를 쓰지 않음).
(1) 저장된 묶음 예측(cache/<trk>/preds_b*.parquet)으로 AUC를 Mann–Whitney 순위식으로 다시 계산해 묶음 JSON과 대조(차 ≤ 1e−9).
(2) 공식 묶음(b0)이 유리한 쪽으로 골라졌는지: K개 묶음 중 공식 skill·CI 하한·강화 Δ의 순위와 경험적 분위.
(3) 동결 시드가 최종시험 전에 고정됐는지: spec 파일 수정 시각·frozen 필드·seeds 값 vs 최종 결과 파일 시각·run_at, spec에 적힌 코드 sha256과 현재 코드.
→ verify_0/v0_result.json"""
import os, sys, json, glob, hashlib, datetime
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.stdout.reconfigure(encoding="utf-8")
V = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V); C = os.path.join(W, "cache")
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"


def mw(y, p):
    y = np.asarray(y).astype(int); r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


sk = lambda a, b: (a - b) / (1 - b)
COLS = {"n4": ("p_champ", "p_M0", "p_M0ens"), "n6": ("p_champ", "p_M0", "p_M0ens"), "n7": ("p_champ", "p_M0", "p_SB"),
        "n8": ("p_champ", "p_M0", "p_SB"), "n9": ("p_champ", "p_M0_L2", "p_strong_L2"), "t4": ("p_champ", "p_M0", "p_strong")}
R = {"rescore": {}, "selection": {}, "freeze": {}}
for t, (c1, c0, cs) in COLS.items():
    rows = []
    for f in sorted(glob.glob(os.path.join(C, t, "b*.json")), key=lambda s: int(os.path.basename(s)[1:-5])):
        b = int(os.path.basename(f)[1:-5]); J = json.load(open(f, encoding="utf-8"))
        P = pd.read_parquet(os.path.join(C, t, f"preds_b{b}.parquet"))
        a1, a0, a2 = mw(P.y, P[c1]), mw(P.y, P[c0]), mw(P.y, P[cs])
        s0 = sk(a1, a0)
        if t == "t4":      # 강화 Δ는 float64 경로 AUC로 저장됨
            a1s = J.get("auc_champ_float64path", a1); s2 = sk(a1s, a2) if "auc_champ_float64path" not in J else sk(J["auc_champ_float64path"], a2)
        else:
            s2 = sk(a1, a2)
        d = max(abs(a1 - J["auc_champ"]), abs(a0 - J["auc_M0"]), abs(a2 - J["auc_strong"]), abs(s0 - J["skill"]), abs(s2 - J["skill_vs_strong"]))
        rows.append(dict(bundle=b, mw_auc=a1, mw_skill=s0, mw_skill_strong=s2, max_abs_diff_vs_json=d, ci_lo=J["ci_lo"], strong_ci_lo=J["strong_ci_lo"]))
    if not rows:
        continue
    D = pd.DataFrame(rows)
    R["rescore"][t] = dict(K=len(D), max_abs_diff=float(D.max_abs_diff_vs_json.max()), pass_1e9=bool(D.max_abs_diff_vs_json.max() <= 1e-9))
    o = D[D.bundle == 0].iloc[0]
    rk = lambda col: int((D[col] > o[col]).sum() + 1)
    R["selection"][t] = dict(K=len(D), official_skill=float(o.mw_skill), rank_skill=rk("mw_skill"), rank_ci_lo=rk("ci_lo"), rank_strong=rk("mw_skill_strong"),
                             share_bundles_below_official_skill=float((D.mw_skill < o.mw_skill).mean()),
                             skill_mean_other=float(D[D.bundle != 0].mw_skill.mean()), skill_official_minus_mean_other=float(o.mw_skill - D[D.bundle != 0].mw_skill.mean()))
# (3) 동결 시점
mt = lambda p: datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
FZ = {"N4": ("N4_특허행태_기업폐업조기경보", "final_spec_r4.json", "final_result_r4.json", ["final_run_r4.py", "n4lib.py", "n4feat.py"]),
      "N6": ("N6_정부RnD기업_첫특허전환", "final_spec_r5.json", "final_result_r5.json", ["final_run_r5.py", "n6_lib.py", "n6_features.py"]),
      "N7": ("N7_특허기업_정부RnD첫수행", "final_spec_r5.json", "final_result_r5.json", ["final_run_r5.py", "n7lib.py", "n7feat.py"]),
      "N8": ("N8_특허분쟁표적레이더", "final_spec_r6.json", "final_result_r6.json", ["final_run_r6.py", "n8lib.py"]),
      "N9": ("N9_정부RnD과제_특허성과예측", "final_spec_r6.json", "final_result_r6.json", ["final_run_r6.py", "n9_lib.py", "n9_build.py"]),
      "T4": ("T4_특허포기예측", os.path.join("R2", "final_spec_r2.json"), os.path.join("R2", "final_result.json"), [os.path.join("R2", "final_run_r2.py"), os.path.join("R2", "r2lib.py")])}
for t, (d, sp, rs, code) in FZ.items():
    S = json.load(open(os.path.join(MC, d, sp), encoding="utf-8"))
    Rj = json.load(open(os.path.join(MC, d, rs), encoding="utf-8"))
    seeds = None
    for path in (("model", "seeds"), ("champion", "neg_seed"), ("spec", "components", "cls", "n_bag")):
        cur = S
        try:
            for k in path:
                cur = cur[k]
            seeds = {".".join(path): cur}; break
        except (KeyError, TypeError):
            continue
    extra = {}
    if t == "N8":
        extra["neg_seed"] = S["champion"].get("neg_seed"); extra["neg_frac"] = S["champion"].get("neg_frac")
    if t == "N9":
        extra["neg_seed"] = S["model"].get("neg_seed"); extra["neg_frac"] = S["model"].get("neg_frac")
    cs = S.get("code_sha256") or {}
    code_match = {f: (sha(os.path.join(MC, d, f)) == h) for f, h in cs.items()}
    R["freeze"][t] = dict(spec_mtime=mt(os.path.join(MC, d, sp)), result_mtime=mt(os.path.join(MC, d, rs)),
                          spec_before_result=os.path.getmtime(os.path.join(MC, d, sp)) < os.path.getmtime(os.path.join(MC, d, rs)),
                          frozen_field=S.get("frozen") or S.get("frozen_at") or S.get("version"), run_at=Rj.get("run_at") if isinstance(Rj, dict) else None,
                          seeds_in_spec=seeds, **extra, spec_code_sha256_match=code_match or "spec에 코드 해시 없음",
                          code_mtime={f: mt(os.path.join(MC, d, f)) for f in code})
json.dump(R, open(os.path.join(V, "v0_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
