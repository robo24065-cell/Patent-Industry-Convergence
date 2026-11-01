# -*- coding: utf-8 -*-
"""C07-T3 s00: 0단계 노출 점검(2025년 이후 결과 집계 전).
- 동결 산출물·코드 sha256·mtime 기록
- reg_lt20250101.pkl 각 표의 날짜 최댓값(< 2025-01-01) 점검
- future_scores 행 수·KSIC 수·top1pct 수 (결과 미사용)
읽기 전용. 출력: ../cache/s00_out.json"""
import os, sys, json, hashlib, time, pickle
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\N8_특허분쟁표적레이더"
REGD = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인"
files = ["future_scores_r6.parquet", "future_ksic_r6.csv", "final_run_r6.py", "n8lib.py", "final_spec_r6.json",
         "final_result_r6.json", "final_preds_r6.parquet", "future_run.log", "final_run.log", r"cache\reg_lt20250101.pkl",
         r"cache\reg_lt20230101.pkl", r"cache\panel_dev.parquet", r"cache\fam_tokens.parquet"]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
out = {"files": {}}
for f in files:
    p = os.path.join(H, f)
    st = os.stat(p)
    out["files"][f] = dict(sha256=sha(p), size=st.st_size, mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)))
for f in ["registry_full.sqlite", "registry.sqlite"]:
    st = os.stat(os.path.join(REGD, f))
    out["files"]["REG:" + f] = dict(size=st.st_size, mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)))
spec = json.load(open(os.path.join(H, "final_spec_r6.json"), encoding="utf-8"))
out["n8lib_sha_matches_frozen"] = out["files"]["n8lib.py"]["sha256"] == spec["code_sha256"]["n8lib.py"]
# reg_lt20250101 날짜 점검
R = pd.read_pickle(os.path.join(H, r"cache\reg_lt20250101.pkl"))
chk = {}
for k in ("rt", "py", "lc", "ow", "um"):
    d = pd.to_numeric(R[k].d.replace("", np.nan), errors="coerce")
    chk[k] = dict(rows=int(len(R[k])), max_d=float(d.max()), n_null_d=int(d.isna().sum()), n_ge_20250101=int((d >= 20250101).sum()))
chk["pt_rows"] = int(len(R["pt"]))
rg = pd.to_numeric(R["pt"].rgstDate, errors="coerce")
chk["pt_rg_gt_20241231"] = int((rg > 20241231).sum())
chk["keys"] = list(R.keys())
out["reg_lt20250101"] = chk
del R
S = pd.read_parquet(os.path.join(H, "future_scores_r6.parquet"))
out["future_scores"] = dict(rows=int(len(S)), cols=list(S.columns), n_ksic=int(S.ksic.nunique()), n_top1=int(S.top1pct.sum()),
                            sum_p=float(S.p.sum()), dup_rgst=int(S.rgst_no.duplicated().sum()))
K = pd.read_csv(os.path.join(H, "future_ksic_r6.csv"), encoding="utf-8-sig", dtype={"ksic": str})
g = S.groupby("ksic").agg(n=("p", "size"), exp=("p", "sum"), n1=("top1pct", "sum")).reset_index()
m = K.merge(g, on="ksic", how="outer")
out["future_ksic_consistency"] = dict(rows=int(len(K)), all_match=bool(((m.n_x == m.n_y) & (np.abs(m.exp_disputes - m.exp) < 1e-3) & (m.n_top1pct == m.n1)).all()),
                                      max_abs_exp_diff=float(np.abs(m.exp_disputes - m.exp).max()))
json.dump(out, open(os.path.join(W, "cache", "s00_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
