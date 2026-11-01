# -*- coding: utf-8 -*-
r"""T4 R7: 제공 KSIC 라벨에 의존하는 피처를 공식 IPC-KSIC 연계표 배정(xw)으로 다시 계산해 xcols/<O>/<stem>.parquet 로 저장.
대상 열(같은 이름 유지): ksic(범주), port_ksic_n, port_share_ksic, aks_rate·aks_rate1y·aks_n·aks_alive, mac_* 5개.
같은 코드로 원래 라벨(ksic)도 다시 계산해 동결 프레임 값과 대조(chk_<O>.json) — 재계산 경로 검증.
python e7_t4_x.py O18   (원점 1개, 9분 이내). 읽기: R2/runs/<O>/*.parquet, R2/cache/loaded_r2dev.pkl(읽기 전용)."""
import os, sys, json, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); T4 = os.path.dirname(HERE); R2 = os.path.join(T4, "R2")
sys.path.insert(0, R2); sys.path.insert(0, os.path.join(HERE, "common"))
os.environ.setdefault("T4R2_MODE", "dev")
import numpy as np, pandas as pd, pyarrow.parquet as pq
import r2lib as R
import t4lib as T, soc as SO, feat2 as F2
from xwalk import load_app_xw

COLS = ["ksic", "port_ksic_n", "port_share_ksic", "aks_rate", "aks_rate1y", "aks_n", "aks_alive"] + F2.MACRO
D = T.load(); pt, pay, lap = D["pt"], D["pay"], D["lap"]
n = len(pt)
xw = load_app_xw().set_index("appl_no").xw
PT_X = pt.applNo.astype(str).map(xw).fillna("XNA").values
PT_K = pt.ksic.astype(str).values
acd = pt.acd.values
_cache = {}


def per_L(L, kvals):
    key = (L, id(kvals))
    if key in _cache: return _cache[key]
    Ld = np.datetime64(L); L1 = T.add_years(np.array([Ld]), -1)[0]
    reg = pt.rd.values <= Ld
    df_a = pd.DataFrame({"acd": acd[reg], "k": kvals[reg]})
    port_n = df_a.groupby("acd").size(); port_k = df_a.groupby("acd").k.nunique(); same_k = df_a.groupby(["acd", "k"]).size()
    # aks (soc_features 와 같은 식)
    pr = pay[(pay.seq > 0) & (pay.pd <= Ld)]
    ren = np.bincount(pr.pid.values, minlength=n).astype(float)
    ren1 = np.bincount(pr[pr.pd > L1].pid.values, minlength=n).astype(float)
    lp = lap[lap.cd <= Ld]
    lpp = np.bincount(lp.pid.values, minlength=n).astype(float)
    lp1 = np.bincount(lp[lp.cd > L1].pid.values, minlength=n).astype(float)
    pids, _, _ = T.risk_set(L)
    alive = np.zeros(n); alive[pids] = 1
    g = lpp.sum() / max(lpp.sum() + ren.sum(), 1)
    k2 = acd + "|" + kvals
    k2 = np.where(reg, k2, "NOTREG_" + np.arange(n).astype(str))
    k2 = np.where(acd == "NA", "NA_" + np.arange(n).astype(str), k2)
    r, r1, no, ar, _ = SO._rate_by_group(k2, ren, lpp, ren1, lp1, alive, g)
    A = np.full((n, 4), np.nan); A[pids, 0] = r[pids]; A[pids, 1] = r1[pids]; A[pids, 2] = no[pids]; A[pids, 3] = ar[pids]
    _cache[key] = (port_n, port_k, same_k, A)
    return _cache[key]


def build(df, kvals):
    out = np.full((len(df), len(COLS) - 1), np.nan)
    pid = df.pid.values.astype(int)
    for L in df.L.unique():
        idx = np.where(df.L.values == L)[0]
        port_n, port_k, same_k, A = per_L(L, kvals)
        p = pid[idx]; a = acd[p]; kk = kvals[p]
        out[idx, 0] = port_k.reindex(a).values
        out[idx, 1] = same_k.reindex(list(zip(a, kk))).values / port_n.reindex(a).values
        out[idx, 2:6] = A[p]
    o = pd.DataFrame(out[:, :6], columns=COLS[1:7])
    # 경기지표: KSIC 3자리 연결. 연계표 코드 'C2610' → '2610'(앞 3자리 261), 없으면 제조업 전체
    m = pd.DataFrame({"L": df.L.values, "ksic": [k[1:] if k.startswith("C") else k for k in kvals[pid]] if kvals is PT_X else kvals[pid]})
    m = F2.add_macro(m)
    for c in F2.MACRO: o[c] = m[c].values
    o.insert(0, "ksic", kvals[pid])
    return o


o = sys.argv[1]
os.makedirs(os.path.join(HERE, "xcols", o), exist_ok=True)
t = time.time(); chk = {}
for stem in ["frame_cls", "frame_haz", "ev", "ev_haz"]:
    fn = os.path.join(R2, "runs", o, stem + ".parquet")
    have = set(pq.ParquetFile(fn).schema.names)
    df = pq.read_table(fn, columns=["pid", "L"] + [c for c in COLS if c in have]).to_pandas()
    df["L"] = df.L.astype(str)
    X = build(df, PT_X)
    X.to_parquet(os.path.join(HERE, "xcols", o, stem + ".parquet"), index=False)
    # 원 라벨 재계산 대조(표본 20만 행)
    s = df.sample(n=min(200000, len(df)), random_state=0).reset_index(drop=True)
    K = build(s, PT_K)
    c = {}
    for col in COLS:
        if col not in s: continue
        a, b = s[col], K[col]
        if col == "ksic":
            c[col] = float((a.astype(str).values == b.astype(str).values).mean())
        else:
            a = a.astype(float).values; b = b.astype(float).values
            c[col] = float(np.mean(np.isclose(a, b, rtol=1e-4, atol=1e-5) | (np.isnan(a) & np.isnan(b))))
    chk[stem] = dict(rows=len(df), match=c, xw_na=float((X.ksic == "XNA").mean()),
                     agree_parent=float(np.mean([x == y for x, y in zip(X.ksic.values[:200000], pd.Series(PT_K[df.pid.values[:200000].astype(int)]).map(__import__("xwalk").K10_TO_XW).values)])))
    print(o, stem, len(df), {k: round(v, 4) for k, v in c.items()}, f"{time.time() - t:.0f}s", flush=True)
    _cache.clear()
json.dump(chk, open(os.path.join(HERE, "xcols", f"chk_{o}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("peak", R.peak_mem_gb())
