# -*- coding: utf-8 -*-
r"""R8 고유장점 아이디어 파일럿 — 공용 함수.
쓰기는 이 폴더(_ideas_R8)에만. 원본·_shared·다른 트랙 파일은 읽기만.
연계표 배정 규칙은 _shared_report/label_validity/lv_common.py(R7 표준, 41.9% 기준)를 그대로 복사해 쓴다(그 폴더에 __pycache__를 만들지 않기 위해 import 대신 복사).
"""
import os, re, sys, time, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
OUT = os.path.join(MC, "_ideas_R8")
CACHE = os.path.join(OUT, "cache")
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
CW_XLSX = os.path.join(BASE, r"공모전데이터\외부데이터\20131002_IPC-KSIC\20131002_IPC-KSIC.xlsx")
KSIC_DEF = os.path.join(BASE, r"공모전데이터\외부데이터\KSIC해설\ksic50_definitions.json")
KOSIS50 = os.path.join(BASE, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv")
_T = time.time()


def log(*a):
    print(*a, f"[{round(time.time() - _T)}s]", flush=True)


def p(*parts):
    return os.path.join(OUT, *parts)


# ---- lv_common.py 복사 (R7 표준 연계표 배정) ----
GROUP_PREFIX = [
    ("C1500", ["15"]), ("C2030", ["202"]), ("C2042", ["2041"]), ("C2043", ["2042"]), ("C2049", ["2049"]),
    ("C2100", ["21"]), ("C2300", ["23"]), ("C2410", ["241"]), ("C2420", ["242"]), ("C2590", ["259"]),
    ("C2610", ["261"]), ("C2620", ["262"]), ("C2630", ["263"]), ("C2640", ["264"]), ("C2650", ["265"]),
    ("C2710", ["271"]), ("C2720", ["272"]), ("C2730", ["273"]), ("C2810", ["281"]), ("C2850", ["285"]),
    ("C2890", ["289"]), ("C2918", ["2918"]), ("C2919", ["291"]), ("C2920", ["292"]), ("C3000", ["30"]),
    ("C3120", ["312"]), ("C3130", ["313"]), ("C3300", ["33"]),
]
SPACE = [g for g, _ in GROUP_PREFIX if g != "C2918"]


def ksic_to_group(code):
    c = re.sub(r"\D", "", str(code))
    for g, pre in GROUP_PREFIX:
        for pp in pre:
            if c.startswith(pp):
                return g
    return None


def ipc_norm(s):
    s = re.sub(r"\s+", "", str(s or "")).upper()
    m = re.match(r"^([A-H]\d{2}[A-Z])(\d+)/(\d+)", s)
    if m:
        return f"{m.group(1)}{int(m.group(2))}/{m.group(3)}", f"{m.group(1)}{int(m.group(2))}", m.group(1)
    m = re.match(r"^([A-H]\d{2}[A-Z])", s)
    if m:
        return "", "", m.group(1)
    return "", "", ""


def load_crosswalk():
    cc = pd.read_excel(CW_XLSX, sheet_name=1, dtype=str)
    cc.columns = ["ind", "ipc", "cnt"]
    cc["cnt"] = pd.to_numeric(cc.cnt, errors="coerce").fillna(0)
    cc = cc[cc.ind.str.match(r"^[A-Z]\d{4}$", na=False)].copy()
    z = cc.ipc.map(ipc_norm)
    cc["sg"] = [a for a, b, c in z]
    cc["mg"] = [b for a, b, c in z]
    cc["sc"] = [c for a, b, c in z]
    return cc


def make_lookup(cc, constrained=True):
    tabs = {}
    for lvl in ("sg", "mg", "sc"):
        g = cc[cc[lvl] != ""].groupby([lvl, "ind"], as_index=False).cnt.sum()
        tot = g.groupby(lvl).cnt.sum().rename("tot")
        gg = g[g.ind.isin(SPACE)] if constrained else g
        gg = gg.sort_values(["cnt", "ind"], ascending=[False, True]).drop_duplicates(lvl)
        gg = gg.merge(tot, left_on=lvl, right_index=True)
        gg["conf"] = gg.cnt / gg.tot
        tabs[lvl] = gg.set_index(lvl)[["ind", "conf"]]
    return tabs


def assign_keys(sg, mg, sc, tabs):
    """세 키 Series → 연계표 그룹(소그룹→메인그룹→서브클래스 순)."""
    out = pd.Series([None] * len(sg), index=sg.index, dtype=object)
    for lvl, key in (("sg", sg), ("mg", mg), ("sc", sc)):
        need = out.isna()
        t = tabs[lvl]
        hit = need & key.isin(t.index)
        out[hit] = t.loc[key[hit], "ind"].values
    return out


KSIC50 = ['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924',
          '25932', '26111', '26112', '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323',
          '26422', '26429', '26519', '26529', '27199', '27211', '27212', '27301', '28111', '28122', '28511', '28902',
          '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280', '30393', '31202', '31312',
          '33301', '33309']
K2G = {k: ksic_to_group(k) for k in KSIC50}
G2K = {}
for k, g in K2G.items():
    G2K.setdefault(g, []).append(k)
MULTI_K = sorted([k for k in KSIC50 if len(G2K[K2G[k]]) > 1])  # 연계표로 5자리 지목 불가 34개


# ---- 평가 ----
from sklearn.metrics import roc_auc_score


def auc(y, s):
    y = np.asarray(y)
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, s)


def skill(a, b):
    """AUC skill = (AUC_m − AUC_b) / (1 − AUC_b)."""
    return (a - b) / (1 - b)


def topk_hit(y, s, frac=0.01):
    y = np.asarray(y); s = np.asarray(s)
    k = max(1, int(round(len(y) * frac)))
    idx = np.argsort(-s, kind="stable")[:k]
    return float(y[idx].mean())


def boot_ci(diffs_by_origin, n=2000, seed=0):
    """원점별 차이의 평균에 대한 원점 부트스트랩 CI (소표본이라 보수적 참고용)."""
    d = np.asarray([x for x in diffs_by_origin if not np.isnan(x)])
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
