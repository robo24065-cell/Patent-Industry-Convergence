# -*- coding: utf-8 -*-
"""C07-T3 공용: 공식 final_run_r6.py의 fit_score를 그대로 옮기고, n8lib 복사본(sha 확인)을 불러온다. 트랙 폴더에는 쓰지 않는다."""
import os, sys, json, hashlib
sys.dont_write_bytecode = True
HERE_ME = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE_ME)
FROZEN_SHA = "f74ad4fbd88bb38a9261f2525068fd5c03e0442489a9ee17f72e8fbd2204c002"
assert hashlib.sha256(open(os.path.join(HERE_ME, "n8lib_copy.py"), "rb").read()).hexdigest() == FROZEN_SHA
sys.path.insert(0, HERE_ME)
import n8lib_copy as L  # noqa
import numpy as np, pandas as pd
# 캐시가 있어야 extract/family_tokens가 쓰기 없이 읽기만 한다
for f in ("reg_lt20230101.pkl", "reg_lt20250101.pkl", "fam_tokens.parquet", "panel_dev.parquet"):
    assert os.path.exists(os.path.join(L.CACHE, f)), f
SPEC = json.load(open(os.path.join(L.HERE, "final_spec_r6.json"), encoding="utf-8"))
CH = SPEC["champion"]; REFS = SPEC["references"]
NEG_FRAC, NEG_SEED = CH["neg_frac"], CH["neg_seed"]
MODELS = {"champ": (CH["feats"], CH["model"]), "M0": (REFS["M0"]["feats"], "xgb"), "SB": (REFS["SB"]["feats"], "xgb")}
NEED = list(dict.fromkeys(CH["feats"] + REFS["M0"]["feats"] + REFS["SB"]["feats"]))


def rss():
    """현재 프로세스 작업 집합(GB) — psutil 없이 Windows API."""
    import ctypes
    from ctypes import wintypes
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    k = ctypes.windll.kernel32; ps = ctypes.windll.psapi
    k.GetCurrentProcess.restype = wintypes.HANDLE
    ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    ps.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return round(c.WorkingSetSize / 2**30, 2), round(c.PeakWorkingSetSize / 2**30, 2)


def all_feats():
    s = list(CH["feats"])
    for r in REFS.values():
        s += r["feats"]
    return list(dict.fromkeys(s))


def fit_score(TR, TE, feats, model):
    """final_run_r6.fit_score 그대로: 음성 5% 표본(시드 0, 가중 1/0.05)."""
    rng = np.random.default_rng(NEG_SEED)
    y = TR.y.values.astype(int)
    keep = (y == 1) | (rng.random(len(TR)) < NEG_FRAC)
    w = np.where(y == 1, 1.0, 1.0 / NEG_FRAC)[keep]
    return L.fit_pred(TR.loc[keep, feats].astype(float), y[keep], TE[feats].astype(float), model, w=w)


def load_dev_panel_like_official():
    """final_run_r6.load_dev_panel과 같은 열 집합(행 순서·dtype 동일)."""
    cols = list(dict.fromkeys(["rgst_no", "t0", "y", "ksic", "owner_v2", "own_disp", "own_trial"] + all_feats()))
    return pd.read_parquet(f"{L.CACHE}/panel_dev.parquet", columns=cols).reset_index(drop=True)
