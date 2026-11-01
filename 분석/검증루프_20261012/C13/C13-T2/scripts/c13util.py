# -*- coding: utf-8 -*-
"""C13-T2 공용 도우미. 트랙 폴더에는 쓰지 않는다(쓰기는 C13-T2 아래만)."""
import os, sys, json, time, hashlib
import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(SC)
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
CACHE = os.path.join(W, "cache")
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
_T = time.time()


def tlog(*a):
    print(*a, f"[{round(time.time() - _T)}s rss={rss()}]", flush=True)


def rss():
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    k = ctypes.WinDLL("kernel32"); ps = ctypes.WinDLL("psapi")
    k.GetCurrentProcess.restype = wintypes.HANDLE
    ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    ps.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return round(c.PeakWorkingSetSize / 2**30, 2)


def tdir(trk):
    p = os.path.join(CACHE, trk); os.makedirs(p, exist_ok=True)
    return p


def dump(obj, path):
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


def top1(y, p, frac=0.01):
    """상위 frac 적중률(정밀도)·포착률(재현율). 동점은 안정 정렬 순서(트랙 관행과 같은 kind='stable')."""
    y = np.asarray(y).astype(float); p = np.asarray(p)
    k = max(1, int(round(frac * len(y))))
    o = np.argsort(-p, kind="stable")[:k]
    return dict(k=int(k), prec=float(y[o].mean()), capture=float(y[o].sum() / max(1.0, y.sum())), hits=int(y[o].sum()))


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def row(trk, bundle, seeds, auc_c, auc_0, sk, ci, auc_s, sk_s, ci_s, t1, n, pos, extra=None):
    r = dict(track=trk, bundle=bundle, seeds=seeds, n=int(n), pos=int(pos), auc_champ=float(auc_c), auc_M0=float(auc_0), skill=float(sk),
             ci_lo=float(ci[0]), ci_hi=float(ci[1]), auc_strong=float(auc_s), skill_vs_strong=float(sk_s), strong_ci_lo=float(ci_s[0]),
             strong_ci_hi=float(ci_s[1]), top1_prec=t1["prec"], top1_capture=t1["capture"], top1_hits=t1["hits"], top1_k=t1["k"])
    if extra:
        r.update(extra)
    return r
