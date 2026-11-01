# -*- coding: utf-8 -*-
"""C19-T2 공용 경로·도우미. 입력은 읽기만, 쓰기는 C19-T2 아래만."""
import os, sys, json, time, hashlib
J = os.path.join
BASE = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
AN = J(BASE, '분석')
MC = J(AN, '모델경쟁_20260927')
T4 = J(MC, 'T4_특허포기예측')
F22 = J(T4, 'R2', 'runs_final', 'F22')
META = J(MC, '_shared', 'meta', '*.parquet').replace(os.sep, '/')
W = J(AN, '검증루프_20261012', 'C19', 'C19-T2')
C = J(W, 'cache')
MIRROR = J(AN, '검증루프_20261012', 'C13', 'C13-T2', 'scripts', 't4mirror')
FRAMES = ['frame_cls', 'frame_haz', 'ev', 'ev_haz']
try:
    sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
_T = time.time()


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


def out(*a):
    print(*a, f'[{round(time.time() - _T)}s peak={rss()}GB]', flush=True)


def jdump(o, path):
    json.dump(o, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def con3():
    import duckdb
    con = duckdb.connect()
    tmp = J(C, 'duck_tmp'); os.makedirs(tmp, exist_ok=True)
    con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
    con.execute(f"set temp_directory='{tmp.replace(os.sep, '/')}'")
    return con
