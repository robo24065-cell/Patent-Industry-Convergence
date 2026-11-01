# -*- coding: utf-8 -*-
"""C17-T2 공용 가드(guard16 사본)(C15-T2 guard15 사본)(C10-T1 guard.py 사본 + 메모리 감시).
 - 쓰기 가드: to_parquet/to_csv/np.save/np.savez/open(w)/mkdir/unlink 가 C15-T2 밖을 가리키면 예외.
 - 메모리 감시: 현재 작업 집합이 LIMIT_GB(3.8)를 넘으면 메시지를 남기고 os._exit(9) (자기 프로세스만).
import 만 하면 설치된다. sys.dont_write_bytecode 도 켠다(트랙 폴더 __pycache__ 쓰기 방지)."""
import os, sys, pathlib, builtins, threading, time, ctypes
from ctypes import wintypes
sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent.parent          # C17-T2
LIMIT_GB = float(os.environ.get('C17_MEM_LIMIT_GB', '3.8'))
try:
    sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np, pandas as pd


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
_k = ctypes.WinDLL("kernel32"); _ps = ctypes.WinDLL("psapi")
_k.GetCurrentProcess.restype = wintypes.HANDLE
_ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]


def _mem():
    c = _PMC(); c.cb = ctypes.sizeof(_PMC)
    _ps.GetProcessMemoryInfo(_k.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 2**30, c.PeakWorkingSetSize / 2**30


def rss_now(): return round(_mem()[0], 2)
def rss_peak(): return round(_mem()[1], 2)


def _watch():
    while True:
        cur, pk = _mem()
        if cur > LIMIT_GB:
            try:
                print(f'[메모리 감시] 작업 집합 {cur:.2f}GB > {LIMIT_GB}GB — 자진 종료(코드 9)', flush=True)
            finally:
                os._exit(9)
        time.sleep(0.1)


threading.Thread(target=_watch, daemon=True).start()


def _ok(p):
    try:
        rp = pathlib.Path(p).resolve()
    except Exception:
        return False
    return str(rp).lower().startswith(str(HERE.resolve()).lower())


def _wrap(fn, name):
    def g(obj, path, *a, **k):
        if isinstance(path, (str, os.PathLike)) and not _ok(path):
            raise PermissionError(f'[쓰기 가드] {name} 밖 경로 거부: {path}')
        return fn(obj, path, *a, **k)
    return g


pd.DataFrame.to_parquet = _wrap(pd.DataFrame.to_parquet, 'to_parquet')
pd.DataFrame.to_csv = _wrap(pd.DataFrame.to_csv, 'to_csv')
pd.Series.to_csv = _wrap(pd.Series.to_csv, 'to_csv')
_np_save, _np_savez = np.save, np.savez


def _save(path, *a, **k):
    if isinstance(path, (str, os.PathLike)) and not _ok(path): raise PermissionError(f'[쓰기 가드] np.save 밖 경로 거부: {path}')
    return _np_save(path, *a, **k)


def _savez(path, *a, **k):
    if isinstance(path, (str, os.PathLike)) and not _ok(path): raise PermissionError(f'[쓰기 가드] np.savez 밖 경로 거부: {path}')
    return _np_savez(path, *a, **k)


np.save, np.savez = _save, _savez
_open = builtins.open


def _gopen(file, mode='r', *a, **k):
    if any(ch in mode for ch in 'wax+') and isinstance(file, (str, os.PathLike)) and not _ok(file):
        raise PermissionError(f'[쓰기 가드] open({mode}) 밖 경로 거부: {file}')
    return _open(file, mode, *a, **k)


builtins.open = _gopen
_mkdir = pathlib.Path.mkdir


def _gmkdir(self, *a, **k):
    if not self.exists() and not _ok(self): raise PermissionError(f'[쓰기 가드] mkdir 밖 경로 거부: {self}')
    return _mkdir(self, *a, **k)


pathlib.Path.mkdir = _gmkdir
_unlink = pathlib.Path.unlink


def _gunlink(self, *a, **k):
    if not _ok(self): raise PermissionError(f'[쓰기 가드] unlink 밖 경로 거부: {self}')
    return _unlink(self, *a, **k)


pathlib.Path.unlink = _gunlink
try:
    import pyarrow.parquet as _pq
    _wt = _pq.write_table
    def _gwt(table, where, *a, **k):
        if isinstance(where, (str, os.PathLike)) and not _ok(where): raise PermissionError(f'[쓰기 가드] pq.write_table 밖 경로 거부: {where}')
        return _wt(table, where, *a, **k)
    _pq.write_table = _gwt
except Exception:
    pass
