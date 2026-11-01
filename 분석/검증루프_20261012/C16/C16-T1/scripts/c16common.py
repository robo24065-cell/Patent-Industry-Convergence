# -*- coding: utf-8 -*-
"""C16-T1 공용: 경로, 읽기 전용 원천 로드, 가시일 계산. 트랙 파일은 읽기만 한다."""
import sys, pathlib, json, os, ctypes, ctypes.wintypes as wt
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
MC = B/'분석/모델경쟁_20260927'; T1 = MC/'T1_산업대기실'; R3 = T1/'R3'; FC = R3/'cache_final'/'o2019'; RD = R3/'runs'/'FROZEN_final'
CACHE = T1/'cache'; KR = B/'공모전데이터/작업데이터/kr_registered.parquet'
HERE = pathlib.Path(__file__).resolve().parent.parent; CC = HERE/'cache'; CC.mkdir(exist_ok=True)
C15 = B/'분석/검증루프_20261012/C15/C15-T2'
CUT = 2019; CAPV = 20191231; END = 20250822


class _PMC(ctypes.Structure):
    _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


def mem():
    c = _PMC(); c.cb = ctypes.sizeof(_PMC); k = ctypes.WinDLL('kernel32'); ps = ctypes.WinDLL('psapi')
    k.GetCurrentProcess.restype = wt.HANDLE; ps.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]
    ps.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb); return round(c.WorkingSetSize / 2**30, 2), round(c.PeakWorkingSetSize / 2**30, 2)


def s8(x):
    s = pd.Series(x).fillna('').astype(str).str.strip().str[:8]
    return pd.to_numeric(s.where(s.str.len() == 8, '99999999'), errors='coerce').fillna(99999999).astype(np.int64).values


def p18(ad):
    d = pd.to_datetime(pd.Series(ad).astype(str).str[:8], format='%Y%m%d', errors='coerce') + pd.DateOffset(months=18)
    return pd.to_numeric(d.dt.strftime('%Y%m%d'), errors='coerce').fillna(99999999).astype(np.int64).values


def load_ins():
    ins = pd.read_parquet(CACHE/'ins_firm.parquet', columns=['rgst_no', 'applicant_cd', 'ksic', 'ay', 'appl_date', 'rgst_date'])
    ins = ins[ins.ay >= 2000].reset_index(drop=True)
    kp = pd.read_parquet(KR, columns=['등록번호', '공개일자']); km = dict(zip(kp['등록번호'].astype(str), kp['공개일자'].astype(str)))
    r = s8(ins.rgst_date.values); pb = s8(ins.rgst_no.astype(str).map(km).values); a18 = p18(ins.appl_date.values)
    ins['v_rgst'] = r; ins['v_pub'] = np.minimum(r, pb); ins['v_p18'] = np.minimum(ins.v_pub.values, a18)
    ins['pub_missing'] = pb == 99999999
    return ins


def load_oos():
    oos = pd.read_parquet(CACHE/'oos.parquet', columns=['applicant_cd', 'rgst_no', 'ay', 'appl_date', 'rgst_date', 'self'])
    oos = oos[oos.self & (oos.ay >= 2000)].drop(columns=['self']).reset_index(drop=True)
    r = s8(oos.rgst_date.values); a18 = p18(oos.appl_date.values)
    oos['v_rgst'] = r; oos['v_pub'] = np.minimum(r, a18); oos['v_p18'] = oos.v_pub.values
    return oos


def kidx():
    K = sorted(pd.read_parquet(CACHE/'ins_pat.parquet', columns=['ksic']).ksic.unique()); return K, {k: i for i, k in enumerate(K)}


def load_app():
    a = pd.read_parquet(CACHE/'app_pub.parquet'); K, ki = kidx()
    a['k'] = a.ksic.astype(str).map(ki); a = a.dropna(subset=['k']); a['k'] = a.k.astype(int)
    a['pub_date'] = a.pub_date.fillna('').astype(str); a['rgst_date'] = a.rgst_date.fillna('').astype(str)
    a = a[a.ay >= 2000].reset_index(drop=True)
    p = s8(a.pub_date.values); r = s8(a.rgst_date.values); a['v'] = np.minimum(p, r)
    return a


def jdump(o, p): json.dump(o, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
