# -*- coding: utf-8 -*-
"""C08-T3 공용: 등록원부(registry_full.sqlite, mode=ro) 직접 적재. t4lib/r2lib import 없음.
pid 규칙 = patent 표 rowid 순서에서 rgstDate 있음 & rgst_no '10' 시작만 남긴 0..n-1 (C04-T2에서 F22 ev.parquet과 불일치 0 확인된 규칙)."""
import os, sys, json, time, sqlite3, hashlib
import numpy as np, pandas as pd
try:
    sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
DB = os.path.join(BASE, "수집파이프라인", "registry_full.sqlite")
TR = os.path.join(BASE, r"분석\모델경쟁_20260927\T4_특허포기예측")
R2 = os.path.join(TR, "R2")
FUT = os.path.join(R2, r"runs_future\FUT")
F22 = os.path.join(R2, r"runs_final\F22")
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(W, "cache")
os.makedirs(CACHE, exist_ok=True)
L0 = np.datetime64("2025-07-01")


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def peak_gb():
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t),
                        ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL('kernel32'); ps = ctypes.WinDLL('psapi')
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return round(c.PeakWorkingSetSize / 1e9, 2)
    except Exception:
        return float('nan')


def pdate(s):
    s = pd.Series(s, dtype="object").astype(str).str.strip()
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce").values.astype("datetime64[D]")


def plus_years(d, k):
    """기념일 + k년 (2/29 → 2/28)"""
    d = pd.DatetimeIndex(d)
    y = d.year.values.astype(float) + np.asarray(k, float)
    mo = d.month.values.astype(float); dd = d.day.values.astype(float)
    dd = np.where((mo == 2) & (dd == 29), 28, dd)
    ok = np.isfinite(y) & np.isfinite(mo)
    out = np.full(len(d), np.datetime64("NaT"), dtype="datetime64[D]")
    if ok.any():
        s = pd.to_datetime(pd.DataFrame({'year': y[ok].astype(int), 'month': mo[ok].astype(int), 'day': dd[ok].astype(int)}), errors='coerce')
        out[ok] = s.values.astype("datetime64[D]")
    return out


def load(force=False):
    """pt(pid, rgst_no, ksic, rd, ad, exp, acd, atype, aname), pay(pid, l, s, pd, seq), rt(pid, cd, name, reason) — parquet 캐시"""
    fp = {k: os.path.join(CACHE, f"reg_{k}.parquet") for k in ('pt', 'pay', 'rt')}
    if not force and all(os.path.exists(v) for v in fp.values()):
        return {k: pd.read_parquet(v) for k, v in fp.items()}
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    pt = pd.read_sql("select rowid as rid, rgst_no, ksic, applDate, rgstDate, cndrtExptnDate from patent order by rowid", con)
    pt["rd"] = pdate(pt.rgstDate); pt["ad"] = pdate(pt.applDate); pt["exp"] = pdate(pt.cndrtExptnDate)
    pt["n_all_rows"] = len(pt)
    pt = pt[(~pd.isna(pt.rd)) & pt.rgst_no.str.startswith("10")].reset_index(drop=True)
    pt["pid"] = np.arange(len(pt))
    m = pd.isna(pt.exp).values
    pt.loc[m, "exp"] = plus_years(pt.loc[m, "ad"].values, np.full(m.sum(), 20))
    idx = pd.Series(pt.pid.values, index=pt.rgst_no.values)
    ap = pd.read_sql("select rgst_no, seq, applicantCd, applicantName from applicant", con)
    ap["seq"] = pd.to_numeric(ap.seq, errors="coerce")
    a1 = ap.sort_values(["rgst_no", "seq"]).drop_duplicates("rgst_no").set_index("rgst_no")
    pt["acd"] = pt.rgst_no.map(a1.applicantCd).fillna("NA")
    pt["aname"] = pt.rgst_no.map(a1.applicantName).fillna("")
    pt["atype"] = pt.acd.str[0].map({'1': 'corp', '2': 'univ', '3': 'inst', '4': 'indiv', '5': 'fcorp', '6': 'findiv'}).fillna('other')
    pay = pd.read_sql("select rgst_no, seq, statAnnl, lastAnnl, payDate from pay", con)
    pay["pid"] = pay.rgst_no.map(idx)
    pay["l"] = pd.to_numeric(pay.lastAnnl, errors="coerce"); pay["s"] = pd.to_numeric(pay.statAnnl, errors="coerce")
    pay["pd"] = pdate(pay.payDate); pay["seq"] = pd.to_numeric(pay.seq, errors="coerce")
    pay = pay[pay.pid.notna() & pay.l.notna() & ~pd.isna(pay.pd)].copy()
    pay["pid"] = pay.pid.astype("int32"); pay["l"] = pay.l.astype("int16"); pay["s"] = pay.s.fillna(pay.l).astype("int16")
    pay = pay[["pid", "seq", "s", "l", "pd"]]
    rt = pd.read_sql("select rgst_no, seq, rgstCsDate, rgstCsName, rgstCsReason from right", con)
    con.close()
    rt["pid"] = rt.rgst_no.map(idx); rt = rt[rt.pid.notna()].copy(); rt["pid"] = rt.pid.astype("int32"); rt["cd"] = pdate(rt.rgstCsDate)
    rt["seq"] = pd.to_numeric(rt.seq, errors="coerce")
    rt = rt.rename(columns={"rgstCsName": "name", "rgstCsReason": "reason"})[["pid", "seq", "cd", "name", "reason"]]
    pt = pt[["pid", "rid", "rgst_no", "ksic", "rd", "ad", "exp", "acd", "aname", "atype", "n_all_rows"]]
    for k, v in (('pt', pt), ('pay', pay), ('rt', rt)): v.to_parquet(fp[k], index=False)
    return dict(pt=pt, pay=pay, rt=rt)


def paid_thru(pt, pay, asof):
    p = pay[pay.pd <= asof]
    lm = p.groupby("pid").l.max()
    lv = np.zeros(len(pt)); lv[lm.index.values] = lm.values
    out = np.full(len(pt), np.datetime64("NaT"), dtype="datetime64[D]")
    ok = lv > 0
    out[ok] = plus_years(pt.rd.values[ok], lv[ok].astype(int))
    return out, lv


MALSO = ["권리의 말소등록", "말소등록", "직권말소등록"]


def risk_set(D, L):
    pt, pay, rt = D['pt'], D['pay'], D['rt']
    L = np.datetime64(L)
    pth, lv = paid_thru(pt, pay, L)
    ok = (pt.rd.values < L) & (pth > L) & (pt.exp.values > L)
    ml = rt[rt.name.isin(MALSO) & (rt.cd <= L)].pid.unique()
    ok[ml] = False
    return np.where(ok)[0], pth, lv


def lapse_events(D):
    """불납 소멸등록 사건 + 실효일 E = 등록일 + (소멸등록일 전 최대 납부연차)년 (t4lib 정의를 독립 구현)"""
    pt, pay, rt = D['pt'], D['pay'], D['rt']
    lap = rt[(rt.name == "소멸등록") & rt.reason.isin(["등록료불납", "연차료불납"]) & ~pd.isna(rt.cd)][["pid", "cd", "seq"]].reset_index(drop=True)
    mm = lap.reset_index().merge(pay[["pid", "pd", "l"]], on="pid")
    mm = mm[mm.pd < mm.cd]
    lmax = mm.groupby("index").l.max()
    lap["lmax"] = lap.index.map(lmax)
    lap = lap[lap.lmax.notna()].copy()
    lap["E"] = plus_years(pt.rd.values[lap.pid.values], lap.lmax.values.astype(int))
    return lap


def jdump(o, p):
    json.dump(o, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
