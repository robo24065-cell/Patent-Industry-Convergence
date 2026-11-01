# -*- coding: utf-8 -*-
"""C10-T1 민감도 실행기 — 동결 R3E29 spec 그대로, 디자인·실용신안 입력만 바꿈.
동결 실행기 R3/final_run_r3.py 의 train_origin()을 import해 그대로 쓰고 산출은 C10-T1/runs/<변형>/ 에만 쓴다.
사용: python run_var.py <S0|S1|S2|S3|S4> <원점: 2019 또는 2008..2012>   (종료코드 3 = 단계 종료, 같은 명령 재실행)
쓰기 가드: to_parquet/to_csv/np.save/np.savez/open('w')가 C10-T1 밖을 가리키면 예외."""
import sys, os, json, pathlib, builtins
HERE = pathlib.Path(__file__).resolve().parent.parent          # C10-T1
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; R3 = T1/'R3'
sys.path.insert(0, str(R3))

# ---------------- 쓰기 가드 ----------------
import numpy as np, pandas as pd
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

os.chdir(str(T1))
import final_run_r3 as F
import r3ext

DS = ['ds_lsum', 'ds_max', 'ds_n20', 'ds_rec3', 'ds_n']; UM = ['um_lsum', 'um_max', 'um_n20', 'um_rec3', 'um_n']
PR = ['pr_ppi_g3', 'pr_ppi_g1', 'pr_exp_g3', 'pr_imp_g3', 'pr_gap']
IO = ['io_down', 'io_up', 'io_sell', 'io_max', 'io_same']


def title_exposure_vf(tr, te, c, kind, pre, FCp, var, thr=0.2):
    """r3ext._title_exposure 와 같은 계산. 단, 행 포함 조건을 rgst_date ≤ cut연말 대신 valid_from ≤ cut연말 로 바꾼다.
    valid_from(정수 YYYYMMDD)은 cache/vf_<var>_<kind>_o<c>.npy (P_<kind>_index 행 순서와 같음, make_vf.py 산출)."""
    P = np.load(FCp/f'P_{kind}.npy'); ix = pd.read_parquet(FCp/f'P_{kind}_index.parquet')
    vf = np.load(HERE/'cache'/f'vf_{var}_{kind}_o{c}.npy'); assert len(vf) == len(ix), (len(vf), len(ix))
    rg = ix.rgst_date.astype(str).str[:8].astype(np.int64).values
    assert (vf >= rg).all(), 'valid_from 은 등록일 이상이어야 함'
    for d in (tr, te):
        for cut in d.cut.unique():
            idx = np.nonzero(d.cut.values == cut)[0]
            firms = pd.Index(d.firm.values[idx]).unique(); fi = pd.Series(np.arange(len(firms)), index=firms)
            m = (vf <= int(f'{cut}1231')) & ix.applicant_cd.isin(firms).values
            r = fi[ix.applicant_cd.values[m]].values.astype(np.int64); Pm = P[m].astype(np.float32); rec = ix.ay.values[m] > cut - 3
            nF = len(firms); SUM = np.zeros((nF, 50), np.float32); np.add.at(SUM, r, Pm)
            MX = np.zeros((nF, 50), np.float32); np.maximum.at(MX, r, Pm)
            N20 = np.zeros((nF, 50), np.float32); np.add.at(N20, r, (Pm > thr).astype(np.float32))
            REC = np.zeros((nF, 50), np.float32); np.add.at(REC, r[rec], Pm[rec])
            n = np.bincount(r, minlength=nF).astype(np.float32)
            f = fi[d.firm.values[idx]].values; k = d.k.values[idx]
            r3ext._put(d, f'{pre}_lsum', np.log1p(SUM[f, k]), idx); r3ext._put(d, f'{pre}_max', MX[f, k], idx); r3ext._put(d, f'{pre}_n20', np.log1p(N20[f, k]), idx)
            r3ext._put(d, f'{pre}_rec3', np.log1p(REC[f, k]), idx); r3ext._put(d, f'{pre}_n', np.log1p(n[f]), idx)
    return tr, te


def run(var, c):
    ec = json.loads(json.dumps(F.EC)); ec['ext'] = list(ec['ext'])
    if c == 2019:
        FCp = R3/'cache_final'/'o2019'
    else:
        assert c in (2008, 2009, 2010, 2011, 2012)
        FCp = R3/'cache'/f'o{c}'
    assert (FCp/'prep_tr.parquet').exists() and (FCp/'P_design.npy').exists()
    RD = HERE/'runs'/var
    o_des, o_um = r3ext.EXT['design'], r3ext.EXT['um']
    if var in ('S1', 'S2'):
        r3ext.EXT['design'] = lambda tr, te, cc, final=False: title_exposure_vf(tr, te, cc, 'design', 'ds', FCp, var)
        r3ext.EXT['um'] = lambda tr, te, cc, final=False: title_exposure_vf(tr, te, cc, 'um', 'um', FCp, var)
    elif var == 'S3':
        def z(fn, cols):
            def g(tr, te, cc, final=False):
                tr, te = fn(tr, te, cc, final)
                for d in (tr, te):
                    for x in cols: d[x] = np.float32(0.0)
                return tr, te
            return g
        r3ext.EXT['design'] = z(o_des, DS); r3ext.EXT['um'] = z(o_um, UM)
    elif var == 'S4':
        o_pr = r3ext.EXT['price']
        def zp(tr, te, cc, final=False):
            tr, te = o_pr(tr, te, cc, final)
            for d in (tr, te):
                for x in PR: d[x] = np.float32(np.nan)
            return tr, te
        r3ext.EXT['price'] = zp
    elif var == 'S5':
        import r2ext
        o_io = r2ext.EXT['io']
        def zi(tr, te, cc, final=False):
            tr, te = o_io(tr, te, cc, final)
            for d in (tr, te):
                for x in IO: d[x] = np.float32(np.nan)
            return tr, te
        r2ext.EXT['io'] = zi
    else:
        assert var == 'S0'
    print(f'[C10-T1] var={var} c={c} FCp={FCp} RD={RD}', flush=True)
    F.train_origin(c, FCp, 'prep_tr.parquet', RD, ec)


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]))
