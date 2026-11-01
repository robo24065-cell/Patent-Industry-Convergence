# -*- coding: utf-8 -*-
"""T1 R7 대체 시험 S_IPC: '목표 KSIC의 IPC 기술 프로파일'을 제공 KSIC 라벨(범위안 특허 집계) 대신 공식 IPC-KSIC 연계표(2013)의 산업별 IPC 건수로.
연계표는 소분류(3자리) 해상도이므로 같은 소분류의 5자리 코드는 같은 프로파일을 받는다. 기업 쪽 프로파일(자사 출원 IPC)은 그대로.
대상 피처: ipc_fit·ipc_fit_rank(r2ext ipcfit, 서브클래스), ipc_mix·ipc_max(r2ext ipc_sim의 KSIC×KSIC 유사도), ipcx_fit·ipcx_fit_rank(r3ext2, 서브클래스),
           ipcm_fit·ipcm_fit_rank(r3ext3, 메인그룹). 동결 함수는 수정하지 않고 이 프로세스 안에서만 대체 함수로 바꿔 끼운다(monkeypatch)."""
import re, sys, pathlib
import numpy as np, pandas as pd, scipy.sparse as sp
HERE = pathlib.Path(__file__).resolve().parent; T1 = HERE.parent; R3 = T1/'R3'
sys.path.insert(0, str(T1.parent/'N9_정부RnD과제_특허성과예측'/'ESSENTIAL_R7'))
import r7_concord as RC

_C = {}
def conc_rows():
    if 'cc' in _C: return _C['cc']
    cc = pd.read_excel(RC.CCX, sheet_name='전체연계표', dtype=str); cc.columns = ['ind', 'ipc', 'cnt']
    cc['cnt'] = pd.to_numeric(cc.cnt, errors='coerce').fillna(0); cc = cc[cc.ind.str.match(r'^[A-Z]\d{4}$', na=False)].copy()
    cc['grp'] = cc.ind.map(RC._grp_of); cc = cc[cc.grp.isin(RC.G50)]
    s = cc.ipc.str.replace(r'\s+', '', regex=True).str.upper()
    cc['sc'] = s.str[:4]; cc['mg'] = s.str.split('/').str[0]
    _C['cc'] = cc; return cc


def profile(K, keys, level):
    """50×len(keys) 프로파일(행 L2 정규화). keys는 피처 함수가 쓰는 IPC 키 목록(공백 차이만 있는 중복 키는 같은 값을 받음)."""
    cc = conc_rows()
    g = cc.groupby([level, 'grp']).cnt.sum()
    lut = {}
    for (kk, gg), w in g.items(): lut.setdefault(kk, {})[gg] = w
    GP = {gg: np.zeros(len(keys)) for gg in RC.G50}
    for j, key in enumerate(keys):
        for gg, w in lut.get(re.sub(r'\s+', '', str(key)).upper(), {}).items(): GP[gg][j] = w
    KP = np.stack([GP[k[:3]] for k in K]); return KP / np.maximum(np.linalg.norm(KP, axis=1, keepdims=True), 1e-12)


def patch():
    import r2ext, r3ext2, r3ext3
    from r3ext2 import _put, ipcx_src, ipcx_src_mg
    K = r2ext.K50()

    # (1) ipc_sim: KSIC×KSIC 유사도
    def ipc_sim_alt(cut):
        cc = conc_rows(); scs = sorted(cc.sc.unique()); KP = profile(K, scs, 'sc')
        S = (KP @ KP.T).astype(np.float32); np.fill_diagonal(S, 0); return S
    r2ext.ipc_sim = ipc_sim_alt

    # (2) ipcfit (등록 범위안 특허 메인 IPC 서브클래스)
    def ext_ipcfit_alt(tr, te, c, final=False):
        ins = r2ext.lib.load()['ins']
        for d in (tr, te):
            out = np.full(len(d), np.nan, np.float32); top = np.full(len(d), np.nan, np.float32)
            for cut in d.cut.unique():
                v = ins[(ins.ay <= cut) & (ins.rgst_date <= f'{cut}1231')].drop_duplicates(['applicant_cd', 'rgst_no'])
                v = v.assign(sc=v.mipc.fillna('').astype(str).str[:4]); v = v[v.sc.str.len() >= 4]
                scs = pd.Index(v.sc.unique()); si = pd.Series(np.arange(len(scs)), index=scs)
                KP = profile(K, list(scs), 'sc').astype(np.float32)
                idx = np.nonzero(d.cut.values == cut)[0]; firms = pd.Index(d.firm.values[idx]).unique(); fi = pd.Series(np.arange(len(firms)), index=firms)
                g = v[v.applicant_cd.isin(firms)]
                FP = np.zeros((len(firms), len(scs)), np.float32); np.add.at(FP, (fi[g.applicant_cd].values, si[g.sc].values), 1)
                has = FP.sum(1) > 0; FP = FP / np.maximum(np.linalg.norm(FP, axis=1, keepdims=True), 1e-9)
                C = FP @ KP.T; f = fi[d.firm.values[idx]].values; k = d.k.values[idx]
                out[idx] = np.where(has[f], C[f, k], np.nan); rk = (-C).argsort(1).argsort(1); top[idx] = np.where(has[f], rk[f, k], np.nan)
            d['ipc_fit'] = out; d['ipc_fit_rank'] = top
        return tr, te
    r2ext.EXT['ipcfit'] = ext_ipcfit_alt

    # (3) ipcx: 서브클래스 이름 목록을 ipcx_src와 같은 방식으로 재구성
    def scs_names():
        if 'scs' in _C: return _C['scs']
        x = pd.read_parquet(R3/'cache'/'src_kr_ipc.parquet', columns=['appl_no', 'ksic', 'ipc_all'])
        x = x[x.ksic.astype(str).isin(K)]
        e = x[['appl_no', 'ipc_all']].assign(sc=x.ipc_all.fillna('').str.split('|')).explode('sc')
        e['sc'] = e.sc.str.strip().str[:4]; e = e[e.sc.str.len() == 4]
        _C['scs'] = sorted(e.sc.unique()); return _C['scs']

    def fit_alt(tr, te, X, ea, ej, ew, nkeys, KP, names):
        nA = len(X['ay']); E = sp.csr_matrix((ew, (ea, ej)), shape=(nA, nkeys))
        for d in (tr, te):
            for cut in d.cut.unique():
                idx = np.nonzero(d.cut.values == cut)[0]; k = d.k.values[idx]; ce = int(f'{cut}1231')
                va = (X['ay'] <= cut) & (X['vis'] <= ce)
                firms = pd.Index(d.firm.values[idx]).unique(); fi = pd.Series(np.arange(len(firms)), index=firms)
                sel = pd.Index(X['fa_firm']).isin(firms) & va[X['fa_a']]
                A = sp.csr_matrix((np.ones(int(sel.sum())), (fi[X['fa_firm'][sel]].values, X['fa_a'][sel])), shape=(len(firms), nA))
                FP = (A @ E).tocsr(); nrm = np.sqrt(np.asarray(FP.multiply(FP).sum(1)).ravel())
                C = np.asarray(FP @ KP.T) / np.maximum(nrm[:, None], 1e-12)
                f = fi[d.firm.values[idx]].values; has = nrm[f] > 0
                _put(d, names[0], np.where(has, C[f, k], np.nan), idx)
                rk = (-C).argsort(1).argsort(1); _put(d, names[1], np.where(has, rk[f, k], np.nan), idx)
        return tr, te

    def ext_ipcx_alt(tr, te, c, final=False):
        X = ipcx_src(); scs = scs_names(); assert len(scs) == X['nsc'], (len(scs), X['nsc'])
        return fit_alt(tr, te, X, X['ea'], X['ej'], X['ew'], X['nsc'], profile(K, scs, 'sc'), ('ipcx_fit', 'ipcx_fit_rank'))
    r3ext2.EXT['ipcx'] = ext_ipcx_alt

    def ext_ipcm_alt(tr, te, c, final=False):
        X = ipcx_src(); Y = ipcx_src_mg(); mgs = list(Y['mgs'])
        return fit_alt(tr, te, X, Y['ea'], Y['ej'], Y['ew'], len(mgs), profile(K, mgs, 'mg'), ('ipcm_fit', 'ipcm_fit_rank'))
    r3ext3.EXT['ipcm'] = ext_ipcm_alt
    print('[S_IPC] 연계표 KSIC IPC 프로파일로 ipcfit·ipcsim·ipcx·ipcm 대체', flush=True)
