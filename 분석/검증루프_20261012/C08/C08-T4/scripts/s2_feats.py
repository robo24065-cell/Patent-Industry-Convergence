# C08-T4 s2: 원점 2015–2022 피처(M0, M1 제공 KSIC, 공개 배정판 KSIC 피처). 모든 피처는 날짜 < t0 기록만.
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from lib8 import *
T = time.time()
pat = pd.read_parquet(os.path.join(C, 'pat.parquet'))
pay = pd.read_parquet(os.path.join(C, 's1_pay.parquet'))
rgt = pd.read_parquet(os.path.join(C, 'rgt.parquet'))
own = pd.read_parquet(os.path.join(C, 'own.parquet'))
lic = pd.read_parquet(os.path.join(C, 'lic.parquet'))
app = pd.read_parquet(os.path.join(C, 'app.parquet'))
L = pd.read_parquet(os.path.join(C, 's1_labels.parquet'))
for df, c in ((pay, 'd8'), (rgt, 'd'), (own, 'd'), (lic, 'd')):
    assert (df[c].dropna() < CUT).all()
a1 = app.drop_duplicates('rgst_no').set_index('rgst_no')
pat['cd'] = pat.rgst_no.map(a1.applicantCd).fillna('NA')
pat['n_app'] = pat.rgst_no.map(app.groupby('rgst_no').size()).fillna(1)
pat['ksic'] = pat.ksic.astype(str)
# 공개 배정 원천
lab = pd.read_parquet(os.path.join(B, '분석', '검증루프_20261012', 'C02', 'C02-T4', 'cache', 'kr_publab_src.parquet'))
lab['mg'] = lab.sg.str.split('/').str[0]; lab['sc'] = lab.sg.str[:4]
lab = lab.set_index('appl_no')
ka = pd.read_parquet(os.path.join(B, '분석', '모델경쟁_20260927', '_ideas_R8', 'cache', 'kr_assign.parquet')).set_index('appl_no')
pat['L_TXT'] = pat.applNo.map(lab.k_free).astype(object)
pat['L_CONC'] = pat.applNo.map(ka.k_txt).astype(object)
LOG = {'applNo_in_lab': float(pat.applNo.isin(lab.index).mean()), 'applNo_in_kassign': float(pat.applNo.isin(ka.index).mean()),
       'agree_TXT': float((pat.L_TXT.astype(str) == pat.ksic).mean()), 'agree_CONC': float((pat.L_CONC.astype(str) == pat.ksic).mean())}
PR = pat.set_index('rgst_no')


def hard(Y, start, nmin):
    sub = lab[lab.appl_year <= Y - 1]
    lv = ['sg', 'mg', 'sc'][['sg', 'mg', 'sc'].index(start):]
    keys = pat.applNo.map(lab.sg)
    out = pd.Series(np.nan, index=pat.index, dtype=object); todo = pd.Series(True, index=pat.index)
    for l in lv:
        s = sub[sub[l].notna()]
        tab = s.groupby([l, 'k_free']).size().reset_index(name='n')
        tot = tab.groupby(l).n.sum()
        mode = tab.sort_values([l, 'n', 'k_free'], ascending=[True, False, True]).drop_duplicates(l).set_index(l).k_free
        mp = mode[mode.index.isin(tot[tot >= nmin].index)]
        kk = keys if l == 'sg' else (keys.str.split('/').str[0] if l == 'mg' else keys.str[:4])
        v = kk.map(mp); hit = todo & v.notna()
        out[hit] = v[hit]; todo &= ~hit
    return out


# 사건 표
ext = rgt[(rgt.nm == '소멸등록') & (rgt.rs == '등록료불납')][['rgst_no', 'd']].copy(); ext['d'] = ok8(ext.d); ext = ext.dropna()
ext['cd'] = ext.rgst_no.map(PR.cd)
tr = own[(own.seq > 0) & own.nm.notna()][['rgst_no', 'd']].copy(); tr['d'] = ok8(tr.d); tr = tr.dropna()
lic['d'] = ok8(lic.d); lic = lic.dropna(subset=['d'])
pay['cd'] = pay.rgst_no.map(PR.cd)
risk = {Y: L[L.origin == Y][['rgst_no', 'late7']] for Y in L.origin.unique()}


def kfeat(Y, labcol):
    """라벨 열(labcol: pat 열 이름 또는 Series) 기준 산업 피처 3종."""
    lb = PR[labcol] if isinstance(labcol, str) else labcol
    rates = {}
    for k in (1, 2, 3):
        r = risk[Y - k].copy(); r['g'] = r.rgst_no.map(lb).astype(str)
        rates[k] = r.groupby('g').late7.mean(); rates[f'n{k}'] = r.groupby('g').size()
        rates[f'all{k}'] = r.late7.mean()
    t0, tm = f'{Y}0101', f'{Y - 1}0101'
    e = ext[(ext.d >= tm) & (ext.d < t0)].copy(); e['g'] = e.rgst_no.map(lb).astype(str)
    ab = e.groupby('g').size() / rates['n1']
    ab_all = len(e) / len(risk[Y - 1])
    return rates, ab.fillna(0), ab_all


for Y in range(2015, 2023):
    t0, t1 = f'{Y}0101', f'{Y + 1}0101'
    t3 = f'{Y - 3}0101'
    F = risk[Y].rename(columns={'late7': 'y'}).copy()
    F['origin'] = Y
    P = PR.loc[F.rgst_no]
    F['ksic'] = P.ksic.values; F['cd'] = P.cd.values
    rd = P.rgstDate.values.astype(str)
    ry = np.array([int(s[:4]) for s in rd]); md = np.array([s[4:] if s[4:] != '0229' else '0228' for s in rd])
    F['age_y'] = ((pd.Timestamp(t0) - pd.to_datetime(pd.Series(rd), format='%Y%m%d')).dt.days / 365.25).values
    F['util'] = F.rgst_no.str[:2].eq('20').astype(np.int8)
    p = pay[pay.d8 < t0]
    cov = p.groupby('rgst_no').lastAnnl.max()
    F['nopay'] = (~F.rgst_no.isin(cov.index)).astype(np.int8)
    cv = F.rgst_no.map(cov).fillna(3).astype(int).values
    nd = pd.to_datetime(pd.Series([f'{a + b}{c}' for a, b, c in zip(ry, cv, md)]), format='%Y%m%d', errors='coerce')
    F['days_to_due'] = (nd - pd.Timestamp(t0)).dt.days.values
    F['overdue'] = (F.days_to_due < 0).astype(np.int8)
    kw = Y - ry + 1
    F['n_due_win'] = np.maximum(0, kw - cv)
    F['next_ann'] = cv + 1
    F['step'] = np.isin(cv + 1, [4, 7, 10, 13]).astype(np.int8)
    p4 = p[p.statAnnl >= 4]
    g = p4.groupby('rgst_no')
    F['pat_late_n'] = F.rgst_no.map(g.late7.sum()).fillna(0).values
    F['pat_pay_n'] = F.rgst_no.map(g.size()).fillna(0).values
    F['last_lag'] = F.rgst_no.map(p4.sort_values('d8').drop_duplicates('rgst_no', keep='last').set_index('rgst_no').lag).values
    F['max_span'] = F.rgst_no.map((p.lastAnnl - p.statAnnl + 1).groupby(p.rgst_no).max()).values
    p3 = p4[p4.d8 >= t3]
    go = p3.groupby('cd')
    F['own_late_r3'] = F.cd.map(go.late7.mean()).values
    F['own_pay3'] = np.log1p(F.cd.map(go.size()).fillna(0).values)
    F['own_aband3'] = np.log1p(F.cd.map(ext[(ext.d >= t3) & (ext.d < t0)].groupby('cd').size()).fillna(0).values)
    F['own_alive'] = np.log1p(F.cd.map(F.groupby('cd').size()).values)
    for c in ['own_late_r3', 'own_pay3', 'own_aband3', 'own_alive']:
        F.loc[F.cd == 'NA', c] = np.nan
    F['app_type'] = F.cd.str[0].map({'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}).fillna(0).astype(np.int8)
    F['n_app'] = P.n_app.values
    F['n_transfer'] = F.rgst_no.map(tr[tr.d < t0].groupby('rgst_no').size()).fillna(0).values
    F['n_lic'] = F.rgst_no.map(lic[lic.d < t0].groupby('rgst_no').size()).fillna(0).values
    F['ipc_sec'] = pd.Series(P.ipcCd.astype(str).str[0].values).map({c: i for i, c in enumerate('ABCDEFGH')}).fillna(8).astype(np.int8).values
    # 산업 피처: 제공 + 공개 배정
    labs = {'prov': 'ksic', 'TXT': 'L_TXT', 'CONC': 'L_CONC'}
    hs = hard(Y, 'sc', 1); labs['HSC1'] = pd.Series(hs.values, index=pat.rgst_no)
    hg = hard(Y, 'sg', 3); labs['HSG3'] = pd.Series(hg.values, index=pat.rgst_no)
    for nm, lc in labs.items():
        rates, ab, ab_all = kfeat(Y, lc)
        lb = (PR[lc] if isinstance(lc, str) else lc).reindex(F.rgst_no).astype(str).values
        gl = pd.Series(lb)
        k1 = gl.map(rates[1]).fillna(rates['all1']).values
        k23 = (gl.map(rates[2]).fillna(rates['all2']).values + gl.map(rates[3]).fillna(rates['all3']).values) / 2
        F[f'k_late1_{nm}'] = k1; F[f'k_trend_{nm}'] = k1 - k23; F[f'k_aband1_{nm}'] = gl.map(ab).fillna(ab_all).values
        F[f'lab_{nm}'] = lb
        if nm != 'prov':
            LOG[f'{Y}_agree_{nm}'] = float((gl.values == F.ksic.values).mean()); LOG[f'{Y}_unassigned_{nm}'] = float((gl.isin(['nan', 'None'])).mean())
    F.to_parquet(os.path.join(C, f'feat_{Y}.parquet'), index=False)
    print(Y, F.shape, int(F.y.sum()), round(time.time() - T, 1), flush=True)
LOG['sec'] = round(time.time() - T, 1)
json.dump(LOG, open(os.path.join(C, 's2_log.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(LOG, ensure_ascii=False, indent=1))
