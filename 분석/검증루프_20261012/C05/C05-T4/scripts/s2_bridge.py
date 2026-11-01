# s2: M1 업종→KSIC 다리 피처 br_gKR, br_gF4 (배정: prov / TXT / IPC). 기준일까지 보인 문서만. 잠금 원점 포함(레이블 무관).
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libC import *
T0 = time.time()
ca = load_ca(); nm2b6, _ = name_map(ca)
docs = load_docs(); docs['b6'] = docs.nname.map(nm2b6)
first_vis = docs[docs.b6.notna() & docs.vis.notna()].groupby('nname').vis.min()
lab = pd.read_parquet(os.path.join(B, '분석', '검증루프_20261012', 'C02', 'C02-T4', 'cache', 'kr_publab_src.parquet'))
lab['mg'] = lab.sg.str.split('/').str[0]; lab['sc'] = lab.sg.str[:4]
lab = lab.set_index('appl_no')
assert lab.index.is_unique
docs = docs.drop_duplicates('appl_no').set_index('appl_no')
docs['k_txt'] = lab.k_free.reindex(docs.index).astype(object)
print('k_free sample', docs.k_txt.dropna().head(3).tolist(), 'prov', docs.ksic.head(3).tolist(), 'txt agree', round(float((docs.k_txt.astype(str) == docs.ksic).mean()), 4), flush=True)
cube = pd.read_parquet(os.path.join(B, '공모전데이터', '작업데이터', 'country_year_cube.parquet'))
cube['ksic'] = cube['세세분류코드'].astype(str)
F4 = cube[cube['국가'].isin(['US', 'EP', 'CN', 'JP'])].groupby(['ksic', '출원연도'])['특허수'].sum().unstack(fill_value=0)


def growth(tab, t0):
    a = tab.reindex(columns=[t0 - 3, t0 - 2], fill_value=0).sum(axis=1); b = tab.reindex(columns=[t0 - 5, t0 - 4], fill_value=0).sum(axis=1)
    return np.log((a + 1) / (b + 1))


def hard_ipc(t0, ref):
    """FREE-IPC: 서브그룹(≥3) → 메인그룹 → 서브클래스의 k_free 최빈. 표는 출원연도 ≤ t0 이고 기준일까지 보인 문서로 구성."""
    vis_ok = docs.vis.reindex(lab.index)
    sub = lab[(lab.appl_year <= t0) & (vis_ok <= ref)]
    out = pd.Series(np.nan, index=lab.index, dtype=object); todo = pd.Series(True, index=lab.index)
    for l in ['sg', 'mg', 'sc']:
        s = sub[sub[l].notna()]
        tab = s.groupby([l, 'k_free']).size().reset_index(name='n'); tot = tab.groupby(l).n.sum()
        mode = tab.sort_values([l, 'n', 'k_free'], ascending=[True, False, True]).drop_duplicates(l).set_index(l).k_free
        v = lab[l].map(mode[mode.index.isin(tot[tot >= 3].index)])
        hit = todo & v.notna(); out[hit] = v[hit]; todo &= ~hit
    return out


units = pd.read_parquet(os.path.join(C, 'units.parquet'), columns=['key', 's0', 'ind6', 'ind2'])
OUT = []; LOG = {}
for s0 in DEV + [LOCK]:
    ref = REF[s0]; t0 = int(s0[:4])
    d0, _ = load_nps(s0, cols=('name', 'biz6', 'stat', 'form', 'indcd', 'ldong', 'adpt_dt', 'n_emp', 'n_new', 'n_loss'), tag='s2_S0')
    f0 = firm_table(d0, ref); fE = f0[f0.E >= 5]
    vis_names = first_vis[first_vis <= ref].index
    vis_keys = set(pd.Series(vis_names).map(nm2b6).astype(str) + '|' + pd.Series(vis_names))
    lk = fE[fE.index.isin(vis_keys)][['ind6', 'ind2']]
    dv = docs[docs.vis.notna() & (docs.vis <= ref)]
    assert dv.vis.max() <= ref
    dl = dv[dv.b6.notna()].copy(); dl['key'] = dl.b6 + '|' + dl.nname
    dl = dl[dl.key.isin(lk.index)]
    ipc = hard_ipc(t0, ref)
    labs = {'prov': dv.ksic, 'TXT': dv.k_txt, 'IPC': ipc.reindex(dv.index)}
    u = units[units.s0 == s0].copy()
    gF = growth(F4.reindex(columns=range(t0 - 5, t0 - 1), fill_value=0), t0)
    for A, Ls in labs.items():
        Ls = Ls.astype(object)
        # KR 성장(그 배정 라벨, 기준일까지 보인 문서의 출원연도)
        kr = pd.DataFrame({'k': Ls, 'ay': dv.ay}).dropna()
        gK = growth(kr.groupby(['k', 'ay']).size().unstack(fill_value=0), t0)
        x = pd.DataFrame({'key': dl.key, 'k': Ls.reindex(dl.index)}).dropna()
        fd = x.groupby(['key', 'k']).size(); fd = fd / fd.groupby(level=0).transform('sum')
        fd = fd.rename('w').reset_index().merge(lk, left_on='key', right_index=True)
        allk = sorted(set(fd.k) | set(gK.index) | set(gF.index))
        glob = fd.groupby('k').w.sum(); glob = (glob / glob.sum()).reindex(allk, fill_value=0)
        S2 = fd.groupby(['ind2', 'k']).w.sum().unstack(fill_value=0).reindex(columns=allk, fill_value=0)
        n2 = fd.groupby('ind2').key.nunique()
        sh2 = (S2.add(5 * glob, axis=1)).div(n2.reindex(S2.index) + 5, axis=0)
        S6 = fd.groupby(['ind6', 'k']).w.sum().unstack(fill_value=0).reindex(columns=allk, fill_value=0)
        n6 = fd.groupby('ind6').key.nunique()
        p6 = fd.drop_duplicates('ind6').set_index('ind6').ind2.reindex(S6.index)
        sh6 = (S6 + 5 * sh2.reindex(p6.values).fillna(0).values).div(n6.reindex(S6.index) + 5, axis=0)
        gk_v = gK.reindex(allk).fillna(0).values; gf_v = gF.reindex(allk).fillna(0).values
        b6K = pd.Series(sh6.values @ gk_v, index=sh6.index); b6F = pd.Series(sh6.values @ gf_v, index=sh6.index)
        b2K = pd.Series(sh2.values @ gk_v, index=sh2.index); b2F = pd.Series(sh2.values @ gf_v, index=sh2.index)
        gl_K, gl_F = float(glob.values @ gk_v), float(glob.values @ gf_v)
        u[f'br_gKR_{A}'] = u.ind6.map(b6K).fillna(u.ind2.map(b2K)).fillna(gl_K)
        u[f'br_gF4_{A}'] = u.ind6.map(b6F).fillna(u.ind2.map(b2F)).fillna(gl_F)
        LOG[f'{s0}_{A}'] = dict(n_link_firms=int(len(lk)), n_link_docs=int(len(x)), ind6_cov=round(float(u.ind6.isin(b6K.index).mean()), 4),
                                unassigned_share=round(float(Ls.isna().mean()), 4))
    OUT.append(u.drop(columns=['ind6', 'ind2']))
    print(s0, LOG[f'{s0}_prov'], LOG[f'{s0}_IPC'], round(time.time() - T0), flush=True)
pd.concat(OUT, ignore_index=True).to_parquet(os.path.join(C, 'bridge.parquet'), index=False)
json.dump(LOG, open(os.path.join(C, 's2_log.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('done', round(time.time() - T0))
