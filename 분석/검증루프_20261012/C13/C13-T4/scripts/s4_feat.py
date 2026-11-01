# C13-T4 s4: X1' 원점별 단위·결과·피처(PREREG 변경 2). 인자: 체계 prov | free | hsc | null<r>
# 출력 cache/u_<sys>_<t0>.parquet (t0 2013–2023). 공개 < 2025 자료만.
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from lib13 import *

SYS = sys.argv[1]
T0S = list(range(2013, 2024))
t_start = time.time()


def load():
    A, D = [], []
    for cc in ['CN', 'EP']:
        a = pd.read_parquet(f'{C}/app_{cc}.parquet'); a['u'] = cc + ':' + a.key; a['cc'] = cc
        d = pd.read_parquet(f'{C}/docs_{cc}.parquet', columns=['appl_no', 'pub', 'ksic', 'main_ipc'])
        d['sc'], d['mg'], d['sg'] = parse_ipc(d.main_ipc); d['cc'] = cc
        d['dk'] = cc + ':' + d.appl_no
        a['dk'] = cc + ':' + a.appl_no
        A.append(a[['dk', 'u', 'cc', 'pub']]); D.append(d[['dk', 'cc', 'pub', 'ksic', 'sc', 'mg', 'sg']])
    A = pd.concat(A, ignore_index=True); D = pd.concat(D, ignore_index=True)
    assert A.pub.max() < CUT and D.pub.max() < CUT
    return A, D


A, D = load()
D = D[D.dk.isin(A.dk)].reset_index(drop=True)   # 조직 출원인이 있는 문서만
# 문서별 조직 수, 쌍 첫 공동출원일
nor = A.groupby('dk').u.nunique()
A['multi'] = A.dk.map(nor).values >= 2
m = A[A.multi][['dk', 'u', 'pub']]
E = m.merge(m[['dk', 'u']], on='dk', suffixes=('', '_g'))
E = E[E.u != E.u_g]
FIRST = E.groupby(['u', 'u_g']).pub.min().rename('first').reset_index()
del m, E

# ---- 라벨 ----
kr = None
if SYS in ('free', 'hsc'):
    import duckdb
    con = con_duck()
    kr = con.execute(f"select appl_no, cast(pub_date as integer) pub, main_ipc from read_parquet('{META}') where cc='KR' and pub_date >= '19000101' and pub_date < '20250101' and length(pub_date)=8").fetchdf()
    kr = kr.drop_duplicates('appl_no')
    tf = con.execute(f"select appl_no, k_free from read_parquet('{TXTFREE}')").fetchdf().drop_duplicates('appl_no').set_index('appl_no').k_free
    kr['kt'] = kr.appl_no.map(tf); kr['sc'], kr['mg'], kr['sg'] = parse_ipc(kr.main_ipc)
    assert kr.pub.max() < CUT
if SYS.startswith('null'):
    r = int(SYS[4:])
    rng = np.random.default_rng(SEED + 1000 * r)
    D['yr'] = D.pub // 10000
    grp = D.cc + '|' + D.yr.astype(str) + '|' + D.sc.fillna('NA')
    # 같은 칸 안에서 제공 라벨을 무작위로 섞는다 = 칸 경험 분포에서 비복원 재추출(칸 분포 정확 보존)
    order = np.lexsort((rng.random(len(D)), grp.values))
    lab = D.ksic.values.copy()
    g_sorted = grp.values[order]
    # 칸별 순열: 정렬된 순서 안에서 라벨을 칸 단위로 섞는다
    perm_lab = np.empty(len(D), dtype=object)
    starts = np.r_[0, np.flatnonzero(g_sorted[1:] != g_sorted[:-1]) + 1, len(D)]
    for s0, s1 in zip(starts[:-1], starts[1:]):
        idx = order[s0:s1]
        perm_lab[idx] = lab[rng.permutation(idx)]
    D['lab_null'] = perm_lab
    assert (D.groupby(grp).lab_null.apply(lambda s: tuple(sorted(s))) == D.groupby(grp).ksic.apply(lambda s: tuple(sorted(s)))).all()

# ---- 외부: 한국 대중 수입 HS6 → KSIC ----
hm = pd.read_csv(B + r'\분석\fusion\ksic_hs6_map.csv', dtype={'k5': str, 'hs6': str})
hi = pd.read_csv(B + r'\분석\fusion\hs6_import_yearly.csv', dtype={'hs6': str})
hi = hi[hi.partner == 'CN']
IMP = hi.merge(hm[['k5', 'hs6']].drop_duplicates(), on='hs6').groupby(['k5', 'year']).value.sum()


def imp_growth(k, t0):
    try:
        a1 = IMP.get((k, t0 - 1), np.nan); a2 = IMP.get((k, t0 - 2), np.nan)
        return np.log((a1 + 1) / (a2 + 1)) if a1 == a1 and a2 == a2 else np.nan
    except Exception:
        return np.nan


def labels_for(t0, sub):
    if SYS == 'prov':
        return sub.ksic
    if SYS.startswith('null'):
        return sub.lab_null
    tabs = build_pub_tables(kr, t0 * 10000 + 1231 + 1)
    return apply_pub(sub, tabs, 'FREE-IPC' if SYS == 'free' else 'H-sc-1')


Y = {}       # t0 -> DataFrame(u, modal, y)
MOD = {}
rows_log = []
for t0 in T0S:
    end = t0 * 10000 + 1231
    a5 = A[(A.pub > end - 50000) & (A.pub <= end)]
    d5 = D[(D.pub > end - 50000) & (D.pub <= end)].copy()
    d5['lab'] = labels_for(t0, d5).values
    a5 = a5.merge(d5[['dk', 'lab', 'sc']], on='dk', how='left')
    cnt3 = A[(A.pub > end - 30000) & (A.pub <= end)].groupby('u').dk.nunique()
    act = cnt3[cnt3 >= 5].index
    lc = a5[a5.lab.notna()].groupby(['u', 'lab']).size().rename('n').reset_index().sort_values(['u', 'n', 'lab'], ascending=[True, False, True])
    modal = lc.drop_duplicates('u').set_index('u').lab
    MOD[t0] = modal
    lo, hi_ = (t0 + 1) * 10000 + 101, (t0 + 1) * 10000 + 630
    nw = FIRST[(FIRST['first'] >= lo) & (FIRST['first'] <= hi_) & FIRST.u.isin(act)].copy()
    nw['mf'] = nw.u.map(modal); nw['mg'] = nw.u_g.map(modal)
    pos = set(nw[nw.mg.notna() & nw.mf.notna() & (nw.mg != nw.mf)].u)
    U = pd.DataFrame({'u': act}); U['cc'] = U.u.str[:2]; U['modal'] = U.u.map(modal); U['y'] = U.u.isin(pos).astype(np.int8)
    Y[t0] = U
    if t0 < 2016:
        continue
    # 기준 피처(라벨 무관)
    U = U.set_index('u')
    a3 = A[(A.pub > end - 30000) & (A.pub <= end)]; a1 = A[(A.pub > end - 10000) & (A.pub <= end)]
    U['b_n3'] = np.log1p(a3.groupby('u').dk.nunique()).reindex(U.index).fillna(0)
    U['b_n1'] = np.log1p(a1.groupby('u').dk.nunique()).reindex(U.index).fillna(0)
    U['b_gr'] = np.log((np.expm1(U.b_n1) + 1) / (np.expm1(U.b_n3) / 3 + 1))
    fp = FIRST[(FIRST['first'] <= end) & FIRST.u.isin(U.index)]
    U['b_npart'] = np.log1p(fp.groupby('u').size()).reindex(U.index).fillna(0)
    U['b_new3'] = np.log1p(fp[fp['first'] > end - 30000].groupby('u').size()).reindex(U.index).fillna(0)
    U['b_new1'] = np.log1p(fp[fp['first'] > end - 10000].groupby('u').size()).reindex(U.index).fillna(0)
    U['b_cosh'] = a3.groupby('u').multi.mean().reindex(U.index).fillna(0)
    U['b_nsc'] = np.log1p(a5[a5.sc.notna()].groupby('u').sc.nunique()).reindex(U.index).fillna(0)
    U['b_iscn'] = (U.cc == 'CN').astype(np.int8)
    # 체계 피처
    lcu = lc[lc.u.isin(U.index)]
    tot = lcu.groupby('u').n.sum()
    p = lcu.n / lcu.u.map(tot)
    U['x_nk'] = np.log1p(lcu.groupby('u').size()).reindex(U.index).fillna(0)
    U['x_ent'] = (-(p * np.log(p))).groupby(lcu.u).sum().reindex(U.index).fillna(0)
    U['x_msh'] = (lcu.groupby('u').n.max() / tot).reindex(U.index).fillna(0)
    fp2 = fp.copy(); fp2['mf'] = fp2.u.map(modal); fp2['mg'] = fp2.u_g.map(modal)
    fpd = fp2[fp2.mg.notna() & fp2.mf.notna()]
    xs = (fpd.mg != fpd.mf).groupby(fpd.u).mean()
    U['x_xpart'] = xs.reindex(U.index).fillna(-1); U['x_xpart_na'] = U.index.isin(xs.index).astype(np.int8) ^ 1
    U['x_xnew3'] = np.log1p(fpd[(fpd['first'] > end - 30000) & (fpd.mg != fpd.mf)].groupby('u').size()).reindex(U.index).fillna(0)
    past = pd.concat([Y[t] for t in (t0 - 3, t0 - 2, t0 - 1)])
    ir = past[past.modal.notna()].groupby('modal').y.agg(['sum', 'size'])
    rate = (ir['sum'] + 1) / (ir['size'] + 50)
    U['x_indr'] = U.modal.map(rate).astype(float).fillna(float(past.y.mean()))
    U['x_indn'] = np.log1p(U.modal.map(U.modal.value_counts())).astype(float).fillna(0)
    U['x_imp'] = [imp_growth(k, t0) if isinstance(k, str) else np.nan for k in U.modal]
    U['x_imp_na'] = U.x_imp.isna().astype(np.int8); U['x_imp'] = U.x_imp.fillna(0)
    U['x_modal_na'] = U.modal.isna().astype(np.int8)
    U = U.reset_index()
    U.to_parquet(f'{C}/u_{SYS}_{t0}.parquet', index=False)
    rows_log.append({'t0': t0, 'n': len(U), 'pos': int(U.y.sum()), 'modal_na': float(U.modal.isna().mean()), 'sec': round(time.time() - t_start, 1)})
    out(SYS, rows_log[-1])
json.dump(rows_log, open(f'{C}/s4_log_{SYS}.json', 'w'), indent=1)
