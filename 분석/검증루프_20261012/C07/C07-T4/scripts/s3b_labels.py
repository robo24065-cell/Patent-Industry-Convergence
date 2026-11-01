# 산업 배정(KSIC)에 의존하는 부분: 레이블 y(산업 간 이전)·주변형·경로 민감도, 산업 피처 (ii)–(v).
# 사용법: python s3b_labels.py prov | <공개배정 태그>  (공개 배정은 cache\pub_assign.parquet의 열)
import sys, time, json
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T4\scripts')
from lib7 import *
TAG = sys.argv[1] if len(sys.argv) > 1 else 'prov'
t = time.time(); R = {'tag': TAG}
def P(*a): print(*a, round(time.time() - t, 1), flush=True)
docs = pd.read_parquet(C + r'\docs.parquet', columns=['appl_no', 'ksic'])
if TAG == 'prov':
    kmap = docs.set_index('appl_no').ksic
else:
    pa = pd.read_parquet(C + r'\pub_assign.parquet', columns=['appl_no', TAG])
    kmap = pa.set_index('appl_no')[TAG].astype(str)
    kmap = kmap[kmap != '-1']
PI = pd.read_parquet(C + r'\PI.parquet')
PI['ksic'] = PI.appl_no.map(kmap)
R['PI_unmapped_share'] = float(PI.ksic.isna().mean())
PI = PI.dropna(subset=['ksic'])
# (name, ksic) 최소 날짜: 전체·A·A∪B
def mk(sub):
    return sub.groupby(['name', 'ksic']).date.min()
MK_all = mk(PI); MK_A = mk(PI[PI.path == 'A']); MK_AB = mk(PI[PI.path != 'C'])
MN_all = PI.groupby('name').date.min(); MN_A = PI[PI.path == 'A'].groupby('name').date.min(); MN_AB = PI[PI.path != 'C'].groupby('name').date.min()
PIu = PI.sort_values('date').drop_duplicates(['name', 'appl_no'])   # 이름×문서 최초 날짜(모든 경로)
P('PI ready', len(PIu))
cube = pd.read_parquet(CUBE); cube['ksic'] = cube['세세분류코드'].astype(str)
cf = cube[cube['국가'].isin(['US', 'EP', 'CN', 'JP'])].groupby(['ksic', '출원연도'])['특허수'].sum()
def kgrowth(ks, Y):
    out = {}
    for k in ks:
        try:
            a = np.mean([cf.get((k, y), 0) for y in (Y - 4, Y - 3, Y - 2)]); b = cf.get((k, Y - 2), 0)
            out[k] = np.log1p(b) - np.log1p(a)
        except Exception:
            out[k] = np.nan
    return out
def lookup(MK, MN, names, ks, t0s):
    d = pd.Series(MK.reindex(pd.MultiIndex.from_arrays([names, ks])).values)
    n = pd.Series(MN.reindex(names).values)
    nonempty = n.notna() & (n < t0s)
    has_c = d.notna() & (d < t0s)
    return nonempty.values, has_c.values
ypos = {}   # Y → 양성 (buyer, ksic) 표(피처 iv용)
for Y in list(range(2012, 2025)) + [LOCK]:
    s = pd.read_parquet(C + rf'\base_{Y}.parquet')
    s['ksic'] = s.applNo.map(kmap)
    t0s = ts(Y)
    out = pd.DataFrame({'rgst_no': s.rgst_no.values, 't0': Y, 'ksic': s.ksic.values})
    if Y < LOCK:
        assert (s.ev_date.dropna() < pd.Timestamp(Y + 1, 1, 1)).all() and (s.ev_date.dropna() < pd.Timestamp(EVENT_CUT)).all()
        ev = s.has_ev.values == 1
        indiv = s.buyer_type.isin([4, 6]).values
        names = s.buyer.fillna('').values; ks = s.ksic.fillna('NA').values
        for lab, MK, MN in [('', MK_all, MN_all), ('_A', MK_A, MN_A), ('_AB', MK_AB, MN_AB)]:
            ne, hc = lookup(MK, MN, names, ks, t0s)
            out['y' + lab] = (ev & ~indiv & ne & ~hc & s.ksic.notna().values).astype(np.int8)
            if lab == '':
                out['buyer_nonempty'] = (ev & ne).astype(np.int8)
        # 주변형: 최빈 KSIC와 다름
        bset = set(names[ev & ~indiv])
        sub = PIu[(PIu.date < t0s) & PIu.name.isin(bset)]
        cnt = sub.groupby(['name', 'ksic']).size().reset_index(name='c').sort_values(['name', 'c', 'ksic'], ascending=[True, False, True])
        mode = cnt.drop_duplicates('name').set_index('name').ksic
        bm = pd.Series(names).map(mode).values
        out['y_mode'] = (ev & ~indiv & out.buyer_nonempty.values.astype(bool) & (bm != ks) & s.ksic.notna().values).astype(np.int8)
        pos = out.y.values == 1
        ypos[Y] = pd.DataFrame({'buyer': names[pos], 'ksic': ks[pos]})
    # 산업 피처
    cur = s.cur.fillna('').values
    sub = PIu[PIu.date < t0s]
    cnt = sub.groupby(['name', 'ksic']).size()
    tot = sub.groupby('name').size()
    c_nc = pd.Series(cnt.reindex(pd.MultiIndex.from_arrays([cur, s.ksic.fillna('NA').values])).values).fillna(0).values
    c_t = pd.Series(tot.reindex(cur).values).values
    out['own_share_c'] = np.where(np.nan_to_num(c_t) > 0, c_nc / np.where(np.nan_to_num(c_t) > 0, c_t, 1), np.nan)
    out['own_nksic'] = pd.Series(sub.groupby('name').ksic.nunique().reindex(cur).values).fillna(0).values
    if Y >= 2015:
        prev = pd.concat([ypos[y] for y in (Y - 3, Y - 2, Y - 1)], ignore_index=True).drop_duplicates()
        ent = prev.groupby('ksic').buyer.nunique()
        alive_k = s.groupby('ksic').size()
        out['ksic_entry3'] = s.ksic.map(ent).fillna(0).values / s.ksic.map(alive_k).values
        kg = kgrowth(s.ksic.dropna().unique(), Y)
        out['kgrowth_f'] = s.ksic.map(kg).values
        out.to_parquet(C + rf'\lab_{TAG}_{Y}.parquet', index=False)
    if Y in DEV:
        R[f'{Y}'] = {'n': len(out), 'pos': int(out.y.sum()), 'pos_A': int(out.y_A.sum()), 'pos_AB': int(out.y_AB.sum()), 'pos_mode': int(out.y_mode.sum())}
    P(Y, R.get(f'{Y}', ''))
R['sec'] = round(time.time() - t, 1)
json.dump(R, open(C + rf'\s3b_{TAG}_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
