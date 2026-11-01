# P5(b)(c) 편향 점검 + data_check용 연결 편향(활성 출원인 문서 대 나머지). 출력 cache/checks.json
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib12 import *

t = time.time()
res = {}
SCL = json.load(open(C + r'\sc_list.json'))
# ---- P5(b)(c) ----
U = []
for cc in CCS:
    for Y in ALL_Y:
        u = pd.read_parquet(C + rf'\u_{cc}_{Y}.parquet', columns=['cc', 'aid', 's', 'Y', 'y', 'w', 'vk'])
        U.append(u)
U = pd.concat(U, ignore_index=True)
for nm, sub in [('dev_all', U), ('eval', U[U.Y >= 2020])]:
    p = sub[sub.y == 1]
    fc = p.groupby(['cc', 'aid']).size().sort_values(ascending=False)
    sp_ = p.groupby('s').size().sort_values(ascending=False)
    rk = sub.groupby('s').w.sum()
    top5 = sp_.index[:5]
    res[nm] = {'n_pos': int(len(p)), 'top10_firm_share': float(fc.iloc[:10].sum() / len(p)),
               'top5_sc': [SCL[i] for i in top5], 'top5_sc_pos_share': float(sp_.iloc[:5].sum() / len(p)),
               'top5_sc_risk_share': float(rk.loc[top5].sum() / rk.sum()),
               'cc_pos_share': {k: float(v) for k, v in (p.groupby('cc').size() / len(p)).items()},
               'vk_rate': float(p.vk.mean()), 'vk_rate_by_ccY': {f'{a}_{b}': float(v) for (a, b), v in p.groupby(['cc', 'Y']).vk.mean().items()}}
    out(nm, json.dumps(res[nm])[:500])
res['P5b_fail'] = bool(res['dev_all']['top10_firm_share'] > 0.5); res['P5b_warn'] = bool(res['dev_all']['top10_firm_share'] > 0.3)
res['P5c_fail'] = bool(res['dev_all']['vk_rate'] > 0.10)
# ---- 연결 편향: 활성 출원인(2024 원점) 문서 대 그 밖 문서 ----
K = json.load(open(C + r'\ksic_list.json'))
for cc in CCS:
    d = pd.read_parquet(C + rf'\docs_{cc}.parquet', columns=['did', 'pub', 'k', 'sc'])
    a = pd.read_parquet(C + rf'\app_{cc}.parquet', columns=['did', 'key'])
    a['pub'] = d.pub.values[a.did.values]
    t0 = ymd(2024)
    w3 = (a.pub >= addy(t0, -3)) & (a.pub < t0)
    n3 = a[w3].groupby('key').size(); n1 = a[(a.pub >= addy(t0, -1)) & (a.pub < t0)].groupby('key').size()
    act = set(n3[n3 >= 10].index) & set(n1.index)
    dd = d[(d.pub >= addy(t0, -3)) & (d.pub < t0)]
    ad = set(a[w3 & a.key.isin(act)].did)
    has_app = set(a.did)
    dd = dd.assign(grp=np.where(dd.did.isin(ad), 'active', np.where(dd.did.isin(has_app), 'inactive', 'noapp')))
    r = {'docs_3y': int(len(dd)), 'share': {k: float(v) for k, v in dd.grp.value_counts(normalize=True).items()}}
    ka = dd[dd.grp == 'active'].k.value_counts(normalize=True); ki = dd[dd.grp != 'active'].k.value_counts(normalize=True)
    r['ksic_tvd_active_vs_rest'] = float(0.5 * ka.subtract(ki, fill_value=0).abs().sum())
    sa = dd[dd.grp == 'active'].sc.value_counts(normalize=True); si = dd[dd.grp != 'active'].sc.value_counts(normalize=True)
    r['sc_tvd_active_vs_rest'] = float(0.5 * sa.subtract(si, fill_value=0).abs().sum())
    r['sc_missing'] = {g: float(x.sc.isna().mean()) for g, x in dd.groupby('grp')}
    r['top3_ksic_active'] = {K[i]: float(v) for i, v in ka.iloc[:3].items()}
    r['top3_ksic_rest'] = {K[i]: float(v) for i, v in ki.iloc[:3].items()}
    res['link_' + cc] = r
    out(cc, json.dumps(r)[:600])
res['secs'] = round(time.time() - t, 1)
json.dump(res, open(C + r'\checks.json', 'w'), indent=1)
out('done', res['secs'])
