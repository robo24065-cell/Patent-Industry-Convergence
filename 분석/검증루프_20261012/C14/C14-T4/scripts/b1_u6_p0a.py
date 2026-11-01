# C14-T4 b1: (B) U6×Eb·Ec P0(a), U6×Eg(CN) 규모, CN 구성 변화 점검 (PREREG §2.2–2.3)
# 입력(읽기만): C13 cache app_{CN,EP}.parquet(조직 출원인, 공개 2010–2024), docs_{CN,EP}.parquet. 날짜 가드 assert.
# 출력: cache/b1_by_origin.csv, cache/b1_out.json, cache/b1_cn_comp.csv
import sys, os, time, json, re
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib14 import *
t = time.time()
HAN = re.compile(r'[一-鿿]')
res = {'guard': {}, 'check_vs_c13': {}}
rows = []
comp = []
for cc in ['CN', 'EP']:
    res['guard'][f'app_{cc}'] = guard_parquet_pub(C13C + rf'\app_{cc}.parquet')
    res['guard'][f'docs_{cc}'] = guard_parquet_pub(C13C + rf'\docs_{cc}.parquet')
    app = pd.read_parquet(C13C + rf'\app_{cc}.parquet')
    docs = pd.read_parquet(C13C + rf'\docs_{cc}.parquet', columns=['appl_no', 'pub'])
    res['check_vs_c13'][cc] = {'app_rows': len(app), 'docs': len(docs), 'docs_with_org': int(app.appl_no.nunique())}
    # CN/EP 구성: 연도별 문서 수, 조직 문서 수, 개인 전용 문서 수, 조직 공동출원 문서 수
    docs['yr'] = docs.pub // 10000
    nk = app.groupby('appl_no').key.nunique()
    docs['org'] = docs.appl_no.isin(nk.index)
    docs['multi'] = docs.appl_no.map(nk).fillna(0) >= 2
    g = docs.groupby('yr').agg(n=('appl_no', 'size'), org=('org', 'sum'), multi=('multi', 'sum'))
    g['indiv_only'] = g.n - g.org; g['org_share'] = g.org / g.n; g['multi_share'] = g.multi / g.n; g['cc'] = cc
    comp.append(g.reset_index())
    # 쌍(순서 없음) 공동 문서
    m = app[app.appl_no.map(nk) >= 2][['appl_no', 'key', 'pub']]
    e = m.merge(m[['appl_no', 'key']], on='appl_no', suffixes=('', '_g'))
    e = e[e.key < e.key_g][['key', 'key_g', 'pub']].reset_index(drop=True)
    # 순서쌍(X1' 정의 재현용)
    eo = m.merge(m[['appl_no', 'key']], on='appl_no', suffixes=('', '_g'))
    eo = eo[eo.key != eo.key_g]
    first = eo.groupby(['key', 'key_g']).pub.min().rename('first').reset_index()
    del eo, m
    # 안정 패널 조직: 2013–2015와 2022–2024 양쪽에 공개 문서
    early = set(app[(app.pub >= 20130101) & (app.pub < 20160101)].key)
    late = set(app[(app.pub >= 20220101) & (app.pub < 20250101)].key)
    stable = early & late
    for t0 in range(2016, 2025):
        end = t0 * 10000 + 1231
        h5 = app[(app.pub > end - 50000) & (app.pub <= end)]
        mc = h5.groupby(['key', 'ksic']).size().rename('n').reset_index().sort_values(['key', 'n', 'ksic'], ascending=[True, False, True])
        modal = mc.drop_duplicates('key').set_index('key').ksic
        e3 = e[(e.pub > end - 30000) & (e.pub <= end)]
        n3 = e3.groupby(['key', 'key_g']).size().rename('n3')
        n1 = e3[e3.pub > end - 10000].groupby(['key', 'key_g']).size().rename('n1')
        R = pd.concat([n3, n1], axis=1).fillna(0).reset_index()
        R = R[(R.n3 >= 2) & (R.n1 >= 1)]
        R['mf'] = R.key.map(modal); R['mg'] = R.key_g.map(modal)
        R_all = len(R)
        R = R[R.mf.notna() & R.mg.notna() & (R.mf != R.mg)].copy()
        R['stable'] = R.key.isin(stable) & R.key_g.isin(stable)
        r = {'cc': cc, 't0': t0, 'active_pairs_all': int(R_all), 'risk': int(len(R)), 'risk_stable': int(R.stable.sum())}
        # X1' 규모(Eg 부분집합): 활성 조직(3년 ≥5) f의 창 안 첫 관계, 상대 최빈 KSIC 정의·상이
        cnt3 = app[(app.pub > end - 30000) & (app.pub <= end)].groupby('key').appl_no.nunique()
        act = cnt3[cnt3 >= 5].index
        r['x1p_risk'] = int(len(act))
        if t0 <= 2023:
            lo, hi = (t0 + 1) * 10000 + 101, (t0 + 1) * 10000 + 630
            ew = e[(e.pub >= lo) & (e.pub <= hi)].groupby(['key', 'key_g']).size().rename('nw')
            R = R.merge(ew, left_on=['key', 'key_g'], right_index=True, how='left'); R['nw'] = R.nw.fillna(0)
            R['y_eb'] = (R.nw == 0); R['y_ec'] = (R.nw >= 2) & (R.nw > R.n3 / 6)
            r.update({'eb_pos': int(R.y_eb.sum()), 'eb_neg': int((~R.y_eb).sum()), 'ec_pos': int(R.y_ec.sum()),
                      'eb_pos_stable': int((R.y_eb & R.stable).sum()), 'eb_neg_stable': int((~R.y_eb & R.stable).sum()),
                      'ec_pos_stable': int((R.y_ec & R.stable).sum())})
            nw_ = first[(first['first'] >= lo) & (first['first'] <= hi) & first.key.isin(act)].copy()
            nw_['mf'] = nw_.key.map(modal); nw_['mg'] = nw_.key_g.map(modal)
            st = nw_[nw_.mg.notna() & (nw_.mg != nw_.mf)].copy()
            st['foreign'] = ~st.key_g.astype(str).str.contains(HAN) if cc == 'CN' else False
            r.update({'x1p_pos_strict': int(st.key.nunique()), 'eg_pos_firms': int(st[st.foreign].key.nunique()) if cc == 'CN' else None,
                      'eg_pairs': int(st.foreign.sum()) if cc == 'CN' else None,
                      'eg_foreign_partner_share_pairs': float(st.foreign.mean()) if (cc == 'CN' and len(st)) else None})
        rows.append(r); out(r, round(time.time() - t, 1))
    del e, first, app, docs
df = pd.DataFrame(rows)
df.to_csv(C + r'\b1_by_origin.csv', index=False, encoding='utf-8-sig')
pd.concat(comp).to_csv(C + r'\b1_cn_comp.csv', index=False, encoding='utf-8-sig')


def lock_exp(col_pos, col_den='risk', risk_col='risk'):
    tot = 0.0; per = {}
    for cc in ['CN', 'EP']:
        d = df[(df.cc == cc) & df.t0.isin([2022, 2023])]
        rate = float((d[col_pos] / d[col_den]).mean())
        rk = int(df[(df.cc == cc) & (df.t0 == 2024)][risk_col].iloc[0])
        per[cc] = rk * rate; tot += rk * rate
    return tot, per


dev = df[df.t0 <= 2023]
out_ = {}
for nm, pc in [('Eb_pos', 'eb_pos'), ('Eb_neg', 'eb_neg'), ('Ec_pos', 'ec_pos')]:
    tot, per = lock_exp(pc)
    out_[nm] = {'dev_sum': int(dev[pc].sum()), 'lock_exp': tot, 'lock_exp_by_cc': per}
for nm, pc in [('Eb_pos_stable', 'eb_pos_stable'), ('Eb_neg_stable', 'eb_neg_stable'), ('Ec_pos_stable', 'ec_pos_stable')]:
    tot, per = lock_exp(pc, 'risk_stable', 'risk_stable')
    out_[nm] = {'dev_sum': int(dev[pc].sum()), 'lock_exp': tot, 'lock_exp_by_cc': per}
cn = df[df.cc == 'CN']
d = cn[cn.t0.isin([2022, 2023])]
out_['Eg_CN'] = {'dev_sum_firms': int(cn.eg_pos_firms.fillna(0).sum()),
                 'lock_exp': float(int(cn[cn.t0 == 2024].x1p_risk.iloc[0]) * (d.eg_pos_firms / d.x1p_risk).mean()),
                 'foreign_share_pairs_by_t0': dict(zip(cn.t0.astype(int), cn.eg_foreign_partner_share_pairs))}
out_['P0a'] = {
    'U6xEb': bool(out_['Eb_pos']['dev_sum'] >= 1000 and out_['Eb_pos']['lock_exp'] >= 1000 and out_['Eb_neg']['dev_sum'] >= 1000 and out_['Eb_neg']['lock_exp'] >= 1000),
    'U6xEc': bool(out_['Ec_pos']['dev_sum'] >= 1000 and out_['Ec_pos']['lock_exp'] >= 1000),
    'U6xEg_CN': bool(out_['Eg_CN']['dev_sum_firms'] >= 1000 and out_['Eg_CN']['lock_exp'] >= 1000),
    'U6xEb_stable': bool(out_['Eb_pos_stable']['lock_exp'] >= 1000 and out_['Eb_neg_stable']['lock_exp'] >= 1000),
    'U6xEc_stable': bool(out_['Ec_pos_stable']['lock_exp'] >= 1000)}
res.update(out_)
res['secs'] = round(time.time() - t, 1)
json.dump(res, open(C + r'\b1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
out(json.dumps(res, ensure_ascii=False, indent=1, default=float))
