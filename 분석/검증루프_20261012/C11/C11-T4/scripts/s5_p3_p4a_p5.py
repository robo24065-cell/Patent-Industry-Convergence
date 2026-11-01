# 후보 A: P3(b,c,d), P4(a) Jaccard, P5(a–d) 점검. 사용: python -I s5_p3_p4a_p5.py
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from lib11 import *

t_start = time.time()
res = {}
K = json.load(open(C + r'\ksic_list.json'))

# ---------- P4(a) Jaccard ----------
jac = {}
for arm in ['fi', 'hs']:
    inter = uni = 0; byv = {}
    for v in EVAL_Y:
        d = pd.read_parquet(C + rf'\u_A_{v}.parquet', columns=['cc', 'aid', 'k', 'risk_p', 'y_p', f'risk_{arm}', f'y_{arm}'])
        p = d.y_p.values & d.risk_p.values
        q = d[f'y_{arm}'].values & d[f'risk_{arm}'].values
        i, u = int((p & q).sum()), int((p | q).sum())
        byv[v] = dict(prov=int(p.sum()), pub=int(q.sum()), inter=i, jac=i / u, rec=i / max(p.sum(), 1))
        inter += i; uni += u
    jac[arm] = dict(jaccard=inter / uni, by_origin=byv, pass_a=bool(inter / uni < 0.70))
    out('P4a', arm, jac[arm]['jaccard'])
res['P4a'] = jac

# ---------- P3(b)/P5(d) 잠금 관측 가능성(월별 건수·출원인 결측만) ----------
con = con_duck()
q = f"""select cc, substr(cast(pub_date as varchar),1,6) m, count(*) n,
        avg(case when applicant is null or trim(applicant) in ('','nan','None','-','NULL') then 1 else 0 end) miss
        from read_parquet('{META}') where cc in ('CN','EP','US','JP')
        and ((pub_date >= '20240101' and pub_date < '20240701') or (pub_date >= '20250101' and pub_date < '20250701')) group by 1,2 order by 1,2"""
mo = con.execute(q).df()
p3b = {}
for cc, g in mo.groupby('cc'):
    n24 = int(g[g.m < '2025'].n.sum()); n25 = int(g[g.m >= '2025'].n.sum())
    mx = float(g[g.m >= '2025'].miss.max())
    p3b[cc] = dict(n2024H1=n24, n2025H1=n25, ratio=n25 / n24, max_miss_2025=mx, ok=bool(n25 >= 0.85 * n24 and mx <= 0.02))
res['P3b'] = p3b
out('P3b', p3b)
# P5(a) KSIC 구성 TVD 2025-H1 대 2024-H1 (집계)
q2 = f"""select cc, case when pub_date >= '20250101' then 'y25' else 'y24' end h, ksic, count(*) n from read_parquet('{META}')
        where cc in ('CN','EP') and ((pub_date >= '20240101' and pub_date < '20240701') or (pub_date >= '20250101' and pub_date < '20250701')) group by 1,2,3"""
kk = con.execute(q2).df()
tvd = {}
for cc, g in kk.groupby('cc'):
    pv = g.pivot_table(index='ksic', columns='h', values='n', fill_value=0)
    pv = pv / pv.sum()
    tvd[cc] = float(0.5 * (pv.y25 - pv.y24).abs().sum())
# 반분 기준선: 2024-H1 대 2023-H1
q3 = q2.replace("'20250101' then 'y25' else 'y24'", "'20240101' then 'y25' else 'y24'").replace("(pub_date >= '20240101' and pub_date < '20240701') or (pub_date >= '20250101' and pub_date < '20250701')",
                                                                                          "(pub_date >= '20230101' and pub_date < '20230701') or (pub_date >= '20240101' and pub_date < '20240701')")
kk3 = con.execute(q3).df()
tvd_ref = {}
for cc, g in kk3.groupby('cc'):
    pv = g.pivot_table(index='ksic', columns='h', values='n', fill_value=0); pv = pv / pv.sum()
    tvd_ref[cc] = float(0.5 * (pv.y25 - pv.y24).abs().sum())
L = json.load(open(C + r'\s0_log.json', encoding='utf-8'))
cont = {}
for cc in ['CN', 'EP']:
    by = {int(k): v for k, v in L[cc]['by_year'].items()}
    flags = [y for y in sorted(by) if y - 1 in by and y >= 2005 and (by[y] < 0.5 * by[y - 1] or by[y] > 2 * by[y - 1])]
    cont[cc] = dict(by_year=by, flagged=flags)
res['P5a'] = dict(tvd_2025H1_vs_2024H1=tvd, tvd_2024H1_vs_2023H1=tvd_ref, continuity=cont)
out('P5a', tvd, tvd_ref, {c: cont[c]['flagged'] for c in cont})

# ---------- 문서 단위 점검(P3c, P3d, P5b, P5c) ----------
p3c = {}; p3d = {}; p5c = {}; posrows = []
for cc in ['CN', 'EP']:
    d = pd.read_parquet(C + rf'\docs_{cc}.parquet', columns=['did', 'pub', 'ad', 'k', 'sc'])
    a = pd.read_parquet(C + rf'\app_{cc}.parquet')
    m = a.merge(d, on='did', how='inner')
    keys = pd.Index(m.key.unique()); m['aid'] = keys.get_indexer(m.key).astype(np.int64)   # s1_build.load()와 같은 순서
    ak = m[['aid', 'akey']].drop_duplicates()
    for v in EVAL_Y:
        t0 = ymd(v); wlo, whi = win(v)
        u = pd.read_parquet(C + rf'\u_A_{v}.parquet', columns=['cc', 'aid', 'k', 'risk_p', 'y_p'])
        u = u[(u.cc == cc) & u.risk_p & u.y_p][['aid', 'k']]
        wd = m[(m.pub >= wlo) & (m.pub <= whi)].merge(u, on=['aid', 'k'])      # 양성 문서
        n_bad = int((wd.ad.notna() & (wd.pub < wd.ad)).sum())
        p3c[f'{cc}_{v}'] = dict(pos_docs=len(wd), pub_before_appl=n_bad)
        h = m[(m.pub >= t0 - 30000) & (m.pub < t0) & m.sc.notna()][['aid', 'sc']].drop_duplicates()
        h['old'] = True
        z = wd.merge(h, on=['aid', 'sc'], how='left')
        z['old'] = z.old.fillna(False).astype(bool)
        pair_all_old = z.groupby(['aid', 'k']).old.all()
        p3d[f'{cc}_{v}'] = dict(doc_share_label_only=float(z.old.mean()), pair_share_all_label_only=float(pair_all_old.mean()),
                                pair_share_any_new_tech=float(1 - pair_all_old.mean()))
        # P5c 변형 키 거짓 사건
        lb = m[(m.pub >= t0 - 50000) & (m.pub < t0)][['akey', 'k']].drop_duplicates()
        lb['hit'] = True
        uu = u.merge(ak, on='aid', how='left').merge(lb, on=['akey', 'k'], how='left')
        hit = uu.groupby(['aid', 'k']).hit.any().fillna(False)
        p5c[f'{cc}_{v}'] = dict(pos=len(u), variant_false=int(hit.sum()), share=float(hit.mean()))
        u2 = u.copy(); u2['cc'] = cc; u2['v'] = v; posrows.append(u2)
        out(cc, v, p3c[f'{cc}_{v}'], p3d[f'{cc}_{v}'], p5c[f'{cc}_{v}'], round(time.time() - t_start, 1))
res['P3c'] = p3c; res['P3d'] = p3d; res['P5c'] = p5c
tc = sum(x['pos_docs'] for x in p3c.values()); tb = sum(x['pub_before_appl'] for x in p3c.values())
res['P3c_share'] = tb / max(tc, 1)
vf = sum(x['variant_false'] for x in p5c.values()); vp = sum(x['pos'] for x in p5c.values())
res['P5c_share'] = vf / vp
# P5b 분포
pos = pd.concat(posrows); pos['fid'] = pos.aid + np.where(pos.cc == 'EP', 10_000_000, 0)
top = pos.fid.value_counts()
risk_k = []
for v in EVAL_Y:
    u = pd.read_parquet(C + rf'\u_A_{v}.parquet', columns=['k', 'risk_p']); risk_k.append(u[u.risk_p].k.value_counts())
rk = pd.concat(risk_k, axis=1).sum(1)
pk = pos.k.value_counts()
top5 = pk.index[:5]
res['P5b'] = dict(top10_firm_share=float(top.iloc[:10].sum() / len(pos)), n_pos=len(pos), ksic_present=int(pk.size),
                  top5_ksic=[K[i] for i in top5], top5_share_pos=float(pk.iloc[:5].sum() / len(pos)), top5_share_risk=float(rk.reindex(top5).sum() / rk.sum()),
                  cc_share=pos.cc.value_counts(normalize=True).to_dict())
# P5c 결측(출원인 없음·짧은 키) 문서 비율
res['P5c_noapp'] = {cc: dict(docs=L[cc]['docs'], no_app=L[cc]['docs_no_app'], share=L[cc]['docs_no_app'] / L[cc]['docs']) for cc in ['CN', 'EP']}
res['P3_pass'] = bool(res['P3c_share'] <= 0.001 and all(p3b[c]['ok'] for c in ['CN', 'EP']))
res['P5_pass'] = bool(res['P5b']['top10_firm_share'] <= 0.50 and res['P5c_share'] <= 0.10)
json.dump(res, open(C + r'\checks_A.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=float)
out('P3 pass', res['P3_pass'], 'P5 pass', res['P5_pass'], 'P5b', res['P5b'], 'P5c', res['P5c_share'], round(time.time() - t_start, 1))
