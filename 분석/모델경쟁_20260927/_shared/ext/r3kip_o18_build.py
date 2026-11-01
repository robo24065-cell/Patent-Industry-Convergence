# r3kip_ 재시도 확장: O18 표본 250건 파싱 + 시점 부여(r3kip_parse.py와 같은 규칙) + as-of 피인용 + 선행검정(O18 단독, O16+O18 합동)
# 원본 DB는 mode=ro. 최종시험 구간 없음(T4는 3라운드 최종시험 없음, 모든 결과 <=2020-12).
import os, re, json, pandas as pd, numpy as np, duckdb, sqlite3, warnings
warnings.filterwarnings('ignore')
from scipy import stats
from statsmodels.discrete.conditional_models import ConditionalLogit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
D = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(D, 'r3kip_raw')
B = os.path.dirname(os.path.dirname(D)); ROOT = os.path.dirname(os.path.dirname(B))
S = pd.read_parquet(os.path.join(D, 'r3kip_o18_sample.parquet'))


def items(body, tag):
    return [{k: v.strip() for k, v in re.findall(r'<(\w+)>(.*?)</\1>', blk, re.S)}
            for blk in re.findall(rf'<{tag}>(.*?)</{tag}>', body, re.S)]


ci = []; nrows = []
for a in S.applNo:
    it = items(open(os.path.join(RAW, f'citing_{a}.xml'), encoding='utf-8').read(), 'citingInfo'); nrows.append(len(it))
    for r in it:
        r['q'] = a; ci.append(r)
S['n_rows'] = nrows
ci = pd.DataFrame(ci).rename(columns={'StandardCitationApplicationNumber': 'cited_appl', 'ApplicationNumber': 'citing_appl',
                                      'CitationLiteratureTypeCode': 'ctype', 'CitationLiteratureTypeCodeName': 'ctype_name',
                                      'StandardStatusCode': 'std_status'})
assert (ci.cited_appl == ci.q).all()
apps = pd.Series(pd.concat([ci.citing_appl, S.applNo]).unique(), name='a')
con = duckdb.connect(); con.register('apps', apps.to_frame())
m = con.sql(f"""select appl_no a, min(appl_date) appl_date, min(pub_date) pub_date, min(rgst_date) rgst_date,
  string_agg(distinct ksic,'|') ksics, min(main_ipc) main_ipc from read_parquet('{B}/_shared/meta/*.parquet') m
  join apps on m.appl_no=apps.a where cc='KR' group by 1""").df()
r = sqlite3.connect(f'file:{ROOT}/수집파이프라인/registry_full.sqlite?mode=ro', uri=True)
reg = pd.read_sql('select applNo a, rgstDate r_rgst, opnDate r_opn, applDate r_appl from patent', r); reg = reg[reg.a.isin(apps)]
br = sqlite3.connect(f'file:{ROOT}/수집파이프라인/business_rights.sqlite?mode=ro', uri=True)
brl = pd.read_sql("select appl_no a, min(appl_date) b_appl, min(rgst_date) b_rgst from right_list where substr(rgst_no,1,2) in ('10','20') group by appl_no", br)
brl = brl[brl.a.isin(apps)]
A = apps.to_frame().merge(m, on='a', how='left').merge(reg.drop_duplicates('a'), on='a', how='left').merge(brl, on='a', how='left')
A['r_appl'] = A.r_appl.fillna(A.b_appl); A['r_rgst'] = A.r_rgst.fillna(A.b_rgst)
A['src'] = np.where(A.appl_date.notna(), 'meta', np.where(A.b_appl.notna() | A.r_appl.notna(), 'rights', 'number_only'))


def dt(s):
    return pd.to_datetime(s.replace('', None), format='%Y%m%d', errors='coerce')


A['appl_dt'] = dt(A.appl_date.fillna(A.r_appl)); A['yr_from_no'] = A.a.str[2:6].astype(int)
A['pub_dt'] = dt(A.pub_date.fillna(A.r_opn)); A['rgst_dt'] = dt(A.rgst_date.fillna(A.r_rgst))
A['vis_cons'] = A.rgst_dt.fillna(pd.to_datetime((A.yr_from_no + 4).astype(str) + '-01-01'))
A['vis_lib'] = A.pub_dt.fillna(A.appl_dt + pd.DateOffset(months=18)).fillna(pd.to_datetime((A.yr_from_no + 2).astype(str) + '-01-01'))
ci = ci.merge(A[['a', 'src', 'ksics', 'appl_dt', 'pub_dt', 'rgst_dt', 'vis_cons', 'vis_lib', 'yr_from_no']].rename(columns={'a': 'citing_appl'}),
              on='citing_appl', how='left')
ci = ci.merge(S[['applNo', 'ksic']].rename(columns={'applNo': 'cited_appl', 'ksic': 'cited_ksic'}), on='cited_appl', how='left')
ci['self_ksic'] = [(isinstance(k, str) and c in k.split('|')) for k, c in zip(ci.ksics, ci.cited_ksic)]
ci.drop(columns=['q']).to_parquet(os.path.join(D, 'r3kip_o18_citing_edges.parquet'), index=False)
EX = ['E0801', 'E0802', 'E0805']
g = ci.groupby(['cited_appl', 'citing_appl']).agg(
    exam=('ctype', lambda s: s.isin(EX).any()), appl=('ctype', lambda s: (s == 'E0806').any()),
    vis_cons=('vis_cons', 'first'), vis_lib=('vis_lib', 'first'), src=('src', 'first'),
    self_ksic=('self_ksic', 'first'), yr=('yr_from_no', 'first')).reset_index()
g = g[g.cited_appl != g.citing_appl]
g.to_parquet(os.path.join(D, 'r3kip_o18_citing_pairs.parquet'), index=False)
u = g.drop_duplicates('citing_appl')
link = dict(edges=int(len(ci)), pairs=int(len(g)), uniq_citing=int(len(u)),
            ctype_counts=ci.ctype.value_counts().to_dict(),
            src_share=u.src.value_counts(normalize=True).round(4).to_dict(),
            rgst_dt_share=round(float(A[A.a.isin(u.citing_appl)].rgst_dt.notna().mean()), 4),
            self_ksic=round(float(g.self_ksic.mean()), 4), snapshot_has_any=round(float((S.n_rows > 0).mean()), 4),
            max_rows=int(S.n_rows.max()))
gy = g.merge(S[['applNo']].assign(cy=S.applNo.str[2:6].astype(int).values), left_on='cited_appl', right_on='applNo')
link['citing_before_cited'] = round(float((gy.yr < gy.cy).mean()), 4)
link['yr_gap_q05_25_50_75_95'] = [float(x) for x in (gy.yr - gy.cy).quantile([.05, .25, .5, .75, .95])]
link['citing_year_range'] = [int(u.yr.min()), int(u.yr.max())]
link['cited_appl_year_range'] = [int(S.applNo.str[2:6].astype(int).min()), int(S.applNo.str[2:6].astype(int).max())]
rows = []
for a in S.applNo:
    e = g[g.cited_appl == a]
    for y in range(2008, 2020):
        dd = pd.Timestamp(f'{y}-01-01'); vc = e.vis_cons < dd
        rows.append(dict(applNo=a, asof=f'{y}-01-01', fwd_cons=int(vc.sum()), fwd_lib=int((e.vis_lib < dd).sum()),
                         fwd_cons_exam=int((vc & e.exam).sum()), fwd_cons_appl=int((vc & e.appl).sum())))
F = pd.DataFrame(rows).merge(S[['applNo', 'set', 'ksic']], on='applNo')
F = F.merge(g.groupby('cited_appl').size().rename('fwd_total_snapshot'), left_on='applNo', right_index=True, how='left')
F['fwd_total_snapshot'] = F.fwd_total_snapshot.fillna(0).astype(int)
F.to_parquet(os.path.join(D, 'r3kip_o18_fwd_asof.parquet'), index=False)
F0 = pd.read_parquet(os.path.join(D, 'r3kip_patent_fwd_asof.parquet'))
pd.concat([F0, F], ignore_index=True).to_parquet(os.path.join(D, 'r3kip_all_fwd_asof.parquet'), index=False)

cov = ['age', 'claims', 'paid_ahead', 'yrs_since_pay', 'n_renew', 'own_corp', 'own_foreign', 'port_alive', 'a_lapse_rate', 'n_transfer', 'n_ipc']


def prep(samp, Ld, asof):
    fe = pd.read_parquet(os.path.join(B, 'T4_특허포기예측', 'R2', 'cache', f'feat_{Ld}.parquet'), columns=['pid'] + cov)
    t = samp[['applNo', 'pid', 'ksic', 'y', 'w']].merge(fe, on='pid', how='left').merge(asof, on='applNo')
    t['lf'] = np.log1p(t.fwd_cons); t['lfl'] = np.log1p(t.fwd_lib); t['anyf'] = (t.fwd_cons > 0).astype(float)
    t['lsnap'] = np.log1p(t.fwd_total_snapshot); t['lfe'] = np.log1p(t.fwd_cons_exam)
    X = t[cov].astype(float); X = X.fillna(X.median()); X = (X - X.mean()) / X.std().replace(0, 1)
    return t, X


def tests(t, X, grp):
    def cl(extra):
        Z = X.copy()
        for c in extra:
            Z[c] = t[c].values
        return ConditionalLogit(t.y.values, Z, groups=grp).fit(disp=0, method='bfgs', maxiter=2000)
    r0 = cl([]); res = {}
    for v in ['lf', 'lfl', 'lfe', 'anyf', 'lsnap']:
        r1 = cl([v]); lr = 2 * (r1.llf - r0.llf)
        res[v] = dict(coef=round(float(r1.params[v]), 4), se=round(float(r1.bse[v]), 4), p_LR=round(float(stats.chi2.sf(lr, 1)), 4))
    res['mean_fwd_lapse'] = round(float(t[t.y == 1].fwd_cons.mean()), 3); res['mean_fwd_keep'] = round(float(t[t.y == 0].fwd_cons.mean()), 3)
    res['share_cited_lapse'] = round(float((t[t.y == 1].fwd_cons > 0).mean()), 3); res['share_cited_keep'] = round(float((t[t.y == 0].fwd_cons > 0).mean()), 3)
    res['mw_p'] = round(float(stats.mannwhitneyu(t[t.y == 1].fwd_cons, t[t.y == 0].fwd_cons).pvalue), 4)
    d0 = []; d1 = []; X1 = X.assign(lf=t.lf.values)
    for seed in range(20):
        p0 = np.zeros(len(t)); p1 = np.zeros(len(t))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, t.y):
            p0[te] = LogisticRegression(max_iter=2000).fit(X.iloc[tr], t.y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
            p1[te] = LogisticRegression(max_iter=2000).fit(X1.iloc[tr], t.y.iloc[tr]).predict_proba(X1.iloc[te])[:, 1]
        d0.append(roc_auc_score(t.y, p0)); d1.append(roc_auc_score(t.y, p1))
    d0 = np.array(d0); d1 = np.array(d1)
    res.update(cv_auc_base=round(float(d0.mean()), 4), cv_auc_plus=round(float(d1.mean()), 4), cv_diff=round(float((d1 - d0).mean()), 4),
               cv_pos=float((d1 > d0).mean()), n=int(len(t)), n_lapse=int(t.y.sum()))
    return res


out = {'link': link}
t18, X18 = prep(S, '20180101', F[F['asof'] == '2018-01-01'].drop(columns=['set', 'ksic']))
out['T4_O18'] = tests(t18, X18, t18.ksic.values)
S16 = pd.read_parquet(os.path.join(D, 'r3kip_sample.parquet')); S16 = S16[S16.set == 'T4']
t16, X16 = prep(S16, '20160101', F0[F0['asof'] == '2016-01-01'].drop(columns=['set', 'ksic']))
out['T4_O16_recheck'] = tests(t16, X16, t16.ksic.values)
tp = pd.concat([t16.assign(o='O16'), t18.assign(o='O18')], ignore_index=True); Xp = pd.concat([X16, X18], ignore_index=True)
out['T4_pooled_O16_O18'] = tests(tp, Xp, (tp.o + '|' + tp.ksic).values)
out['asof_mean_O18_sample_cons'] = F.groupby('asof').fwd_cons.mean().round(3).to_dict()
out['O18_at_L_mean'] = dict(cons=round(float(t18.fwd_cons.mean()), 3), lib=round(float(t18.fwd_lib.mean()), 3),
                            snapshot=round(float(t18.fwd_total_snapshot.mean()), 3))
json.dump(out, open(os.path.join(D, 'r3kip_o18_pretest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
