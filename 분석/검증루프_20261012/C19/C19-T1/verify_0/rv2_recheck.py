# C19-T1 반박 재검증(rv2): 원 메타 parquet에서 독립 재계산. 읽기 전용, 출력은 verify_0\rv2_out.json 만.
import sys, os, glob, json
sys.dont_write_bytecode = True
import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
import pandas as pd, numpy as np
MC = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rv2_out.json')
cols = ['appl_no', 'appl_date', 'appl_year', 'rgst_no', 'rgst_date', 'pub_date', 'title', 'applicant', 'cc', 'ksic']
parts = []
for f in sorted(glob.glob(os.path.join(MC, '_shared', 'meta', '*.parquet'))):
    t = pq.read_table(f, columns=cols)
    t = t.filter(pc.equal(pc.cast(t['cc'], pa.string()), 'JP'))
    df = t.drop(['cc']).to_pandas(); df['ksic'] = df['ksic'].astype(str)
    parts.append(df)
d = pd.concat(parts, ignore_index=True); del parts
for c in ('rgst_no', 'rgst_date', 'pub_date', 'applicant', 'title', 'appl_date'):
    d[c] = d[c].where(d[c].fillna('').str.strip() != '', None)
R = {'rows': len(d), 'uniq_appl': int(d.appl_no.nunique())}
d['U'] = d.appl_no.str.endswith('U')
nop = d.pub_date.isna() & d.rgst_date.notna()
d['ry'] = d.rgst_date.str[:4]
koho = {2016: 217778, 2017: 199592, 2018: 191073, 2019: 180399, 2020: 179017, 2021: 181368, 2022: 210998}
pubtot = {2016: 226294+41230+10496, 2017: 229241+39200+53986, 2018: 207786+38787+23178, 2019: 221136+37923+25079,
          2020: 205747+37866+21074, 2021: 197906+36725+31057, 2022: 188382+52686}
np_pat = d[nop & ~d.U]
cnt = np_pat.ry.value_counts()
R['nopub_pat_by_ry'] = {y: int(cnt.get(str(y), 0)) for y in range(2012, 2025)}
ratio = {y: cnt.get(str(y), 0) / koho[y] for y in koho}
base = np.mean([ratio[y] for y in range(2016, 2021)])
R['base_ratio'] = base
R['rel'] = {y: ratio[y] / base for y in (2021, 2022)}
exc = {y: cnt.get(str(y), 0) - base * koho[y] for y in (2021, 2022)}
R['excess'] = exc; R['excess_sum'] = sum(exc.values())
# pubno10
p10 = d[d.pub_date.isna() & d.rgst_no.fillna('').str.fullmatch(r'\d{10}')].copy()
R['pubno10_n'] = len(p10)
R['pubno10_hasrgdate'] = int(p10.rgst_date.notna().sum())
R['pubno10_U'] = int(p10.U.sum())
R['pubno10_first4_eq_year'] = int((p10.rgst_no.str[:4] == p10.ry).sum())
dd = (pd.to_datetime(p10.rgst_date, format='%Y%m%d', errors='coerce') - pd.to_datetime(p10.appl_date, format='%Y%m%d', errors='coerce')).dt.days
R['pubno10_lag_q'] = [float(x) for x in dd.quantile([.1, .25, .5, .75, .9])]
R['pubno10_applicant_missing'] = float(p10.applicant.isna().mean())
R['pubno10_months'] = p10.rgst_date.str[:6].value_counts().sort_index().to_dict()
R['pubno10_applno_len'] = p10.appl_no.str.replace('U', '').str.len().value_counts().to_dict()
R['pubno10_appl_year'] = {int(a): int(b) for a, b in p10.appl_year.value_counts().head(6).items()}
# 10-digit rgst_no among rows WITH pub date (are there any?)
R['rgst10_with_pub'] = int((d.pub_date.notna() & d.rgst_no.fillna('').str.fullmatch(r'\d{10}')).sum())
R['rgst_no_len_nopub_2012_20'] = np_pat[np_pat.ry.between('2012', '2020')].rgst_no.fillna('').str.len().value_counts().to_dict()
# 독립 키: 숫자부 변환
s = d.appl_no.str.replace('U', '', regex=False)
k = np.where(s.str.fullmatch(r'\d{8}'), (1988 + pd.to_numeric(s.str[:2], errors='coerce').fillna(0).astype(int)).astype(str) + s.str[2:],
             np.where(s.str.fullmatch(r'\d{10}'), s, d.appl_no))
d['key'] = pd.Series(k, index=d.index) + np.where(d.U, 'U', '')
pub = d[d.pub_date.notna()][['key', 'appl_no', 'appl_date', 'pub_date', 'title']]
# 짝: 같은 key, 다른 appl_no, 공개일 있음
m = d[nop][['key', 'appl_no', 'appl_date', 'rgst_date', 'rgst_no', 'ry', 'U', 'title']].reset_index().merge(
    pub.rename(columns={'appl_no': 'a2', 'appl_date': 'ad2', 'pub_date': 'pd2', 'title': 't2'}), on='key', how='inner')
m = m[m.appl_no != m.a2]
g = m.groupby('index').agg(sd=('ad2', lambda x: False), n=('a2', 'size'))
m['same_ad'] = m.appl_date == m.ad2
m['same_dt'] = m.rgst_date == m.pd2
gi = m.groupby('index').agg(same_ad=('same_ad', 'max'), same_dt=('same_dt', 'max'))
d['k1'] = False; d.loc[gi.index, 'k1'] = True
d['k1ad'] = False; d.loc[gi.index[gi.same_ad.values], 'k1ad'] = True
d['k1dt'] = False; d.loc[gi.index[gi.same_dt.values], 'k1dt'] = True
R['k1_pairs_same_appl_date_share'] = float(m.same_ad.mean())
# 키2 (제목 정규화, 출원일)
d['tn'] = d.title.fillna('').str.replace(r'\s+', '', regex=True).str.lower()
pk = d[d.pub_date.notna() & (d.tn != '') & d.appl_date.notna()][['tn', 'appl_date', 'appl_no']]
m2 = d[nop & (d.tn != '') & d.appl_date.notna()][['tn', 'appl_date', 'appl_no']].reset_index().merge(pk.rename(columns={'appl_no': 'a2'}), on=['tn', 'appl_date'])
m2 = m2[m2.appl_no != m2.a2]
d['k2'] = False; d.loc[m2['index'].unique(), 'k2'] = True
d['dup'] = d.k1 | d.k2
NP = d[nop & ~d.U]
P10m = d.index.isin(p10.index)
res = {}
for y in [str(x) for x in range(2012, 2023)]:
    q = NP[NP.ry == y]; qp = d[P10m & (d.ry == y)]
    res[y] = dict(n=len(q), k1=int(q.k1.sum()), k1ad=int(q.k1ad.sum()), k2=int(q.k2.sum()), dup=int(q.dup.sum()),
                  p10=len(qp), p10_dup=int(qp.dup.sum()), p10_k1dt=int(qp.k1dt.sum()), p10_k1=int(qp.k1.sum()))
R['by_year'] = res
dsum = res['2021']['dup'] + res['2022']['dup']
R['dup_share_excess'] = dsum / R['excess_sum']
R['k1_share_excess'] = (res['2021']['k1'] + res['2022']['k1']) / R['excess_sum']
R['k1ad_share_excess'] = (res['2021']['k1ad'] + res['2022']['k1ad']) / R['excess_sum']
R['k1_share_excess_2021_only'] = res['2021']['k1'] / exc[2021]
R['dup_share_excess_2021_only'] = res['2021']['dup'] / exc[2021]
# 기저 이중수록률(2016-20)을 빼고 본 '초과 이중수록' 몫
bdup = np.mean([res[str(y)]['dup'] / koho[y] for y in range(2016, 2021)])
R['dup_excess_over_base_share'] = (dsum - bdup * (koho[2021] + koho[2022])) / R['excess_sum']
# 공개 총량 rel (공개일 행 특허 + pubno10 특허 - pubno10 이중수록)
pp = d[d.pub_date.notna() & ~d.U].pub_date.str[:4].value_counts()
pr = {y: pp.get(str(y), 0) / pubtot[y] for y in pubtot}
pb = np.mean([pr[y] for y in range(2016, 2021)])
R['pub_rel'] = {y: pr[y] / pb for y in (2021, 2022)}
for y in (2021, 2022):
    a = pp.get(str(y), 0) + res[str(y)]['p10'] - res[str(y)]['p10_dup']
    R[f'pub_plus_uniq_p10_rel_{y}'] = a / pubtot[y] / pb
    R[f'pub_plus_all_p10_rel_{y}'] = (pp.get(str(y), 0) + res[str(y)]['p10']) / pubtot[y] / pb
# no-pub ratio after removing dups (해석 확인용)
for y in (2021, 2022):
    R[f'nopub_rel_after_dup_removal_{y}'] = (res[str(y)]['n'] - res[str(y)]['dup']) / koho[y] / base
    R[f'nopub_rel_after_removing_all_p10_{y}'] = (res[str(y)]['n'] - res[str(y)]['p10']) / koho[y] / base
# 급증 행 정의(T3 py, appl_year>=2000, 특허) 기준 pubno10 비중
pps, rrs = d.pub_date.fillna('99999999'), d.rgst_date.fillna('99999999')
py = np.where(pps <= rrs, pps, rrs); d['py'] = pd.Series(py, index=d.index).str[:4]
sur = d[nop & ~d.U & (d.appl_year >= 2000) & d.py.isin(['2021', '2022'])]
R['surge_rows_t3'] = len(sur); R['surge_p10'] = int(sur.index.isin(p10.index).sum())
json.dump(R, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
