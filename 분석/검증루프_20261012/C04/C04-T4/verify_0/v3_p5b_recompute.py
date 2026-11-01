# verify_0 v3: C04-A P5-b 독립 재계산(libA 미사용). X2를 원천(kr_patents, country_year_cube)에서 다시 만들고
# 단변량 AUC(X2->yA), AUC_prov 자체의 기업키 군집 부트스트랩 CI, TXT-FREE ΔAUC, 민감도(판정 밖)를 계산한다.
import sys, json, re, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
T0 = time.time()
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
W = B + r'\분석\검증루프_20261012\C04\C04-T4'
OUT = W + r'\verify_0\v3_out.json'
REF = {'201512': '20151130', '201612': '20161130', '201712': '20171130', '201812': '20181130', '201912': '20191130',
       '202012': '20201231', '202112': '20211231'}
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)

def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    m = ~np.isnan(s); y, s = y[m], s[m]
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

u = pd.read_parquet(W + r'\cache\A_units_dev.parquet', columns=['s0', 'key', 'nname', 'E0', 'yA', 'ksic_mode', 'n_vis'])
k = pd.read_parquet(B + r'\공모전데이터\작업데이터\kr_patents.parquet', columns=['출원번호', '출원연도', '공개일자', '등록일자', '출원인명', '세세분류코드'])
k.columns = ['appl_no', 'ay', 'pub', 'rg', 'aname', 'ksic']
for c in ['pub', 'rg']:
    k[c] = k[c].fillna('').astype(str).str.strip(); k.loc[k[c].str.len() != 8, c] = '99999999'
k['vis'] = np.where(k.rg < k.pub, k.rg, k.pub)
k['ksic'] = k.ksic.astype(str)
k['multi'] = k.aname.fillna('').astype(str).str.contains(r'\|')
k['nname'] = k.aname.map(norm)
res = {'kr_rows': len(k), 'multi_applicant_share': round(float(k.multi.mean()), 4), 'appl_no_dup': int(k.appl_no.duplicated().sum())}
print(res, round(time.time() - T0), flush=True)

cube = pd.read_parquet(B + r'\공모전데이터\작업데이터\country_year_cube.parquet')
cube.columns = ['ksic', 'kname', 'cty', 'yr', 'n', 'nreg']
cube['ksic'] = cube.ksic.astype(str)
F = cube[cube.cty.isin(['US', 'EP', 'CN', 'JP'])].pivot_table(index='ksic', columns='yr', values='n', aggfunc='sum', fill_value=0)
def g_of(t0, a=(-3, -2), b=(-5, -4)):
    A = sum(F.get(t0 + d, 0) for d in a); Bv = sum(F.get(t0 + d, 0) for d in b)
    return np.log((A + 1) / (Bv + 1))

def build(s0, g, recent=True):
    ref = REF[s0]; t0 = int(s0[:4]); uu = u[u.s0 == s0]
    dv = k[(k.vis <= ref) & k.nname.isin(set(uu.nname))]
    assert dv.vis.max() <= ref
    if recent:
        rec = dv[dv.ay >= t0 - 2]; hr = set(rec.nname)
        dv = pd.concat([rec, dv[~dv.nname.isin(hr)]])
    x = dv.ksic.map(g).groupby(dv.nname).mean()
    out = uu.copy(); out['x'] = out.nname.map(x).values
    return out

S = pd.concat([build(s0, g_of(int(s0[:4]))) for s0 in REF], ignore_index=True)
res['units'] = len(S); res['x_nan'] = int(S.x.isna().sum()); res['pos'] = int(S.yA.sum())
res['auc_prov_pooled'] = round(auc(S.yA, S.x), 4)
res['auc_prov_by_origin'] = {s0: round(auc(g_.yA, g_.x), 4) for s0, g_ in S.groupby('s0')}
res['auc_prov_within_mean'] = round(float(np.mean(list(res['auc_prov_by_origin'].values()))), 4)
print(res, round(time.time() - T0), flush=True)

# AUC_prov 군집 부트스트랩(기업키) 200회
kc, keys = pd.factorize(S.key); rng = np.random.default_rng(7)
bs = []
for b in range(200):
    w = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)))[kc]
    ix = np.repeat(np.arange(len(S)), w)
    bs.append(auc(S.yA.values[ix], S.x.values[ix]))
res['auc_prov_boot95'] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
print('boot', res['auc_prov_boot95'], round(time.time() - T0), flush=True)

# TXT-FREE 공개 배정 ΔAUC(점추정)
lab = pd.read_parquet(B + r'\분석\검증루프_20261012\C02\C02-T4\cache\kr_publab_src.parquet', columns=['appl_no', 'k_free']).drop_duplicates('appl_no').set_index('appl_no').k_free.astype(str)
k_txt = k.copy(); k_txt['ksic'] = k_txt.appl_no.map(lab)
kk = k; k = k_txt
St = pd.concat([build(s0, g_of(int(s0[:4]))) for s0 in REF], ignore_index=True)
k = kk
res['auc_txtfree_pooled'] = round(auc(St.yA, St.x), 4)
res['d_auc_prov_minus_txt'] = round(res['auc_prov_pooled'] - res['auc_txtfree_pooled'], 4)

# 민감도(판정 밖): 전체 문서, 성장창 다르게, 규모 통제(E0 십분위 안 AUC 평균)
sens = {}
Sa = pd.concat([build(s0, g_of(int(s0[:4])), recent=False) for s0 in REF], ignore_index=True)
sens['all_docs'] = round(auc(Sa.yA, Sa.x), 4)
Sb = pd.concat([build(s0, g_of(int(s0[:4]), a=(-2, -1, 0), b=(-5, -4, -3))) for s0 in REF], ignore_index=True)
sens['growth_t0m2..t0_vs_t0m5..m3(포함 미래 누출 가능, 참고)'] = round(auc(Sb.yA, Sb.x), 4)
S['dec'] = S.groupby('s0').E0.transform(lambda v: pd.qcut(v.rank(method='first'), 5, labels=False))
sens['within_s0_E0quintile_mean'] = round(float(np.nanmean([auc(g_.yA, g_.x) for _, g_ in S.groupby(['s0', 'dec']) if 0 < g_.yA.sum() < len(g_)])), 4)
sens['spearman_x_E0'] = round(float(S[['x', 'E0']].corr('spearman').iloc[0, 1]), 4)
# X2 상위 20% vs 나머지 양성률
S['top'] = S.groupby('s0').x.transform(lambda v: v >= v.quantile(0.8))
sens['posrate_top20_vs_rest'] = [round(float(S[S.top].yA.mean()), 4), round(float(S[~S.top].yA.mean()), 4)]
res['sensitivity_posthoc'] = sens
print(res, round(time.time() - T0), flush=True)
json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('done', round(time.time() - T0))
