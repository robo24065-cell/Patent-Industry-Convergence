# verify_1 r1: C04-C 반박 검증. libA 미사용 독립 구현.
# (1) 원자료 NPS에서 E0/amt0/E1/amt1 기업키 일치만으로 재계산 -> yC 재구성, 캐시와 비교
# (2) X2를 kr_patents + country_year_cube에서 독립 재계산 -> AUC(sklearn) 및 변형 민감도
# (3) TXT-FREE / H-sc-1 / FREE-IPC 공개 배정 X2로 dAUC 군집 부트스트랩(200, 다른 시드)
import sys, json, re, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
T0 = time.time()
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
EXT = B + r'\분석\모델경쟁_20260927\_shared\ext'
W = B + r'\분석\검증루프_20261012\C04\C04-T4'
OUT = W + r'\verify_1'
S1_OF = {'201512': '201712', '201612': '201812', '201712': '201912', '201812': '202012', '201912': '202112', '202012': '202212', '202112': '202312'}
REF = {'201512': '20151130', '201612': '20161130', '201712': '20171130', '201812': '20181130', '201912': '20191130', '202012': '20201231', '202112': '20211231'}
CORP = {str(x) for x in range(81, 89)}
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def nm(s):
    s = SUF.sub('', str(s or '')).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", '', s)

u = pd.read_parquet(W + r'\cache\A_units_dev.parquet')
out = {}

def firm(s):
    assert s <= '202312'
    d = pd.read_parquet(EXT + rf'\firm_nps_wkpl_{s}.parquet', columns=['name', 'biz6', 'stat', 'n_emp', 'amt'])
    d = d[(d.stat.astype(str) == '1') & d.biz6.astype(str).str[3:5].isin(CORP)]
    d = d.assign(key=d.biz6 + '|' + d.name.map(nm))
    return d.groupby('key').agg(E=('n_emp', 'sum'), amt=('amt', 'sum'))

R = []
for s0, s1 in S1_OF.items():
    uu = u[u.s0 == s0][['s0', 'key', 'nname', 'E0', 'amt0', 'E1', 'amt1', 'lw', 'yC', 'yA', 'eligC']].copy()
    f0, f1 = firm(s0), firm(s1)
    uu['E0r'] = uu.key.map(f0.E); uu['amt0r'] = uu.key.map(f0.amt)
    uu['E1r'] = uu.key.map(f1.E).fillna(0); uu['amt1r'] = uu.key.map(f1.amt).fillna(0)
    el = (uu.E0r >= 10) & (uu.E1r >= 0.8 * uu.E0r)
    lw = np.log((uu.amt1r / uu.E1r.replace(0, np.nan)) / (uu.amt0r / uu.E0r))
    thr = lw[el].quantile(0.9)
    uu['lw_r'] = lw; uu['yC_r'] = (el & (lw >= thr)).astype(int)
    # 대안 정의: 상위 10%를 E0>=10 전체(E1<0.8E0 포함, 단 lw 유한) 기준으로
    el2 = (uu.E0r >= 10) & np.isfinite(lw)
    thr2 = lw[el2].quantile(0.9)
    uu['yC_alt'] = (el & (lw >= thr2)).astype(int)
    rec = dict(s0=s0, units=len(uu), E0_match=float((uu.E0r == uu.E0).mean()), amt0_match=float((uu.amt0r == uu.amt0).mean()),
               E1_match=float((uu.E1r == uu.E1).mean()), amt0_zero=int((uu.amt0r == 0).sum()), lw_inf=int(np.isinf(uu.lw).sum()), lw_inf_r=int(np.isinf(lw[el]).sum()),
               elig=int(el.sum()), posC_cache=int(uu.yC.sum()), posC_r=int(uu.yC_r.sum()), posC_alt=int(uu.yC_alt.sum()),
               yC_agree=float((uu.yC == uu.yC_r).mean()), thr=float(thr))
    print(rec, round(time.time() - T0), flush=True); R.append(rec)
    out.setdefault('nps', []).append(rec)
    u.loc[uu.index, ['yC_r', 'yC_alt', 'E0r']] = uu[['yC_r', 'yC_alt', 'E0r']].values
del f0, f1
u['eligC_r'] = (u.E0r >= 10).astype(int)
out['P1_nonoverlap_cache'] = int(u[u.s0.isin(['201512', '201712', '201912', '202112'])].yC.sum())
out['P1_nonoverlap_r'] = int(u[u.s0.isin(['201512', '201712', '201912', '202112'])].yC_r.sum())

# (2) X2 독립 재계산
k = pd.read_parquet(B + r'\공모전데이터\작업데이터\kr_patents.parquet', columns=['출원번호', '출원연도', '공개일자', '등록일자', '출원인명', '세세분류코드'])
k.columns = ['appl_no', 'ay', 'pub', 'rg', 'aname', 'ksic']
for c in ['pub', 'rg']:
    k[c] = k[c].fillna('').astype(str).str.strip(); k.loc[k[c].str.len() != 8, c] = '99999999'
k['vis'] = np.minimum(k.pub.values.astype(np.int64), k.rg.values.astype(np.int64))
k['nname'] = k.aname.map(nm); k['ksic'] = k.ksic.astype(str)
cube = pd.read_parquet(B + r'\공모전데이터\작업데이터\country_year_cube.parquet')
cube.columns = ['ksic', 'kname', 'cty', 'ay', 'n', 'nreg']
cube['ksic'] = cube.ksic.astype(str)
F = cube[cube.cty.isin(['US', 'EP', 'CN', 'JP'])].groupby(['ksic', 'ay']).n.sum().unstack(fill_value=0)
lab = pd.read_parquet(B + r'\분석\검증루프_20261012\C02\C02-T4\cache\kr_publab_src.parquet')
lab['sc'] = lab.sg.str[:4]; lab['mg'] = lab.sg.str.split('/').str[0]
lab = lab.drop_duplicates('appl_no').set_index('appl_no')

def g_of(t0):
    a = F.reindex(columns=[t0 - 3, t0 - 2], fill_value=0).sum(axis=1); b = F.reindex(columns=[t0 - 5, t0 - 4], fill_value=0).sum(axis=1)
    return np.log((a + 1) / (b + 1))

def hard(t0, levels, nmin):
    sub = lab[lab.appl_year <= t0]
    res = pd.Series(np.nan, index=lab.index, dtype=object)
    for l in levels:
        t = sub[sub[l].notna()].groupby([l, 'k_free']).size().reset_index(name='n')
        tot = t.groupby(l).n.sum()
        md = t.sort_values([l, 'n', 'k_free'], ascending=[True, False, True]).drop_duplicates(l).set_index(l).k_free
        md = md[md.index.isin(tot[tot >= nmin].index)]
        v = lab[l].map(md)
        fill = res.isna() & v.notna(); res[fill] = v[fill]
    return res

X = []
for s0 in S1_OF:
    t0 = int(s0[:4]); ref = int(REF[s0]); g = g_of(t0)
    uu = u[u.s0 == s0]
    dv = k[(k.vis <= ref) & k.nname.isin(set(uu.nname))][['appl_no', 'ay', 'nname', 'ksic']]
    assert dv.appl_no.map(k.set_index('appl_no').vis).max() <= ref
    rec = dv[dv.ay >= t0 - 2]; hr = set(rec.nname)
    dd = pd.concat([rec, dv[~dv.nname.isin(hr)]])
    labs = {'prov': dd.ksic, 'TXT-FREE': dd.appl_no.map(lab.k_free), 'FREE-IPC': dd.appl_no.map(hard(t0, ['sg', 'mg', 'sc'], 3)),
            'H-sc-1': dd.appl_no.map(hard(t0, ['sc'], 1))}
    xx = pd.DataFrame(index=uu.nname)
    for a, L in labs.items():
        xx[a] = L.astype(object).map(g).groupby(dd.nname.values).mean().reindex(uu.nname).values
    # 변형: 보이는 전체 문서(최근 3년 제한 없음)
    xx['prov_all'] = dv.ksic.map(g).groupby(dv.nname.values).mean().reindex(uu.nname).values
    xx = xx.reset_index(drop=True)
    xx.index = uu.index
    X.append(xx)
    print(s0, 'docs', len(dd), round(time.time() - T0), flush=True)
X = pd.concat(X)
u = u.join(X)

def A(y, s):
    m = np.isfinite(s)
    return float(roc_auc_score(y[m], s[m]))

res = {}
for ycol, ecol in [('yC', 'eligC'), ('yC_r', 'eligC_r'), ('yC_alt', 'eligC_r')]:
    S = u[u[ecol] == 1]
    for xc in ['prov', 'prov_all', 'TXT-FREE', 'FREE-IPC', 'H-sc-1']:
        res[f'{ycol}|{xc}'] = round(A(S[ycol].values, S[xc].values.astype(float)), 4)
    res[f'{ycol}|prov|within'] = [round(A(g_[ycol].values, g_['prov'].values.astype(float)), 4) for _, g_ in S.groupby('s0')]
    res[f'{ycol}|n_pos'] = int(S[ycol].sum()); res[f'{ycol}|n'] = len(S)
    res[f'{ycol}|nan_x'] = int(S['prov'].isna().sum())
# 상위 20% X2(원점 내) 양성률 비
S = u[u.eligC == 1].copy()
S['top'] = S.groupby('s0').prov.transform(lambda v: v >= v.quantile(0.8))
res['yC_rate_top20_vs_rest'] = [round(float(S[S.top].yC.mean()), 4), round(float(S[~S.top].yC.mean()), 4)]
# (3) 부트스트랩 dAUC (시드 20261013 - 원 시드와 다름)
kc, keys = pd.factorize(S.key); rng = np.random.default_rng(20261013)
boot = {}
for a in ['TXT-FREE', 'FREE-IPC', 'H-sc-1']:
    ok = np.isfinite(S.prov.values.astype(float)) & np.isfinite(S[a].values.astype(float))
    base = A(S.yC.values[ok], S.prov.values[ok].astype(float)) - A(S.yC.values[ok], S[a].values[ok].astype(float))
    ds = []
    r2 = np.random.default_rng(20261013)
    for b in range(200):
        cnt = r2.multinomial(len(keys), np.full(len(keys), 1 / len(keys)))
        ix = np.repeat(np.arange(len(S)), cnt[kc]); ix = ix[ok[ix]]
        yy = S.yC.values[ix]
        ds.append(A(yy, S.prov.values[ix].astype(float)) - A(yy, S[a].values[ix].astype(float)))
    boot[a] = [round(base, 4), round(float(np.percentile(ds, 2.5)), 4), round(float(np.percentile(ds, 97.5)), 4)]
    print(a, boot[a], round(time.time() - T0), flush=True)
out['auc'] = res; out['boot_yC'] = boot
print(json.dumps(res, ensure_ascii=False))
json.dump(out, open(OUT + r'\r1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('done', round(time.time() - T0))
