# verify_0 v3: 잠금 원점(S0=2024-12) 챔피언 후보 M1의 다리 피처 br_gKR_prov·br_gF4_prov 독립 재구현(libC/s2 미사용).
# 기준일(2024-12-31)까지 보인 문서만 사용함을 재구현 안에서 assert 한다. 잠금 레이블은 계산하지 않는다.
import sys, os, re, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
ROOT = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
WD = os.path.join(ROOT, '분석', '검증루프_20261012', 'C05', 'C05-T4')
EXTD = os.path.join(ROOT, '분석', '모델경쟁_20260927', '_shared', 'ext')
REFI = 20241231; T0Y = 2024
SUFX = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
nz = lambda s: re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", SUFX.sub("", str(s or "")).upper())
TC = [str(i) for i in range(81, 89)]
rows = []
with open(os.path.join(ROOT, '공모전데이터', '외부데이터', 'Corporate_20260720', 'CORP_APPLICANT.txt'), encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\r\n').split('¶')
        if len(p) >= 5:
            rows.append((p[1].strip(), re.sub(r'\D', '', p[4].strip())))
ca = pd.DataFrame(rows, columns=['nm', 'bz']); ca['n'] = ca.nm.map(nz)
ca['okb'] = ca.bz.str.len() == 10; ca['b6'] = ca.bz.str[:6].where(ca.okb); ca['tc'] = ca.okb & ca.bz.str[3:5].isin(TC)
g = ca[ca.n.str.len() >= 2].groupby('n').agg(N=('bz', 'size'), NB=('okb', 'sum'), U6=('b6', 'nunique'), B6=('b6', 'first'), TCm=('tc', 'max'))
n2b = g[(g.N == g.NB) & (g.U6 == 1) & g.TCm].B6.to_dict()
kp = pd.read_parquet(os.path.join(ROOT, '공모전데이터', '작업데이터', 'kr_patents.parquet'), columns=['출원번호', '출원연도', '공개일자', '등록일자', '출원인명', '세세분류코드'])
kp = kp.drop_duplicates('출원번호')
pub = kp['공개일자'].fillna('').astype(str).str.strip(); rg = kp['등록일자'].fillna('').astype(str).str.strip()
kp['v'] = np.minimum(pub.where(pub.str.len() == 8, '99999999').astype(np.int64), rg.where(rg.str.len() == 8, '99999999').astype(np.int64))
kp['n'] = kp['출원인명'].map(nz); kp['b6'] = kp.n.map(n2b); kp['k'] = kp['세세분류코드'].astype(str)
fv = kp[kp.b6.notna() & (kp.v < 99999999)].groupby('n').v.min()
seen = kp[kp.v <= REFI]
assert seen.v.max() <= REFI
d = pd.read_parquet(os.path.join(EXTD, 'firm_nps_wkpl_202412.parquet'), columns=['name', 'biz6', 'stat', 'form', 'indcd', 'n_emp'])
d = d[(d.stat.astype(str) == '1') & (d.form.astype(str) == '1') & d.biz6.astype(str).str[3:5].isin(TC)].copy()
d['key'] = d.biz6 + '|' + d.name.map(nz)
d = d.sort_values('n_emp', ascending=False)
gg = d.groupby('key', sort=False)
F = pd.DataFrame({'E': gg.n_emp.sum(), 'ic': gg.indcd.first()})
F = F[F.E >= 5]
ic = F.ic.fillna('').astype(str).str.strip()
F['i6'] = np.where(ic.str.fullmatch(r'\d{1,6}') & ~ic.isin(['', '000000']), ic.str.zfill(6), 'NA')
F['i2'] = np.where(F.i6 == 'NA', 'NA', F.i6.str[:2])
vn = fv[fv <= REFI].index
vk = set(pd.Series(vn).map(n2b) + '|' + pd.Series(vn))
LK = F[F.index.isin(vk)]
sl = seen[seen.b6.notna()].copy(); sl['key'] = sl.b6 + '|' + sl.n
sl = sl[sl.key.isin(LK.index)]
# 기업별 KSIC 몫(합 1)
cnt = sl.groupby(['key', 'k']).size().rename('c').reset_index()
cnt['w'] = cnt.c / cnt.groupby('key').c.transform('sum')
cnt = cnt.join(LK[['i6', 'i2']], on='key')
def gr(tab):
    a = tab.reindex(columns=[T0Y - 3, T0Y - 2], fill_value=0).sum(axis=1); b = tab.reindex(columns=[T0Y - 5, T0Y - 4], fill_value=0).sum(axis=1)
    return np.log((a + 1) / (b + 1))
gK = gr(seen.groupby(['k', '출원연도']).size().unstack(fill_value=0))
cb = pd.read_parquet(os.path.join(ROOT, '공모전데이터', '작업데이터', 'country_year_cube.parquet'))
cb.columns = ['k', 'kn', 'cc', 'ay', 'n', 'nreg']; cb['k'] = cb.k.astype(str)
cb = cb[cb.cc.isin(['US', 'EP', 'CN', 'JP']) & (cb.ay <= T0Y - 2)]
gF = gr(cb.groupby(['k', 'ay']).n.sum().unstack(fill_value=0))
K = sorted(set(cnt.k) | set(gK.index) | set(gF.index))
G = cnt.groupby('k').w.sum().reindex(K, fill_value=0); G = G / G.sum()
def share(level):
    S = cnt.pivot_table(index=level, columns='k', values='w', aggfunc='sum', fill_value=0).reindex(columns=K, fill_value=0)
    n = cnt.groupby(level).key.nunique()
    return S, n
S2, n2 = share('i2'); sh2 = (S2 + 5 * G.values) .div(n2 + 5, axis=0)
S6, n6 = share('i6'); par = cnt.drop_duplicates('i6').set_index('i6').i2
sh6 = (S6 + 5 * sh2.loc[par.reindex(S6.index).values].values).div(n6 + 5, axis=0)
vK = gK.reindex(K).fillna(0).values; vF = gF.reindex(K).fillna(0).values
u = pd.read_parquet(os.path.join(WD, 'cache', 'units.parquet'), columns=['key', 's0', 'ind6', 'ind2'])
u = u[u.s0 == '202412'].set_index('key')
br = pd.read_parquet(os.path.join(WD, 'cache', 'bridge.parquet')); br = br[br.s0 == '202412'].set_index('key').reindex(u.index)
out = {}
for nm, vec, col in [('gKR', vK, 'br_gKR_prov'), ('gF4', vF, 'br_gF4_prov')]:
    b6 = pd.Series(sh6.values @ vec, index=sh6.index); b2 = pd.Series(sh2.values @ vec, index=sh2.index); gl = float(G.values @ vec)
    mine = u.ind6.map(b6).fillna(u.ind2.map(b2)).fillna(gl)
    out[col] = dict(maxabs=float(np.abs(mine - br[col]).max()), corr=float(np.corrcoef(mine, br[col])[0, 1]), mean_mine=float(mine.mean()), mean_frozen=float(br[col].mean()))
out['n_linked_firms'] = int(len(LK)); out['n_linked_docs'] = int(len(sl)); out['max_vis_used'] = int(seen.v.max())
out['pass'] = all(out[c]['maxabs'] < 1e-9 for c in ['br_gKR_prov', 'br_gF4_prov'])
print(out)
json.dump(out, open(os.path.join(WD, 'verify_0', 'v3_bridge_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
