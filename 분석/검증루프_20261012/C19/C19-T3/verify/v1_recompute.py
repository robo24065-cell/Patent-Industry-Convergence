# C19-T3 verify V1: diff_numbers.csv 무작위 10행(seed 20261017) 독립 재계산 — duckdb·원 스크립트 미사용(pyarrow/pandas), V1E도 직접 다시 만든다.
#  - lo 계열(13·60·71·496·517행): 해외 문서 family의 KR 등록번호 토큰(KR10/20+7자리) ↔ KR 등록번호 앞 9자리 연결을 pandas로 구현
#  - 562행: 패널(ksic×cc×출원연도, 우선권 KR 기원)과 그룹 안 분산 비중을 pandas로 구현, 순열 p는 다른 시드(7)로 5,000회
#  - 544·551행: 모형 재구현 대신 다른 에이전트가 독립 생성한 메타로 원 스크립트를 돌린 C18-T2 verify_0/r2/map_myV1E 산출을 직접 읽어 대조
import os, sys, json, csv, random, re
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, pyarrow as pa
sys.stdout.reconfigure(encoding='utf-8')
AN = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석'
MC = AN + r'\모델경쟁_20260927'; VL = AN + r'\검증루프_20261012'; W = VL + r'\C19\C19-T3'
META = MC + r'\_shared\meta'
rows = list(csv.DictReader(open(W + r'\diff_numbers.csv', encoding='utf-8-sig')))
idx = sorted(random.Random(20261017).sample(range(len(rows)), 10))
D = ds.dataset(META, format='parquet')
def load(cols, filt):
    return D.to_table(columns=cols, filter=filt).to_pandas()
# --- 이중 수록 제거 대상(독립): US 7자리 중 앞에 0을 붙인 8자리가 있는 것, EP 앞 0 제거 키가 같은 서로 다른 번호 중 짧은 것
us = load(['appl_no'], ds.field('cc') == 'US'); ep = load(['appl_no'], ds.field('cc') == 'EP')
us_set = set(us.appl_no); ep_set = set(ep.appl_no)
drop_us = {a for a in us_set if len(a) == 7 and ('0' + a) in us_set}
ek = pd.DataFrame({'a': list(ep_set)}); ek['k'] = ek.a.str.lstrip('0'); ek['L'] = ek.a.str.len()
g = ek.groupby('k').L.transform('max'); cnt = ek.groupby('k').a.transform('size')
drop_ep = set(ek[(cnt > 1) & (ek.L < g)].a)
del us, ep, ek
R = {'sample_idx': idx, 'n_drop_us': len(drop_us), 'n_drop_ep': len(drop_ep)}
# --- lo: 연결
kr = load(['appl_no', 'rgst_no', 'ksic'], (ds.field('cc') == 'KR') & ds.field('rgst_no').is_valid())
kr = kr[kr.rgst_no.str.len() >= 9]; kr['r9'] = kr.rgst_no.str[:9]; kr['ksic'] = kr.ksic.astype(str)
krm = kr[['appl_no', 'r9', 'ksic']].rename(columns={'appl_no': 'kr_appl_no', 'ksic': 'kk'})
parts = []
for bt in D.to_batches(columns=['cc', 'appl_no', 'ksic', 'family'], filter=(ds.field('cc') != 'KR'), batch_size=500000):
    m = pc.fill_null(pc.match_substring(bt.column('family'), 'KR'), False)
    parts.append(bt.filter(m).to_pandas())
fo = pd.concat(parts, ignore_index=True); del parts
fo['cc'] = fo.cc.astype(str); fo['ksic'] = fo.ksic.astype(str)
tok = fo.assign(t=fo.family.str.split('|')).explode('t')
tok = tok[tok.t.str.fullmatch(r'KR(10|20)[0-9]{7}', na=False)]
tok['r9'] = tok.t.str[2:]
tok = tok[['cc', 'appl_no', 'ksic', 'r9']].drop_duplicates()
del fo
def links(t):
    m = t.merge(krm, on='r9')
    out = {}
    for cc, d in m.groupby('cc'):
        out[cc] = dict(n=int(len(d)), agree=float((d.ksic == d.kk).mean()))
    return out
L0 = links(tok)
tokV = tok[~(((tok.cc == 'US') & tok.appl_no.isin(drop_us)) | ((tok.cc == 'EP') & tok.appl_no.isin(drop_ep)))]
LV = links(tokV)
R['lo_P0'] = L0; R['lo_V1E'] = LV
del tok, tokV, kr, krm
# --- 562: 패널·그룹 안 분산 비중
sys.path.insert(0, MC + r'\_shared_report\label_validity')
from lv_common import ksic_to_group
parts = []
for bt in D.to_batches(columns=['ksic', 'cc', 'appl_no', 'appl_year', 'prio_cc'], filter=(ds.field('appl_year') >= 2005) & (ds.field('appl_year') <= 2017), batch_size=500000):
    b = bt.to_pandas()
    cc = b.cc.astype(str); an = b.appl_no.astype(str)
    first = b.prio_cc.astype(object).where(b.prio_cc.notna(), '').astype(str).str.split('|').str[0]
    origin = np.where(first.isin(['', 'None', 'nan']), cc, first)
    drop = ((cc == 'US') & an.isin(drop_us)) | ((cc == 'EP') & an.isin(drop_ep))
    parts.append(pd.DataFrame({'ksic': b.ksic.astype(str).values, 'cc': cc.values, 'h': pd.util.hash_array(an.values.astype(object)), 'appl_year': b.appl_year.astype('int16').values,
                               'kro': (origin == 'KR'), 'drop': drop.values}))
pn = pd.concat(parts, ignore_index=True); del parts
def panel(p):
    p = p.drop_duplicates(['ksic', 'cc', 'h', 'appl_year'])
    return p.groupby(['ksic', 'appl_year']).agg(n=('h', 'size'), k=('kro', 'sum')).reset_index()
def share_change(P):
    E, Lw = [2005, 2006, 2007], [2015, 2016, 2017]
    a = P[P.appl_year.isin(E)].groupby('ksic')[['n', 'k']].sum(); b = P[P.appl_year.isin(Lw)].groupby('ksic')[['n', 'k']].sum()
    return 100 * (b.k / b.n - a.k / a.n)
def within(v, gser, seed=7, n=5000):
    v = v.astype(float); c = pd.factorize(gser)[0]
    def ws(vv, cc):
        m = np.bincount(cc, weights=vv) / np.bincount(cc); return ((vv - m[cc]) ** 2).sum() / ((vv - vv.mean()) ** 2).sum()
    obs = ws(v.values, c); rng = np.random.default_rng(seed)
    sims = np.array([ws(v.values, rng.permutation(c)) for _ in range(n)])
    return float(obs), float(sims.mean()), float((sims <= obs + 1e-12).mean())
res562 = {}
for lab, p in (('P0', pn), ('V1E', pn[~pn['drop']])):
    P = panel(p); s = share_change(P)
    G = pd.Series({k: ksic_to_group(k) for k in s.index}); gs = G.value_counts(); K34 = [k for k in s.index if G[k] is not None and gs[G[k]] > 1]
    obs, null, pv = within(s[K34], G[K34])
    res562[lab] = dict(n=len(K34), within=obs, null=null, p=pv)
R['562'] = res562
# --- 544·551: C18-T2 r2 독립 메타판 산출 직접 읽기
r2 = VL + r'\C18\C18-T2\verify_0\r2'
t3 = pd.read_csv(r2 + r'\map_myV1E\t3a_forecast_summary.csv', encoding='utf-8-sig'); t3p = pd.read_csv(r2 + r'\run_P0\t3a_forecast_summary.csv', encoding='utf-8-sig')
m3 = json.load(open(r2 + r'\map_myV1E\m3_summary.json', encoding='utf-8')); m3p = json.load(open(r2 + r'\run_P0\m3_summary.json', encoding='utf-8'))
R['544'] = dict(P0_pairacc=float(t3p.iloc[0]['쌍순서_M']), V1E_pairacc=float(t3.iloc[0]['쌍순서_M']), P0_kr_l2=float(t3p.iloc[6]['RMSE차이(M−G)']), V1E_kr_l2=float(t3.iloc[6]['RMSE차이(M−G)']),
                rows_ok=bool(t3.iloc[0]['목표'] == '5국' and int(t3.iloc[6]['L']) == 2 and t3.iloc[6]['목표'] == 'KR'))
R['551'] = dict(P0=m3p['dev_fwd_rawp<0.05(bonf)'], V1E=m3['dev_fwd_rawp<0.05(bonf)'], V1E_rev=m3['dev_rev_rawp<0.05(bonf)'])
# --- 판정: 본 계산(diff_numbers) 새값과 표기 자릿수에서 같은가
chk = []
for i in idx:
    r = rows[i]; ln = int(r['원고행']); new = r['새문자열']
    if ln in (13, 60):
        a = min(LV['US']['agree'], LV['JP']['agree'], LV['EP']['agree']); b = max(LV['US']['agree'], LV['JP']['agree'], LV['EP']['agree'])
        mine = f'{round(a*100):.0f}–{round(b*100):.0f}%'; ok = mine in new
    elif ln in (71, 496):
        mine = f"{LV['EP']['agree']*100:.1f}%"; ok = mine in new
    elif ln == 517 and 'US' in r['원래문자열']:
        mine = f"{LV['US']['n']:,}"; ok = mine in new
    elif ln == 517:
        mine = f"{LV['EP']['n']:,}"; ok = mine in new
    elif ln == 544 and '쌍 순서' in r['원래문자열']:
        mine = f"{R['544']['V1E_pairacc']:.3f}"; ok = mine in new
    elif ln == 544:
        mine = f"+{R['544']['V1E_kr_l2']:.4f}"; ok = mine in new
    elif ln == 551:
        mine = f"{R['551']['V1E']}개"; ok = mine in new
    elif ln == 562:
        mine = f"{res562['V1E']['within']*100:.1f}%"; ok = mine in new
        mine += f"(p≈{res562['V1E']['p']:.4f}, P0 {res562['P0']['within']*100:.1f}%·p≈{res562['P0']['p']:.4f})"
    chk.append(dict(idx=i, 원고행=ln, 본계산=new, 독립=mine, 일치=bool(ok)))
R['checks'] = chk; R['all_ok'] = all(c['일치'] for c in chk)
R['P0_repro'] = dict(US=L0['US'], EP=L0['EP'], JP=L0['JP'])
json.dump(R, open(W + r'\verify\v1_recompute.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
for c in chk: print(c)
print('all_ok', R['all_ok'], 'drops', len(drop_us), len(drop_ep), 'P0', L0.get('US'), L0.get('EP'), '562', res562)
