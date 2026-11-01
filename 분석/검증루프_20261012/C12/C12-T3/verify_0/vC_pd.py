# -*- coding: utf-8 -*-
"""verify_0 재시도(vC): (A) SAB 비율·빼도 skill을 pandas+sklearn으로 세 번째 독립 재계산.
작업 스크립트(scripts\\*)·cache·vA_sql.py 미사용. 원천: R3 base_rows, _shared meta, registry_full.sqlite(mode=ro), 동결 예측.
추가 반박: 이름 정규화 방식 변경, 등록 연결 키, 공동출원 귀속 시 키 섞임(이름/코드), 경계 포함 여부, 등록 문서만 키 교체 효과 분리."""
import os, sys, re, json, html, sqlite3, time
import numpy as np, pandas as pd, duckdb
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
T = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
R3 = ROOT + r"\분석\모델경쟁_20260927\T3_기술셀부상예측\R3"
BR = R3 + r"\cache_pub_final\base_rows.parquet"
META = ROOT + r"\분석\모델경쟁_20260927\_shared\meta"
PRED = R3 + r"\preds\FINAL_R3_t0_2021.parquet"
RF = ROOT + r"\수집파이프라인\registry_full.sqlite"
OUT = os.path.dirname(os.path.abspath(__file__))
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
br = con.execute(f"select ksic, appl_no, year, ipc_raw, app1 from read_parquet('{BR.replace(chr(92),'/')}') where cc='KR' and year between 2016 and 2024").df()
pr = pd.read_parquet(PRED)
pr['p_base'] = pr['p_base'].astype(float)
RX1 = re.compile(r'^([A-H]\d{2}[A-Z])\s*(\d{1,4})\s*/')
RX2 = re.compile(r'^([A-H]\d{2}[A-Z])(\d{3,6})$')
def mg(s):
    if s is None: return None
    s2 = s.replace(' ', '')
    m = RX1.match(s2)
    if m: return f"{m.group(1)} {int(m.group(2))}"
    m = RX2.match(s2)
    if m: return f"{m.group(1)} {int(m.group(2)[:-2])}"
    return None
br['mg'] = br['ipc_raw'].map(mg)
br = br[br['mg'].notna()].copy()
br['cell'] = br['ksic'] + '|' + br['mg']
SUFX = ['주식회사', '(주)', '㈜', '(株)', '（주）', '유한회사', ]
PUN = re.compile(r"[\s\.,·\-\(\)\[\]'\"“”‘’&/]+")
def nmA(s):  # 정규화 A: 작업 규칙과 같은 취지(html 해제, 소문자, 법인 표기 제거, 기호 제거)
    s = html.unescape(s or '').lower()
    for x in ['주식회사', '(주)', '㈜', '(株)', '（주）']: s = s.replace(x, '')
    return PUN.sub('', s)
def nmB(s):  # 정규화 B(반박용, 더 느슨): 유한회사·co.,ltd·inc 등도 제거, 전각 괄호 등 모든 비문자 제거
    s = html.unescape(s or '').lower()
    for x in SUFX + ['co.,ltd', 'co., ltd', 'co.ltd', 'inc.', 'inc', 'corporation', 'corp.', 'ltd.', 'ltd']: s = s.replace(x, '')
    return re.sub(r'[\W_]+', '', s)
cells = set(pr['cell'])
d = br[br['cell'].isin(cells) & br['year'].between(2019, 2024)][['cell', 'appl_no', 'year', 'app1']].drop_duplicates()
d['w'] = np.where(d['year'] <= 2021, 'past', 'fut')
res = {}
res['dup_cell_appl'] = int(d.duplicated(['cell', 'appl_no']).sum())
cnt = d.groupby(['cell', 'w']).size().unstack(fill_value=0).reindex(pr['cell']).fillna(0)
P = pr.set_index('cell').join(cnt)
P['D'] = P['fut'] - P['past']
res['krW_mismatch'] = int((np.abs(P['past'] - P['kr_W']) > 1e-6).sum())
res['n_elig_pos'] = [int(len(P)), int(P['yb'].sum())]
res['pos_D_le0'] = int(((P['yb'] == 1) & (P['D'] <= 0)).sum())
def skill(mask_keep):
    q = P[mask_keep]
    a = roc_auc_score(q['yb'], q['p']); b = roc_auc_score(q['yb'], q['p_base'])
    return (a - b) / (1 - b), a, b
res['skill_full'] = skill(np.ones(len(P), bool))
def sab_eval(cr, tag, op='ge'):
    # cr: (cell, appl_no, w, key) 귀속표
    cr = cr.drop_duplicates(['cell', 'appl_no', 'w', 'key'])
    g = cr.assign(v=np.where(cr['w'] == 'fut', 1, -1)).groupby(['cell', 'key'])['v'].sum()
    mx = g.groupby(level=0).max().reindex(P.index).fillna(0)
    thr = 0.5 * P['D']
    hit = (mx >= thr) if op == 'ge' else (mx > thr)
    sab = (P['yb'] == 1) & (P['D'] > 0) & hit
    sk, a, b = skill(~sab.values)
    res[tag] = dict(n_sab=int(sab.sum()), rate=float(sab.sum() / P['yb'].sum()), skill_excl=float(sk), auc=float(a), auc_b=float(b))
    return sab
# (i) 이름 정규화 A·B
for nm, f in [('A', nmA), ('B', nmB)]:
    cr = d.assign(key='N:' + d['app1'].map(f))
    for op in ['ge', 'gt']: s_i = sab_eval(cr, f'i_{nm}_{op}', op)
    if nm == 'A': SAB_I = sab_eval(cr, 'i_A_ge')
# 등록 연결
sq = sqlite3.connect(f"file:{RF}?mode=ro", uri=True)
ap = pd.read_sql("select rgst_no, seq, applicantCd, applicantName from applicant", sq)
pt = pd.read_sql("select rgst_no, applNo from patent", sq)
ap['code'] = np.where(ap['applicantCd'].fillna('').str.len() > 0, ap['applicantCd'], 'NOCD:' + ap['applicantName'].fillna('').map(nmA))
ap['n1'] = ap['applicantName'].fillna('').map(nmA)
res['registry_coapp_share'] = float((ap.groupby('rgst_no').size() > 1).mean())
meta = con.execute(f"select appl_no, any_value(nullif(trim(rgst_no),'')) rgst_no from read_parquet('{META.replace(chr(92),'/')}/*.parquet') where cc='KR' group by appl_no").df()
links = {'K1_meta': meta.dropna().drop_duplicates('appl_no'), 'K2_applNo': pt.rename(columns={'applNo': 'appl_no'}).sort_values('rgst_no').drop_duplicates('appl_no')}
alld = br[['appl_no', 'app1']].drop_duplicates('appl_no')
alld['kname'] = alld['app1'].map(nmA)
for kn, lk in links.items():
    dk = d.merge(lk[['appl_no', 'rgst_no']], on='appl_no', how='left')
    dk['kname'] = dk['app1'].map(nmA)
    res[f'{kn}_reg_share'] = {w: float(dk.loc[dk.w == w, 'rgst_no'].notna().mean()) for w in ['past', 'fut']}
    # 문서별 대표 코드(이름 일치 우선, 그다음 seq 최소) — 전체 문서표(공개 2016–24)
    a2 = alld.merge(lk[['appl_no', 'rgst_no']], on='appl_no').merge(ap[['rgst_no', 'seq', 'code', 'n1']], on='rgst_no')
    a2['m'] = (a2['n1'] == a2['kname']).astype(int)
    rep = a2.sort_values(['appl_no', 'm', 'seq'], ascending=[True, False, True]).drop_duplicates('appl_no')[['appl_no', 'kname', 'code']]
    vc = rep.groupby(['kname', 'code']).size().rename('n').reset_index()
    vc['sh'] = vc['n'] / vc.groupby('kname')['n'].transform('sum')
    n2c = vc.sort_values(['kname', 'n', 'code'], ascending=[True, False, True]).drop_duplicates('kname')
    n2c = n2c[n2c['sh'] >= 0.5].set_index('kname')['code']
    reg = dk[dk['rgst_no'].notna()]; unr = dk[dk['rgst_no'].isna()]
    ukey = np.where(unr['kname'].map(n2c).notna(), 'C:' + unr['kname'].map(n2c).astype(str), 'N:' + unr['kname'])
    unr_cr = unr.assign(key=ukey)[['cell', 'appl_no', 'w', 'key']]
    rr = reg.merge(rep[['appl_no', 'code']], on='appl_no', how='left')
    ii = pd.concat([rr.assign(key='C:' + rr['code'].astype(str))[['cell', 'appl_no', 'w', 'key']], unr_cr])
    allj = reg.merge(ap[['rgst_no', 'code']], on='rgst_no')
    res[f'{kn}_reg_docs_without_applicant_rows'] = int(reg['rgst_no'].isin(ap['rgst_no']).eq(False).sum())
    iii = pd.concat([allj.assign(key='C:' + allj['code'])[['cell', 'appl_no', 'w', 'key']], unr_cr])
    for op in ['ge', 'gt']:
        sab_eval(ii, f'ii_{kn}_{op}', op)
        S3 = sab_eval(iii, f'iii_{kn}_{op}', op)
    # 반박: 공동출원 효과만 분리 — (ii) 키는 그대로 두고 등록 문서의 '대표 외' 공동출원인만 추가 귀속
    if kn == 'K1_meta':
        SAB_III = sab_eval(iii, 'iii_K1_ge_again')
        # 공동출원 문서 비율(창별, 양성 셀)
        nco = ap.groupby('rgst_no').size()
        dk['co'] = dk['rgst_no'].map(nco).fillna(0) > 1
        pos = set(P.index[P['yb'] == 1])
        res['coapp_share_pos_cells'] = {w: float(dk[(dk.w == w) & dk.cell.isin(pos) & dk.rgst_no.notna()]['co'].mean()) for w in ['past', 'fut']}
        # 반박: 대칭 처리 — 결과창·과거창 등록 비율 비대칭을 없애기 위해 공동출원 귀속을 '두 창 모두 등록 문서가 있는 셀'로 한정해 비교
        flips_in = SAB_III & ~SAB_I; flips_out = SAB_I & ~SAB_III
        res['flips_i_to_iii'] = dict(n_in=int(flips_in.sum()), n_out=int(flips_out.sum()), cells_in=list(P.index[flips_in]))
        fi = P[flips_in]
        res['flips_in_detail'] = fi[['D', 'past', 'fut', 'p', 'p_base']].reset_index().to_dict('records')
        # 반박: 빼도 skill의 차가 넘는 것이 4셀 때문인지 — 4셀 중 1셀씩만 넣었을 때 skill
        base_ex = SAB_I.copy(); one = {}
        for c in P.index[flips_in]:
            m = base_ex.copy(); m[c] = True
            one[c] = float(skill(~m.values)[0])
        res['skill_excl_add_one_flip'] = one
        # 반박: 경계 tie(Δa == 0.5D) 셀 중 (iii)에서 새로 들어온 셀이 tie인지
        cr = iii.drop_duplicates(['cell', 'appl_no', 'w', 'key'])
        g = cr.assign(v=np.where(cr['w'] == 'fut', 1, -1)).groupby(['cell', 'key'])['v'].sum()
        mx = g.groupby(level=0).max()
        res['flips_in_maxdelta_vs_halfD'] = {c: [int(mx.get(c, 0)), float(0.5 * P.loc[c, 'D'])] for c in P.index[flips_in]}
# 부트스트랩: 빼도 skill 차(iii − i)의 표본 변동(참고: 문턱 0.005가 잡음보다 큰지)
rng = np.random.default_rng(20261017)
yb = P['yb'].values; p = P['p'].values; pb = P['p_base'].values
si = SAB_I.values; s3 = SAB_III.values
diffs = []
for b in range(300):
    ix = rng.integers(0, len(P), len(P))
    def sk(mask):
        k = ix[~mask[ix]]
        a = roc_auc_score(yb[k], p[k]); bb = roc_auc_score(yb[k], pb[k]); return (a - bb) / (1 - bb)
    diffs.append(sk(s3) - sk(si))
res['boot_diff_iii_minus_i'] = dict(mean=float(np.mean(diffs)), sd=float(np.std(diffs)), q025=float(np.quantile(diffs, .025)), q975=float(np.quantile(diffs, .975)), frac_gt_0005=float(np.mean(np.array(diffs) > 0.005)))
res['sec'] = round(time.time() - T, 1)
json.dump(res, open(os.path.join(OUT, 'vC_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
for k, v in res.items(): print(k, v)
