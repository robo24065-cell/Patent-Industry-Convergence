# -*- coding: utf-8 -*-
"""(A) 2단계: 재현 관문(A0) → 정의 (i)–(iv)·(R)·(H)·(P)별 SAB와 '빼도 skill' → a_concentration.csv, cache/a2_results.json
재학습 없음. 동결 예측 p·p_base(R3\\preds\\FINAL_R3_t0_2021.parquet)만 재채점."""
import os, sys, json, hashlib, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
t_start = time.time(); OUT = {}
T0 = 2021

# ---------- 패널·레이블 (C04 origin 규칙) ----------
pan = pd.read_parquet(os.path.join(CACHE, 'a_panel.parquet')); pan['cell'] = pan.ksic + '|' + pan.mg
tot = pan[(pan.year >= T0 - 2) & (pan.year <= T0)].groupby('cell').n.sum()
kr = pan[pan.cc == 'KR']
krW = kr[(kr.year >= T0 - 2) & (kr.year <= T0)].groupby('cell').n.sum()
krF = kr[(kr.year >= T0 + 1) & (kr.year <= T0 + 3)].groupby('cell').n.sum()
krP = kr[(kr.year >= T0 - 5) & (kr.year <= T0 - 3)].groupby('cell').n.sum()
d = pd.DataFrame(dict(tot_W=tot)); d = d[d.tot_W >= 50].copy()
d['kr_W'] = krW.reindex(d.index).fillna(0); d['kr_fut'] = krF.reindex(d.index).fillna(0); d['kr_P'] = krP.reindex(d.index).fillna(0)
d = d.reset_index().rename(columns={'index': 'cell'}).sort_values('cell').reset_index(drop=True)
d['y'] = np.log((d.kr_fut + 1) / (d.kr_W + 1)); d['D'] = d.kr_fut - d.kr_W
hk = np.array([int(hashlib.md5(s.encode()).hexdigest()[:8], 16) for s in d.cell], dtype=np.int64)
o = np.lexsort((hk, -d.D.values, -d.y.values)); k = int(round(0.10 * len(d))); d['yb_r'] = 0; d.loc[o[:k], 'yb_r'] = 1
P = pd.read_parquet(os.path.join(R3, 'preds', 'FINAL_R3_t0_2021.parquet'))
m = P.merge(d, on='cell', how='inner', suffixes=('', '_r'))
def auc(y, p): return float(roc_auc_score(y, p))
def sk(y, p, pb):
    a, b = auc(y, p), auc(y, pb); return dict(auc=a, auc_base=b, skill=(a - b) / (1 - b), n=int(len(y)), pos=int(np.sum(y)))
base = sk(m.yb.values, m.p.values, m.p_base.values.astype(float))
top = np.zeros(len(m), bool); top[np.argsort(-m.p.values)[:504]] = True
hit = top & (m.yb.values == 1)
gate = dict(n_elig=int(len(d)), n_merged=int(len(m)), pos=int(m.yb.sum()), yb_mismatch=int((m.yb != m.yb_r).sum()), y_maxdiff=float(np.abs(m.y - m.y_r).max()),
            krW_maxdiff=float(np.abs(m.kr_W - m.kr_W_r).max()), auc=base['auc'], auc_base=base['auc_base'], skill=base['skill'],
            hits_top10=int(hit.sum()), hits_krW_le1=int((hit & (m.kr_W_r.values <= 1)).sum()), hits_D_le3=int((hit & (m.D.values <= 3)).sum()))
gate['hits_krW_le1_share'] = gate['hits_krW_le1'] / gate['hits_top10']; gate['hits_D_le3_share'] = gate['hits_D_le3'] / gate['hits_top10']
m['kr_W'] = m.kr_W_r; m = m.drop(columns=['kr_W_r', 'yb_r', 'y_r'])
m['stratum'] = np.select([m.kr_W == 0, m.kr_W == 1], ['S0', 'S1'], 'S2')
print('GATE', gate, flush=True)

# ---------- 문서표·키 ----------
kd = pd.read_parquet(os.path.join(CACHE, 'a_krdocs.parquet')); kd['cell'] = kd.ksic + '|' + kd.mg
kd = kd[kd.cell.isin(set(m.cell)) & (kd.year >= T0 - 2) & (kd.year <= T0 + 3)].copy()
kd['w'] = np.where(kd.year <= T0, 'past', 'fut')
nm_map = {x: n1(x) for x in kd.app_raw.fillna('').unique()}; kd['kname'] = kd.app_raw.fillna('').map(nm_map)
rfa = pd.read_parquet(os.path.join(CACHE, 'a_rfapp.parquet'))
rfa['n1'] = rfa.applicantName.fillna('').map(n1); rfa['code'] = rfa.applicantCd.fillna('').astype(str)
rfa.loc[rfa.code == '', 'code'] = 'NOCD:' + rfa.n1
rfa = rfa.sort_values(['rgst_no', 'seq'])
kd['reg'] = kd.rgst_no.notna()
# 문서별 등록원부 출원인 수(고유 코드)
ncode = rfa.groupby('rgst_no').code.nunique(); kd['n_rf_app'] = kd.rgst_no.map(ncode).fillna(0).astype(int); kd['joint'] = kd.n_rf_app >= 2
# (ii) 등록 문서: 제공 이름과 N1이 같은 등록원부 출원인의 코드, 없으면 seq 최소
# 변경 기록 2: 문서별 코드 배정을 문서표 전체(공개 2016–24)로 만든다(창 밖 등록 문서가 'C:?'로 뭉치던 (P)·(M) 기반 오류 수정; 창 안 문서의 배정은 같은 규칙이라 불변)
_kfull = pd.read_parquet(os.path.join(CACHE, 'a_krdocs.parquet'), columns=['appl_no', 'rgst_no', 'app_raw']).drop_duplicates('appl_no')
for x in _kfull.app_raw.fillna('').unique():
    if x not in nm_map: nm_map[x] = n1(x)
_kfull['kname'] = _kfull.app_raw.fillna('').map(nm_map)
docs_u = _kfull[['appl_no', 'rgst_no', 'kname']]
reg_u = docs_u[docs_u.rgst_no.notna()].merge(rfa[['rgst_no', 'seq', 'code', 'n1']], on='rgst_no', how='left')
reg_u['match'] = reg_u.kname == reg_u.n1
first = reg_u.sort_values(['appl_no', 'seq']).groupby('appl_no').head(1).set_index('appl_no').code
matched = reg_u[reg_u.match].sort_values(['appl_no', 'seq']).groupby('appl_no').head(1).set_index('appl_no').code
code_of = first.copy(); code_of.loc[matched.index] = matched
OUT['ii_name_match_rate_reg'] = float(len(matched) / len(first))
OUT['ii_seq0_eq_provided_reg'] = float((reg_u[reg_u.seq == reg_u.groupby('appl_no').seq.transform('min')].match).mean())
jdocs = reg_u.groupby('appl_no').code.nunique(); jd = set(jdocs[jdocs >= 2].index)
rj = reg_u[reg_u.appl_no.isin(jd)]
OUT['ii_joint_provided_is_seq0'] = float(rj[rj.seq == rj.groupby('appl_no').seq.transform('min')].match.mean())
OUT['ii_joint_provided_matches_any'] = float(rj.groupby('appl_no').match.any().mean())
# 이름 → 최빈 코드(몫 ≥0.5): 문서표 전체 연도(2016–24) 등록 문서 기준
kd_all = pd.read_parquet(os.path.join(CACHE, 'a_krdocs.parquet'), columns=['appl_no', 'rgst_no', 'app_raw']).drop_duplicates('appl_no')
kd_all['kname'] = kd_all.app_raw.fillna('').map(lambda x: nm_map.get(x) if x in nm_map else n1(x))
ra = kd_all[kd_all.rgst_no.notna()].merge(rfa[['rgst_no', 'seq', 'code', 'n1']], on='rgst_no')
ra['match'] = ra.kname == ra.n1
# 변경 기록 1(가): 등록 문서마다 배정 코드 = 이름 일치 코드(seq 최소), 없으면 seq 최소 코드 → 제공 이름별 최빈(몫 ≥0.5)
ra = ra.sort_values(['appl_no', 'match', 'seq'], ascending=[True, False, True]).groupby('appl_no').head(1)
OUT['name2code_basis_docs'] = int(len(ra)); OUT['name2code_basis_nomatch_share'] = float((~ra.match).mean())
nc = ra.groupby(['kname', 'code']).size().rename('n').reset_index()
nc['sh'] = nc.n / nc.groupby('kname').n.transform('sum'); nc = nc.sort_values(['kname', 'n'], ascending=[True, False]).groupby('kname').head(1)
name2code = nc[nc.sh >= 0.5].set_index('kname').code.to_dict()
code2name = rfa.groupby('code').n1.agg(lambda s: s.value_counts().index[0]).to_dict()
def key_ii(row_appl, row_reg, kname):
    if row_reg: return 'C:' + code_of.get(row_appl, '?')
    c = name2code.get(kname); return 'C:' + c if c else 'N:' + kname
kd['k_i'] = 'N:' + kd.kname
kd['k_ii'] = [key_ii(a, r, n) for a, r, n in zip(kd.appl_no.values, kd.reg.values, kd.kname.values)]
OUT['ii_unreg_mapped_to_code'] = float(kd[~kd.reg].k_ii.str.startswith('C:').mean())

# ---------- 집단 규칙 ----------
HMG = ['현대자동차', '기아', '현대모비스', '현대제철', '현대위아', '현대트랜시스', '현대케피코', '현대오토에버', '현대로템', '현대건설', '현대엔지비', '현대엔지니어링', '현대파워텍', '현대다이모스', '현대오트론', '현대엠엔소프트']
HDH = ['에이치디현대', '현대중공업', '한국조선해양', '현대일렉트릭', '현대건설기계', '현대두산인프라코어', '현대오일뱅크', '현대삼호중공업', '현대미포조선', '에이치디한국조선해양']
PRE = [('삼성', '삼성'), ('엘지', '엘지'), ('에스케이', '에스케이'), ('한화', '한화'), ('포스코', '포스코'), ('롯데', '롯데'), ('두산', '두산'), ('씨제이', '씨제이'),
       ('에이치엘', '에이치엘'), ('만도', '에이치엘'), ('효성', '효성'), ('코오롱', '코오롱'), ('케이티', '케이티')]
def grp(nm):
    for p in HDH:
        if nm.startswith(p): return 'G:에이치디현대'
    for p in HMG:
        if nm.startswith(p): return 'G:현대차'
    for p, g in PRE:
        if nm.startswith(p): return 'G:' + g
    return None
C04G = [('삼성', '삼성'), ('엘지', '엘지'), ('에스케이', '에스케이'), ('현대', '현대'), ('기아', '현대'), ('에이치디현대', '에이치디현대'), ('한화', '한화'), ('포스코', '포스코'),
        ('롯데', '롯데'), ('두산', '두산'), ('씨제이', '씨제이'), ('에이치엘', '에이치엘'), ('만도', '에이치엘'), ('효성', '효성'), ('코오롱', '코오롱'), ('케이티', '케이티')]
C04ALIAS = {'엘지에너지솔루션': '엘지화학', '에스케이온': '에스케이이노베이션', '에이치엘만도': '만도', '만도모빌리티솔루션즈': '만도', '두산에너빌리티': '두산중공업',
            '에이치디현대중공업': '현대중공업', '한국조선해양': '현대중공업', '에이치디한국조선해양': '현대중공업', '포스코홀딩스': '포스코', '에이치디현대인프라코어': '두산인프라코어',
            '현대자동차': '현대자동차', '에스케이시그넷': '시그넷이브이'}
def c04g(nm):
    nm = C04ALIAS.get(nm, nm)
    for p, g in C04G:
        if nm.startswith(p): return 'G:' + g
    return 'N:' + nm
def key_name(kk):
    if kk.startswith('C:'): return code2name.get(kk[2:], '')
    return kk[2:]
def to_group(kk):
    g = grp(key_name(kk)); return g if g else kk

# ---------- 귀속 표(credit) ----------
base_cols = ['cell', 'appl_no', 'year', 'w', 'reg', 'joint']
cr_i = kd[base_cols + ['k_i']].rename(columns={'k_i': 'key'})
cr_ii = kd[base_cols + ['k_ii']].rename(columns={'k_ii': 'key'})
allc = rfa.groupby('rgst_no').code.apply(lambda s: sorted(set(s))).to_dict()
rows = []
for r in kd[base_cols + ['rgst_no', 'k_ii']].itertuples(index=False):
    if r.reg:
        for c in allc.get(r.rgst_no, []): rows.append((r.cell, r.appl_no, r.year, r.w, r.reg, r.joint, 'C:' + c))
    else:
        rows.append((r.cell, r.appl_no, r.year, r.w, r.reg, r.joint, r.k_ii))
cr_iii = pd.DataFrame(rows, columns=base_cols + ['key']).drop_duplicates(['cell', 'appl_no', 'year', 'key'])
cr_iv = cr_iii.assign(key=cr_iii.key.map(lambda x: to_group(x))).drop_duplicates(['cell', 'appl_no', 'year', 'key'])
cr_ivb = cr_i.assign(key=cr_i.key.map(lambda x: c04g(x[2:]))).drop_duplicates(['cell', 'appl_no', 'year', 'key'])
cr_ii_g = cr_ii.assign(key=cr_ii.key.map(lambda x: to_group(x))).drop_duplicates(['cell', 'appl_no', 'year', 'key'])
# (P) 대치: 등록 문서(공개 2016–24 문서표 전체)에서 제공 키 a 문서의 ≥50%가 상대 c와 공동출원(a 문서 ≥20건)이면 미등록 a 문서에 c 추가
kd_allc = pd.read_parquet(os.path.join(CACHE, 'a_krdocs.parquet'), columns=['appl_no', 'rgst_no', 'app_raw']).drop_duplicates('appl_no')
kd_allc['kname'] = kd_allc.app_raw.fillna('').map(n1)
kd_allc['k_ii'] = [key_ii(a, r, n) for a, r, n in zip(kd_allc.appl_no.values, kd_allc.rgst_no.notna().values, kd_allc.kname.values)]
rg = kd_allc[kd_allc.rgst_no.notna()]
pairs = rg[['appl_no', 'rgst_no', 'k_ii']].merge(rfa[['rgst_no', 'code']], on='rgst_no')
pairs['c'] = 'C:' + pairs.code; pairs = pairs[pairs.c != pairs.k_ii].drop_duplicates(['appl_no', 'c'])
na = rg.groupby('k_ii').size(); pc = pairs.groupby(['k_ii', 'c']).size().rename('n').reset_index(); pc['sh'] = pc.n / pc.k_ii.map(na); pc['na'] = pc.k_ii.map(na)
imp = pc[(pc.sh >= 0.5) & (pc.na >= 20)]
impmap = imp.groupby('k_ii').c.apply(list).to_dict()
OUT['P_impute_pairs'] = [dict(a=key_name(r.k_ii)[:20] if grp(key_name(r.k_ii)) else 'hash:' + hashlib.md5(r.k_ii.encode()).hexdigest()[:8],
                              c=key_name(r.c)[:20] if grp(key_name(r.c)) else 'hash:' + hashlib.md5(r.c.encode()).hexdigest()[:8], share=round(float(r.sh), 3), n_a=int(r.na))
                         for r in imp.sort_values('na', ascending=False).head(15).itertuples()]
OUT['P_n_impute_keys'] = int(len(impmap))
add = []
for r in cr_iii[~cr_iii.reg].itertuples(index=False):
    for c in impmap.get(r.key, []): add.append((r.cell, r.appl_no, r.year, r.w, r.reg, r.joint, c))
cr_P = pd.concat([cr_iii, pd.DataFrame(add, columns=base_cols + ['key'])], ignore_index=True).drop_duplicates(['cell', 'appl_no', 'year', 'key'])
OUT['P_added_credits'] = len(add)

# ---------- SAB ----------
mi = m.set_index('cell')
def delta_tab(cr):
    g = cr.groupby(['cell', 'key', 'w']).size().unstack('w', fill_value=0)
    for c in ['past', 'fut']:
        if c not in g: g[c] = 0
    g['delta'] = g.fut - g.past
    return g.reset_index()
def sab_from(cr, Dser=None):
    g = delta_tab(cr); mx = g.groupby('cell').delta.max()
    Dv = (m.D if Dser is None else Dser).values
    mxv = mx.reindex(m.cell.values).fillna(0).values
    return (m.yb.values == 1) & (Dv > 0) & (mxv >= 0.5 * Dv), mxv
# 문서 수 일치 점검(귀속 표 문서 수 = 패널 D)
chk = kd.groupby(['cell', 'w']).appl_no.size().unstack('w', fill_value=0).reindex(m.cell.values).fillna(0)
OUT['doc_panel_check_maxdiff'] = float(max(np.abs(chk.past.values - m.kr_W.values).max(), np.abs(chk.fut.values - m.kr_fut.values).max()))
RES = {}; FLAGS = {}
def record(tag, s, note=''):
    yv, pv, pbv = m.yb.values, m.p.values, m.p_base.values.astype(float)
    rA = sk(yv[~s], pv[~s], pbv[~s]); yB = yv.copy(); yB[s] = 0; rB = sk(yB, pv, pbv)
    pos = yv == 1; D = m.D.values
    RES[tag] = dict(note=note, n_sab=int(s.sum()), rate=float(s.sum() / pos.sum()), skill_excl=rA['skill'], auc_excl=rA['auc'], auc_base_excl=rA['auc_base'],
                    d_skill_excl_vs_04378=rA['skill'] - 0.4378, skill_B=rB['skill'],
                    rate_by_stratum={st: float(s[(m.stratum == st).values].sum() / (pos & (m.stratum == st).values).sum()) for st in ['S0', 'S1', 'S2']},
                    rate_D5=float((s & (D >= 5)).sum() / (pos & (D >= 5)).sum()), hits_top10_sab=int((top & s).sum()))
    FLAGS[tag] = s
    print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in RES[tag].items() if k not in ('rate_by_stratum',)}, flush=True)
s_i, mx_i = sab_from(cr_i); record('i_name', s_i, '제공 이름 N1(원판)')
s_ii, _ = sab_from(cr_ii); record('ii_code', s_ii, '등록원부 코드(미등록은 이름→코드)')
s_iii, mx_iii = sab_from(cr_iii); record('iii_alljoint', s_iii, '등록 문서 공동출원인 전원 귀속, 미등록은 (ii)')
s_iv, _ = sab_from(cr_iv); record('iv_group_on_iii', s_iv, '(iii) + 기업집단 묶음')
s_ivb, _ = sab_from(cr_ivb); record('ivb_C04G_on_name', s_ivb, 'C04 G 규칙(이름 기준) 재현')
s_iig, _ = sab_from(cr_ii_g); record('iv_group_on_ii', s_iig, '(ii) + 기업집단 묶음(공동출원 미반영)')
s_P, _ = sab_from(cr_P); record('P_impute', s_P, '(iii) + 미등록 문서 공동상대 대치(≥50%, ≥20건)')
# (R) 등록 부분집합
regd = kd[kd.reg]
cntR = regd.groupby(['cell', 'w']).appl_no.size().unstack('w', fill_value=0).reindex(m.cell.values).fillna(0)
DR = pd.Series(cntR.fut.values - cntR.past.values, index=m.index)
s_Ri, _ = sab_from(cr_i[cr_i.reg], DR); record('R_reg_name', s_Ri, '등록 문서만, 이름 키')
s_Rii, _ = sab_from(cr_ii[cr_ii.reg], DR); record('R_reg_code', s_Rii, '등록 문서만, 코드 키(seq/이름 일치)')
s_Riii, _ = sab_from(cr_iii[cr_iii.reg], DR); record('R_reg_alljoint', s_Riii, '등록 문서만, 공동출원인 전원')
OUT['R_cells_DR_le0_among_pos'] = int(((m.yb == 1).values & (DR.values <= 0)).sum())
# (H) 최악 범위: (iii) 귀속 기준, 미등록 문서의 공동출원인 미지
g = delta_tab(cr_iii)
ur = kd[~kd.reg].groupby(['cell', 'w']).appl_no.size().unstack('w', fill_value=0)
for c in ['past', 'fut']:
    if c not in ur: ur[c] = 0
ua = cr_iii[~cr_iii.reg].groupby(['cell', 'key', 'w']).size().unstack('w', fill_value=0)
for c in ['past', 'fut']:
    if c not in ua: ua[c] = 0
g = g.merge(ua.rename(columns={'past': 'u_past_a', 'fut': 'u_fut_a'}).reset_index(), on=['cell', 'key'], how='left').fillna({'u_past_a': 0, 'u_fut_a': 0})
g['U_fut'] = g.cell.map(ur.fut).fillna(0); g['U_past'] = g.cell.map(ur.past).fillna(0)
g['dU'] = g.delta + (g.U_fut - g.u_fut_a); g['dL'] = g.delta - (g.U_past - g.u_past_a)
mU = np.maximum(g.groupby('cell').dU.max().reindex(m.cell.values).fillna(0).values, m.cell.map(ur.fut).fillna(0).values)
mL = g.groupby('cell').dL.max().reindex(m.cell.values).fillna(-1e9).values
D = m.D.values; pos = m.yb.values == 1
sU = pos & (D > 0) & (mU >= 0.5 * D); sL = pos & (D > 0) & (mL >= 0.5 * D)
record('H_upper', sU, '최악 상한: 미등록 결과창 문서를 최대 후보에 추가 귀속'); record('H_lower', sL, '최악 하한: 미등록 과거창 문서를 모든 후보에 추가 귀속')

# (M) 몬테카를로 대치(변경 기록 1(다)): 미등록 문서의 공동출원 여부·상대를 그 출원인의 등록 문서 분포에서 뽑는다
rgM = kd_allc[kd_allc.rgst_no.notna()][['appl_no', 'rgst_no', 'k_ii']]
plist = [['C:' + c for c in allc.get(rn, []) if 'C:' + c != a] for rn, a in zip(rgM.rgst_no.values, rgM.k_ii.values)]
rgM = rgM.assign(npart=[len(x) for x in plist]); rgM['plist'] = plist
overall_joint = float((rgM.npart > 0).mean())
agg = rgM.groupby('k_ii').agg(n_reg=('appl_no', 'size'), n_joint=('npart', lambda s: int((s > 0).sum())))
agg['rate'] = np.where(agg.n_reg >= 5, agg.n_joint / agg.n_reg, overall_joint)
pools = rgM[rgM.npart > 0].groupby('k_ii').plist.apply(list).to_dict()
U = cr_iii[~cr_iii.reg].reset_index(drop=True)
U_rate = U.key.map(agg.rate).fillna(overall_joint).values
U_pool = [pools.get(k, []) for k in U.key.values]; U_psz = np.array([len(x) for x in U_pool])
# 정수 키
allkeys = pd.Index(pd.unique(pd.concat([cr_iii.key, pd.Series([p for pl in pools.values() for e in pl for p in e])], ignore_index=True)))
cell_ix = pd.Index(m.cell.values); NK = len(allkeys) + 1
gb = delta_tab(cr_iii); gb = gb[gb.cell.isin(cell_ix)]
base_ck = cell_ix.get_indexer(gb.cell).astype(np.int64) * NK + allkeys.get_indexer(gb.key).astype(np.int64)
base_s = pd.Series(gb.delta.values, index=base_ck)
base_max = gb.groupby('cell').delta.max().reindex(m.cell.values).fillna(0).values
U_cell = cell_ix.get_indexer(U.cell).astype(np.int64); U_sign = np.where(U.w.values == 'fut', 1, -1)
rng = np.random.default_rng(20261013); MC = []
OUT['M_overall_joint_reg'] = overall_joint; OUT['M_unreg_rows'] = int(len(U)); OUT['M_unreg_with_pool_share'] = float((U_psz > 0).mean())
OUT['M_expected_joint_share_unreg'] = float(U_rate.mean())
for it in range(200):
    jt = (rng.random(len(U)) < U_rate) & (U_psz > 0)
    idx = np.where(jt)[0]; ck_add = []; sg = []
    for i in idx:
        e = U_pool[i][int(rng.random() * U_psz[i])]
        for p in e:
            ck_add.append(U_cell[i] * NK + allkeys.get_loc(p)); sg.append(U_sign[i])
    if ck_add:
        a_ = pd.Series(sg, index=np.array(ck_add, dtype=np.int64)).groupby(level=0).sum()
        tot_ = a_ + base_s.reindex(a_.index).fillna(0).values
        cm = pd.Series(tot_.values, index=tot_.index // NK).groupby(level=0).max()
        mx = base_max.copy(); ii = cm.index.values; mx[ii] = np.maximum(mx[ii], cm.values)
    else:
        mx = base_max
    s = (m.yb.values == 1) & (m.D.values > 0) & (mx >= 0.5 * m.D.values)
    yv, pv, pbv = m.yb.values, m.p.values, m.p_base.values.astype(float)
    rA = sk(yv[~s], pv[~s], pbv[~s])
    MC.append(dict(it=it, n_joint_unreg=int(jt.sum()), n_sab=int(s.sum()), rate=float(s.sum() / 504), skill_excl=rA['skill']))
MCd = pd.DataFrame(MC); MCd.to_csv(os.path.join(CACHE, 'a_M_mc.csv'), index=False)
OUT['M_mc'] = {c: dict(mean=float(MCd[c].mean()), q025=float(MCd[c].quantile(0.025)), q975=float(MCd[c].quantile(0.975)), min=float(MCd[c].min()), max=float(MCd[c].max()))
               for c in ['n_joint_unreg', 'n_sab', 'rate', 'skill_excl']}
print('M_mc', json.dumps(OUT['M_mc'], ensure_ascii=False), flush=True)

# ---------- 문서 구성 보고 ----------
pc_ = set(m.cell[m.yb == 1])
def comp(sub):
    return dict(docs=int(len(sub)), reg_share=float(sub.reg.mean()), joint_share_all=float(sub.joint.mean()), joint_share_reg=float(sub[sub.reg].joint.mean()) if sub.reg.any() else None)
OUT['composition'] = {f'{lab}_{w}': comp(kd[(kd.cell.isin(cs)) & (kd.w == w)]) for lab, cs in [('all', set(m.cell)), ('pos', pc_)] for w in ['past', 'fut']}
OUT['composition_by_year_pos'] = {int(y): comp(gg) for y, gg in kd[kd.cell.isin(pc_)].groupby('year')}
# 정의 간 전환 셀
def flips(a, b):
    return dict(out_=int((FLAGS[a] & ~FLAGS[b]).sum()), in_=int((~FLAGS[a] & FLAGS[b]).sum()))
OUT['flips_vs_i'] = {k: flips('i_name', k) for k in FLAGS if k != 'i_name'}
# 바뀐 셀의 주 출원인(i 기준 top; 대기업 집단만 이름)
gi = delta_tab(cr_i).sort_values(['cell', 'delta', 'fut', 'key'], ascending=[True, False, False, True]).groupby('cell').head(1).set_index('cell')
giii = delta_tab(cr_iii).sort_values(['cell', 'delta', 'fut', 'key'], ascending=[True, False, False, True]).groupby('cell').head(1).set_index('cell')
def lab(kk):
    if not isinstance(kk, str): return None
    nm = key_name(kk); gname = grp(nm)
    return (gname + ':' + nm[:12]) if gname else 'hash:' + hashlib.md5(kk.encode()).hexdigest()[:8]
ch = m[(FLAGS['i_name'] != FLAGS['iii_alljoint'])][['cell', 'stratum', 'D', 'kr_W', 'kr_fut']].copy()
ch['sab_i'] = FLAGS['i_name'][ch.index]; ch['sab_iii'] = FLAGS['iii_alljoint'][ch.index]
ch['top_i'] = ch.cell.map(gi.key).map(lab); ch['top_i_delta'] = ch.cell.map(gi.delta)
ch['top_iii'] = ch.cell.map(giii.key).map(lab); ch['top_iii_delta'] = ch.cell.map(giii.delta)
OUT['changed_cells_i_vs_iii'] = ch.to_dict('records')

# ---------- 판정 ----------
def verdict(tag):
    r = RES[tag]; num = abs(r['rate'] - 205 / 504) > 0.010 or abs(r['skill_excl'] - 0.4378) > 0.005
    flip = r['rate'] <= 0.30 or r['skill_excl'] < 0.3961
    return '오류(판정)' if flip else ('오류(숫자)' if num else '영향 없음')
OUT['verdict'] = {t: verdict(t) for t in RES}
OUT['gate'] = gate; OUT['base'] = base; OUT['results'] = RES
OUT['gate_pass'] = bool(gate['yb_mismatch'] == 0 and gate['y_maxdiff'] < 1e-6 and abs(base['skill'] - 0.42289) < 5e-6 and gate['hits_top10'] == 183
                        and gate['hits_krW_le1'] == 149 and gate['hits_D_le3'] == 83 and RES['i_name']['n_sab'] == 205 and abs(RES['i_name']['skill_excl'] - 0.4378) < 1e-4)
# ---------- a_concentration.csv ----------
ac = m[['cell', 'ksic', 'mg', 'stratum', 'kr_W', 'kr_fut', 'D', 'y', 'yb', 'p', 'p_base']].copy()
ac['in_top10'] = top
for t in FLAGS: ac['sab_' + t] = FLAGS[t]
ac['maxdelta_i'] = mx_i; ac['maxdelta_iii'] = mx_iii
ac['reg_share_fut'] = ac.cell.map(kd[kd.w == 'fut'].groupby('cell').reg.mean()); ac['joint_share_fut'] = ac.cell.map(kd[kd.w == 'fut'].groupby('cell').joint.mean())
ac['top_i'] = ac.cell.map(gi.key).map(lab); ac['top_iii'] = ac.cell.map(giii.key).map(lab)
ac = ac[ac.yb == 1]
summ = pd.DataFrame([dict(definition=t, **{k: v for k, v in r.items() if k != 'rate_by_stratum'}, rate_S0=r['rate_by_stratum']['S0'], rate_S1=r['rate_by_stratum']['S1'],
                          rate_S2=r['rate_by_stratum']['S2'], verdict=OUT['verdict'][t]) for t, r in RES.items()])
summ.to_csv(os.path.join(CACHE, 'a_summary_by_definition.csv'), index=False, encoding='utf-8-sig')
ac.to_csv(os.path.join(CACHE, 'a_cells_pos.csv'), index=False, encoding='utf-8-sig')
OUT['sec'] = round(time.time() - t_start, 1)
json.dump(OUT, open(os.path.join(CACHE, 'a2_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(summ[['definition', 'n_sab', 'rate', 'skill_excl', 'skill_B', 'rate_D5', 'verdict']].round(4).to_string())
print({k: OUT[k] for k in ['gate_pass', 'ii_name_match_rate_reg', 'ii_seq0_eq_provided_reg', 'ii_joint_provided_is_seq0', 'ii_joint_provided_matches_any', 'ii_unreg_mapped_to_code',
                           'P_n_impute_keys', 'P_added_credits', 'doc_panel_check_maxdiff', 'R_cells_DR_le0_among_pos']})
print(json.dumps(OUT['composition'], ensure_ascii=False)); print(OUT['flips_vs_i']); print('sec', OUT['sec'])
