# -*- coding: utf-8 -*-
"""T1 의미 검정: 기업이 정부과제 성과 특허를 (목표 산업 쪽에) 갖고 있으면 (기업×KSIC) 첫 진입 확률이 높은가.
- 평가행: T1 R3 개발 원점 패널 `R3/cache/o{c}/prep_te.parquet`(R3 감사 레이블; R2 캐시는 y_u 0.26% 불일치) c=2008..2012 (진입창 (c,c+5] ≤2017, 읽기만). 레이블 = y_u(공개∪등록, R2 정의).
- 챔피언 점수: T1 `R3/runs/R3E11/o{c}.parquet`(px·pl, blend 0.9, prior_any 제외 — R2 score_origin 규칙) 읽기만.
- 기업 특허: T1 `cache/ins_firm.parquet`(범위안)·`cache/oos.parquet`(범위밖, self) — 등록번호. 출원번호↔등록번호: `cache/ins_pat`·`cache/rl_pat`.
- 시점: 출원연도 ≤ c 이고 NTIS 성과 레코드 Year ≤ c 인 정부 기원 특허만. 분류→KSIC 배분표는 레코드 Year ≤ c-2.
🔒 c ≤ 2012 만(assert). 출력: ntis_T_work/ntis_T1_*.csv
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from ntis_T_common import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

CUTS = [2008, 2009, 2010, 2011, 2012]
if len(sys.argv) > 1 and sys.argv[1] != 'combine':
    CUTS = [int(x) for x in sys.argv[1].split(',')]
COMBINE = len(sys.argv) > 1 and sys.argv[1] == 'combine'
CORP = os.environ.get('NTIS_T1_CORP', '0') == '1'          # 강건성: 대학·연구기관·공공 출원인 제외(기업만)
SFX = '_corp' if CORP else ''
assert max(CUTS) + 5 <= 2017
C1 = os.path.join(T1, 'cache')
K = sorted(pd.read_parquet(os.path.join(C1, 'ins_pat.parquet'), columns=['ksic']).ksic.astype(str).unique())
kidx = {k: i for i, k in enumerate(K)}

# ---- 기업 특허(등록번호) + 출원번호
ins = pd.read_parquet(os.path.join(C1, 'ins_firm.parquet'), columns=['rgst_no', 'applicant_cd', 'ksic', 'ay', 'mipc'])
oos = pd.read_parquet(os.path.join(C1, 'oos.parquet'), columns=['applicant_cd', 'rgst_no', 'ay', 'self'])
oos = oos[oos.self].drop(columns='self')
fp = pd.concat([ins.assign(scope='in'), oos.assign(scope='out', ksic=None, mipc=None)], ignore_index=True)
fp = fp[fp.rgst_no.astype(str).str[:2] == '10']
a2r = pd.concat([pd.read_parquet(os.path.join(C1, 'ins_pat.parquet'), columns=['appl_no', 'rgst_no']),
                 pd.read_parquet(os.path.join(C1, 'rl_pat.parquet'), columns=['appl_no', 'rgst_no'])]).dropna().drop_duplicates('rgst_no')
a2r['appl_no'] = a2r.appl_no.astype(str).str.replace('-', '', regex=False)
fp = fp.merge(a2r, on='rgst_no', how='left')
print('기업 특허 행', len(fp), '출원번호 확보', round(float(fp.appl_no.notna().mean()), 3), flush=True)

# ---- NTIS KR 특허 성과 → 분류·주체·IPC
rp = pd.read_parquet(os.path.join(EXT, 'ntis_rpatent_raw.parquet'),
                     columns=['num_digits', 'RegistType', 'Year', 'is_kr', 'IprType', 'ProjectID', 'PerformAgent', 'ipc_codes'])
rp = rp[(rp.is_kr == True) & (rp.IprType.astype(str) == '특허') & (rp.num_digits.str.len() == 13)].copy()
rp['yr'] = pd.to_numeric(rp.Year, errors='coerce').astype(float)
rp['grp'] = rp.PerformAgent.astype(str).map(GRP_MAP).fillna('oth')
pj = pd.read_parquet(os.path.join(EXT, 'ntis_project_raw.parquet'), columns=['pjt_no', 'cls_L_canon'])
pj['pjt_no'] = pj.pjt_no.astype(str); cls_of = pj.drop_duplicates('pjt_no').set_index('pjt_no').cls_L_canon.astype(str)
rp['cls'] = rp.ProjectID.astype(str).map(cls_of); rp.loc[rp.cls.isin(['NA', '<NA>', 'nan']), 'cls'] = np.nan
g_r = fp.merge(rp[rp.RegistType.astype(str) == '등록'], left_on='rgst_no', right_on='num_digits')
g_a = fp[fp.appl_no.notna()].merge(rp[rp.RegistType.astype(str) == '출원'], left_on='appl_no', right_on='num_digits')
G = pd.concat([g_r, g_a], ignore_index=True)[['applicant_cd', 'rgst_no', 'ay', 'scope', 'ksic', 'mipc', 'yr', 'grp', 'cls', 'ipc_codes']]
print('정부 기원 기업특허 레코드', len(G), '고유 특허', G.rgst_no.nunique(), '기업', G.applicant_cd.nunique(), flush=True)

# ---- 분류→KSIC 배분표 재료(제공 KR 연결)
lk = pd.read_parquet(os.path.join(EXT, 'ntis_patent_link.parquet'), columns=['appl_no', 'ksic', 'Year', 'ProjectID'])
lk['yr'] = pd.to_numeric(lk.Year, errors='coerce').astype(float); lk['cls'] = lk.ProjectID.astype(str).map(cls_of)
lk = lk[lk.cls.notna() & ~lk.cls.isin(['NA', '<NA>', 'nan'])]
rp_cls = rp[rp.cls.notna()]

# ---- KSIC IPC 서브클래스 프로파일(제공 KR, 출원연도 ≤ c)
src = pd.read_parquet(os.path.join(T1, 'R3', 'cache', 'src_kr_ipc.parquet'), columns=['ksic', 'ay', 'ipc_all'])
src = src[src.ipc_all.notna()]
src['sc'] = src.ipc_all.astype(str).str.split('|').apply(lambda v: sorted({x.replace(' ', '')[:4] for x in v if len(x) >= 4}))
src = src.explode('sc')
src = src[src.sc.notna()]


def sc_list(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    return sorted({x.strip().replace(' ', '')[:4] for x in str(s).replace('|', ',').split(',') if len(x.strip()) >= 4})


G['sc'] = G.ipc_codes.apply(sc_list)
ins['sc'] = ins.mipc.apply(sc_list)

champ_cfg = json.load(open(os.path.join(T1, 'R3', 'champion_A.json'), encoding='utf-8'))
BW = champ_cfg['config'].get('blend_w', 0.9)
CTRL = ['lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 't_mean', 't_max', 't_rw', 't_n20', 'has_dart']
FE_FEATS = ['g_any', 'g_ln', 'g_share', 'g_self', 'g_coop', 'g_tgt_ipc', 'g_tgt_cls', 'g_tgt_top']
desc, regs, cvs = [], [], []
for c in ([] if COMBINE else CUTS):
    d = pd.read_parquet(os.path.join(T1, 'R3', 'cache', f'o{c}', 'prep_te.parquet'),
                        columns=['firm', 'k', 'y_u', 'newc', 'prior_any'] + CTRL)
    d = d[~d.prior_any.values].reset_index(drop=True)
    d['firm'] = d.firm.astype(str)
    if CORP:
        fm = pd.read_parquet(os.path.join(C1, 'firm_meta.parquet'), columns=['applicant_cd', 'applicant_name'])
        bad = fm.applicant_name.astype(str).str.contains('대학|산학협력|학교|연구원|연구소|연구회|재단|진흥원|공사|공단|국방|청$|부$|KAIST|과학기술원|센터|협회|조합', regex=True)
        d = d[~d.firm.isin(set(fm.applicant_cd.astype(str)[bad]))].reset_index(drop=True)
    firms = pd.Index(d.firm.unique())
    # 기업 정부 기원 특허(시점 규칙)
    g = G[(G.ay <= c) & (G.yr <= c) & G.applicant_cd.astype(str).isin(firms)].copy()
    g['applicant_cd'] = g.applicant_cd.astype(str)
    tot = fp[(fp.ay <= c)].assign(a=lambda x: x.applicant_cd.astype(str)).drop_duplicates(['a', 'rgst_no']).groupby('a').size()
    gp = g.drop_duplicates(['applicant_cd', 'rgst_no'])
    gn = gp.groupby('applicant_cd').size()
    gs = gp[gp.grp.isin(['big', 'sme'])].groupby('applicant_cd').rgst_no.nunique()
    gc = gp[gp.grp.isin(['univ', 'gri'])].groupby('applicant_cd').rgst_no.nunique()
    gc2 = g[g.grp.isin(['univ', 'gri'])].groupby('applicant_cd').rgst_no.nunique()
    gs2 = g[g.grp.isin(['big', 'sme'])].groupby('applicant_cd').rgst_no.nunique()
    fid = pd.Series(np.arange(len(firms)), index=firms)
    n_g = gn.reindex(firms).fillna(0).values; n_t = tot.reindex(firms).fillna(0).values
    # KSIC 프로파일(c 이하)
    sp = src[src.ay <= c].groupby(['ksic', 'sc']).size().unstack(fill_value=0)
    sp.index = sp.index.astype(str); sp = sp.reindex(K).fillna(0)
    scs = pd.Index(sp.columns); scix = {s: i for i, s in enumerate(scs)}
    Pk = sp.values / np.maximum(np.linalg.norm(sp.values, axis=1, keepdims=True), 1e-9)       # 50 × SC
    Pshare = sp.values / np.maximum(sp.values.sum(1, keepdims=True), 1e-9)

    def firm_vec(frame, col_firm):
        rows_, cols_ = [], []
        for a, lst in zip(frame[col_firm].values, frame.sc.values):
            for s in lst:
                if s in scix:
                    rows_.append(fid[a]); cols_.append(scix[s])
        M = np.zeros((len(firms), len(scs)), np.float32)
        np.add.at(M, (np.array(rows_, int), np.array(cols_, int)), 1.0)
        return M
    Mg = firm_vec(gp[gp.applicant_cd.isin(firms)], 'applicant_cd')
    ia = ins[(ins.ay <= c)].assign(a=lambda x: x.applicant_cd.astype(str))
    ia = ia[ia.a.isin(firms)]
    Ma = firm_vec(ia, 'a')
    ng = np.linalg.norm(Mg, axis=1, keepdims=True); na_ = np.linalg.norm(Ma, axis=1, keepdims=True)
    cos_g = (Mg / np.maximum(ng, 1e-9)) @ Pk.T; cos_a = (Ma / np.maximum(na_, 1e-9)) @ Pk.T           # 기업 × 50
    top_g = (Mg / np.maximum(Mg.sum(1, keepdims=True), 1e-9)) @ Pshare.T                            # 정부특허 서브클래스가 k 프로파일에서 차지하는 비중
    # 분류→KSIC 배분표(레코드 Year ≤ c-2) 리프트
    den = rp_cls[rp_cls.yr <= c - 2].groupby('cls').size()
    lkk = lk[lk.yr <= c - 2].copy(); lkk['m'] = 1.0
    num = lkk.groupby(['cls', 'ksic']).m.sum().unstack(fill_value=0.0).reindex(index=den.index, fill_value=0.0)
    num.columns = num.columns.astype(str); num = num.reindex(columns=K, fill_value=0.0)
    Sk = (num.values / den.values[:, None])                                   # cls × 50
    sbar = num.values.sum(0) / max(den.values.sum(), 1)
    clsix = {cc: i for i, cc in enumerate(den.index)}
    gcl = gp[gp.cls.isin(den.index) & gp.applicant_cd.isin(firms)]
    Mc = np.zeros((len(firms), len(den)), np.float32)
    np.add.at(Mc, (fid[gcl.applicant_cd].values, gcl.cls.map(clsix).values.astype(int)), 1.0)
    has_c = Mc.sum(1) > 0
    lift = np.log(((Mc / np.maximum(Mc.sum(1, keepdims=True), 1e-9)) @ Sk + 1e-4) / (sbar + 1e-4))
    lift[~has_c] = 0.0
    if den.sum() < 1000:                                                     # c-2 < 2007: 배분표 재료 없음
        lift[:] = np.nan
    fi = fid[d.firm].values; ki = d.k.values.astype(int)
    d['g_any'] = (n_g[fi] > 0).astype(float)
    d['g_ln'] = np.log1p(n_g[fi])
    d['g_share'] = n_g[fi] / np.maximum(n_t[fi], 1)
    d['g_self'] = (gs.reindex(firms).fillna(0).values[fi] > 0).astype(float)
    d['g_coop'] = (gc.reindex(firms).fillna(0).values[fi] > 0).astype(float)
    d['g_tgt_ipc'] = np.where(ng[fi, 0] > 0, cos_g[fi, ki], 0.0)
    d['g_tgt_top'] = np.where(ng[fi, 0] > 0, top_g[fi, ki], 0.0)
    d['g_tgt_cls'] = lift[fi, ki]
    d['c_all_ipc'] = cos_a[fi, ki]
    # --- 기술통계
    base = d.y_u.mean()
    for f in ['g_any', 'g_self', 'g_coop']:
        v = d[f] == 1
        desc.append(dict(c=c, feat=f, n=len(d), share_pairs=round(float(v.mean()), 4), firms=int(d.firm[v].nunique()),
                         entry_flag=round(float(d.y_u[v].mean()), 5), entry_other=round(float(d.y_u[~v].mean()), 5)))
    gg = d[d.g_any == 1]
    q = gg.g_tgt_ipc.rank(pct=True)
    desc.append(dict(c=c, feat='g_tgt_ipc 상위25%(정부특허 보유 기업 안)', n=len(gg), share_pairs=round(float((q > 0.75).mean()), 4),
                     firms=int(gg.firm.nunique()), entry_flag=round(float(gg.y_u[q > 0.75].mean()), 5), entry_other=round(float(gg.y_u[q <= 0.25].mean()), 5)))
    # --- LPM, KSIC 고정효과, 기업 군집 (계수 ×100 = %p)
    dz = d.copy()
    for cc_ in CTRL + ['c_all_ipc', 'g_ln', 'g_share', 'g_tgt_ipc', 'g_tgt_top', 'g_tgt_cls']:
        dz[cc_] = zscore(dz[cc_].astype(float))
    dz['kk'] = dz.k.astype(str)
    specs = {'M1 정부특허 보유': ['g_any'],
             'M2 수행 유형': ['g_self', 'g_coop'],
             'M3 목표쪽 IPC(보유기업 전체 IPC 적합도 통제)': ['g_any', 'g_tgt_ipc', 'c_all_ipc'],
             'M4 목표쪽 비중': ['g_any', 'g_tgt_top', 'c_all_ipc'],
             'M5 과제분류→KSIC 리프트': ['g_any', 'g_tgt_cls'],
             'M6 정부특허 비율': ['g_share', 'c_all_ipc']}
    for nm, xs in specs.items():
        if any(dz[x_].isna().all() for x_ in xs):
            continue
        r = fe_ols(dz, 'y_u', xs + CTRL, 'kk', 'firm')
        for x in xs:
            regs.append(dict(c=c, model=nm, x=x, coef_pp=round(float(r.loc[x, 'coef']) * 100, 4), t=round(float(r.loc[x, 't']), 2),
                             base_pp=round(base * 100, 3), n=int(r.n.iloc[0]), n_firm=int(r.n_clu.iloc[0])))
    # --- 챔피언 대비 증분: 기업 2분할 교차(한쪽 기업으로 결합계수 학습 → 다른 쪽 평가)
    o = pd.read_parquet(os.path.join(T1, 'R3', 'runs', 'R3E11', f'o{c}.parquet'), columns=['firm', 'k', 'px', 'pl', 'y_u', 'prior_any'])
    o = o[~o.prior_any.values]; o['firm'] = o.firm.astype(str)
    m = d.merge(o[['firm', 'k', 'px', 'pl', 'y_u']], on=['firm', 'k'], suffixes=('', '_o'))
    assert (m.y_u == m.y_u_o).mean() > 0.999
    s = BW * rankdata(m.px) / len(m) + (1 - BW) * rankdata(m.pl) / len(m)
    m['zc'] = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
    m['zx'] = np.log(np.clip(m.px, 1e-6, 1 - 1e-6) / (1 - np.clip(m.px, 1e-6, 1 - 1e-6)))
    fold = (pd.util.hash_array(m.firm.values.astype(object)) % 2).astype(int)
    a_champ = roc_auc_score(m.y_u, s)
    for nm, xs in {'g_any': ['g_any'], 'g_any+g_ln': ['g_any', 'g_ln'], 'g_tgt_ipc': ['g_any', 'g_tgt_ipc'],
                   'g_tgt_top': ['g_any', 'g_tgt_top'], 'g_tgt_cls': ['g_any', 'g_tgt_cls'], 'g_self+g_coop': ['g_self', 'g_coop']}.items():
        if any(m[x_].isna().all() for x_ in xs):
            continue
        pred = np.zeros(len(m)); pred0 = np.zeros(len(m)); coefs = []
        for f_ in (0, 1):
            tr, te = fold != f_, fold == f_
            X = m[['zx'] + xs].values
            lr0 = LogisticRegression(C=1e4, max_iter=1000).fit(m.loc[tr, ['zx']].values, m.y_u[tr])
            lr1 = LogisticRegression(C=1e4, max_iter=1000).fit(X[tr], m.y_u[tr])
            pred0[te] = lr0.decision_function(m.loc[te, ['zx']].values); pred[te] = lr1.decision_function(X[te]); coefs.append(lr1.coef_[0][1:])
        a0 = roc_auc_score(m.y_u, pred0); a1 = roc_auc_score(m.y_u, pred)
        cvs.append(dict(c=c, feats=nm, n=len(m), auc_champ_blend=round(a_champ, 5), auc_px_cv=round(a0, 5), auc_plus_cv=round(a1, 5),
                        d_auc=round(a1 - a0, 5), coef=';'.join(f'{v:.3f}' for v in np.mean(coefs, 0))))
    print(c, 'rows', len(d), 'g_any 쌍 비중', round(float(d.g_any.mean()), 4), peak_mem_gb(), flush=True)
    for nm_, L_ in (('desc', desc), ('fe', regs), ('cv', cvs)):
        pd.DataFrame([r_ for r_ in L_ if r_['c'] == c]).to_csv(out(f'ntis_T1_part_{nm_}_{c}{SFX}.csv'), index=False, encoding='utf-8-sig')
if not COMBINE:
    sys.exit(0)
for nm_, L_ in (('desc', desc), ('fe', regs), ('cv', cvs)):
    for c in [2008, 2009, 2010, 2011, 2012]:
        L_.extend(pd.read_csv(out(f'ntis_T1_part_{nm_}_{c}{SFX}.csv'), encoding='utf-8-sig').to_dict('records'))

pd.DataFrame(desc).to_csv(out('ntis_T1_desc' + SFX + '.csv'), index=False, encoding='utf-8-sig')
R = pd.DataFrame(regs); R.to_csv(out('ntis_T1_fe' + SFX + '.csv'), index=False, encoding='utf-8-sig')
CV = pd.DataFrame(cvs); CV.to_csv(out('ntis_T1_champ_cv' + SFX + '.csv'), index=False, encoding='utf-8-sig')
S = R.groupby(['model', 'x']).agg(coef_pp=('coef_pp', 'mean'), base_pp=('base_pp', 'mean'), n_pos=('coef_pp', lambda v: int((v > 0).sum())),
                                  n_orig=('c', 'size'), t_min=('t', 'min'), t_max=('t', 'max')).reset_index()
S.to_csv(out('ntis_T1_summary' + SFX + '.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
print(pd.DataFrame(desc).to_string()); print(S.to_string())
print(CV.groupby('feats').agg(d_auc=('d_auc', 'mean'), win=('d_auc', lambda v: int((v > 0).sum()))).to_string()); print(CV.to_string())
