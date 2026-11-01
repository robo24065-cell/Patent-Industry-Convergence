# verify_0 v1: 동결 독립 검토(libC/libM 미사용 재구현). 잠금 레이블은 계산하지 않는다.
# (i) 해시 (ii) 기준일 이후 정보 없음(로드 기록·잠금 피처 재계산·M SMD) (iii) 잠금 레이블 미열람 (iv) 위험집합 = PREREG_final
import sys, os, re, json, hashlib, calendar
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
ROOT = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
WD = os.path.join(ROOT, '분석', '검증루프_20261012', 'C05', 'C05-T4')
CA_ = os.path.join(WD, 'cache')
EXTD = os.path.join(ROOT, '분석', '모델경쟁_20260927', '_shared', 'ext')
OUT = {}
def h(p):
    x = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            x.update(c)
    return x.hexdigest()
# ---------- (i) 해시
spec = json.load(open(os.path.join(WD, 'final_spec_C05T4.json'), encoding='utf-8'))
bad = []
for rel, hv in spec['code_sha256'].items():
    if h(os.path.join(WD, rel)) != hv:
        bad.append(rel)
fs_hash_rec = open(os.path.join(WD, 'frozen_scores.sha256'), encoding='utf-8').read().split()[0]
fs_ok = h(os.path.join(WD, 'frozen_scores.parquet')) == fs_hash_rec
pf_ok = h(os.path.join(WD, 'PREREG_final.txt')) == open(os.path.join(WD, 'PREREG_final.sha256'), encoding='utf-8').read().split()[0]
OUT['i_hash'] = dict(code_mismatch=bad, frozen_scores_ok=fs_ok, prereg_final_ok=pf_ok, spec_prereg_final_sha=spec.get('prereg_final_sha256') == h(os.path.join(WD, 'PREREG_final.txt')))
print('(i)', OUT['i_hash'], flush=True)
# ---------- (iii) 잠금 레이블 미열람: 코드 검색 + 산출물 컬럼
scripts = [os.path.join(WD, 'scripts', f) for f in os.listdir(os.path.join(WD, 'scripts')) if f.endswith('.py')]
hits = []
for s in scripts:
    t = open(s, encoding='utf-8').read()
    if re.search(r"2025\d{4}|'202506|20250630|add_months\(REF\[LOCK\]", t) and 's6_' not in os.path.basename(s):
        hits.append(os.path.basename(s))
fz = pd.read_parquet(os.path.join(WD, 'frozen_scores.parquet'))
un = pd.read_parquet(os.path.join(CA_, 'units.parquet'), columns=['key', 's0', 'y'])
OUT['iii_label'] = dict(scripts_with_2025_dates_excl_s6=hits, frozen_cols=list(fz.columns), lock_y_values=sorted(un[un.s0 == '202412'].y.unique().tolist()),
                        s6_exists=os.path.exists(os.path.join(WD, 'scripts', 's6_lock_score.py')),
                        final_result_exists=os.path.exists(os.path.join(WD, 'final_result_C05T4.json')),
                        lock_labels_exists=os.path.exists(os.path.join(CA_, 'lock_labels.parquet')),
                        cache_files_with_lock=[f for f in os.listdir(CA_) if 'lock' in f.lower()])
print('(iii)', OUT['iii_label'], flush=True)
# ---------- 재구현 로더
SUFX = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def nz(s):
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", SUFX.sub("", str(s or "")).upper())
rows = []
with open(os.path.join(ROOT, '공모전데이터', '외부데이터', 'Corporate_20260720', 'CORP_APPLICANT.txt'), encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\r\n').split('¶')
        if len(p) >= 5:
            rows.append((p[1].strip(), re.sub(r'\D', '', p[4].strip())))
ca = pd.DataFrame(rows, columns=['nm', 'bz']); ca['n'] = ca.nm.map(nz)
ca['okb'] = ca.bz.str.len() == 10; ca['b6'] = ca.bz.str[:6].where(ca.okb); ca['tc'] = ca.okb & ca.bz.str[3:5].isin([str(i) for i in range(81, 89)])
ca2 = ca[ca.n.str.len() >= 2]
g = ca2.groupby('n').agg(N=('bz', 'size'), NB=('okb', 'sum'), U6=('b6', 'nunique'), B6=('b6', 'first'), TC=('tc', 'max'))
n2b = g[(g.N == g.NB) & (g.U6 == 1) & g.TC].B6.to_dict()
kp = pd.read_parquet(os.path.join(ROOT, '공모전데이터', '작업데이터', 'kr_patents.parquet'), columns=['출원번호', '출원연도', '공개일자', '등록일자', '출원인명', '세세분류코드', '우선권국가'])
pub = kp['공개일자'].fillna('').astype(str).str.strip(); rg = kp['등록일자'].fillna('').astype(str).str.strip()
pub = pub.where(pub.str.len() == 8, '99999999'); rg = rg.where(rg.str.len() == 8, '99999999')
kp['v'] = np.minimum(pub.values.astype(np.int64), rg.values.astype(np.int64))
kp['pubi'] = pub.astype(np.int64)
kp['n'] = kp['출원인명'].map(nz); kp['b6'] = kp.n.map(n2b)
fv = kp[kp.b6.notna() & (kp.v < 99999999)].groupby('n').v.min()
REF_L = 20241231
def nps(s):
    assert s not in ('202512', '202608')
    d = pd.read_parquet(os.path.join(EXTD, f'firm_nps_wkpl_{s}.parquet'), columns=['name', 'biz6', 'stat', 'form', 'ldong', 'indcd', 'adpt_dt', 'n_emp', 'n_new', 'n_loss'])
    d = d[(d.stat.astype(str) == '1') & (d.form.astype(str) == '1') & d.biz6.astype(str).str[3:5].isin([str(i) for i in range(81, 89)])].copy()
    d['k'] = d.biz6 + '|' + d.name.map(nz)
    return d
d0 = nps('202412')
E = d0.groupby('k').n_emp.sum()
fE = E[E >= 5]
vis_n = fv[fv <= REF_L].index
vk = set(pd.Series(vis_n).map(n2b) + '|' + pd.Series(vis_n))
nt = pd.read_parquet(os.path.join(EXTD, 'ntis_project_raw.parquet'), columns=['year', 'corp_rgst_no', 'research_agency'])
nt['bz'] = nt.corp_rgst_no.fillna('').str.replace(r'\D', '', regex=True); nt = nt[nt.bz.str.len() == 10]
nt['year'] = pd.to_numeric(nt.year, errors='coerce'); ntw = nt[nt.year.between(2019, 2021)]
blk_i = set((ntw.bz.str[:6] + '|' + ntw.research_agency.map(nz))[ntw.bz.str[3:5].isin([str(i) for i in range(81, 89)])])
cab = ca[ca.okb]; cab = cab.assign(k=cab.b6 + '|' + cab.n)
blk_ii = set(cab[cab.bz.isin(set(ntw.bz))].k)
risk = set(fE.index) - vk - blk_i - blk_ii
fzk = set(fz.key)
OUT['iv_riskset'] = dict(n_indep=len(risk), n_frozen=len(fzk), n_frozen_rows=int(len(fz)), equal=risk == fzk, only_indep=len(risk - fzk), only_frozen=len(fzk - risk),
                         E0ge5=int(len(fE)), linked=int(len(set(fE.index) & vk)))
print('(iv)', OUT['iv_riskset'], flush=True)
# ---------- (ii) 잠금 피처 재계산 비교
u = pd.read_parquet(os.path.join(CA_, 'units.parquet')); u = u[u.s0 == '202412'].set_index('key')
d0s = d0.sort_values('n_emp', ascending=False)
gg = d0s.groupby('k')
re_ = pd.DataFrame({'E': gg.n_emp.sum(), 'nw': gg.size(), 'nn': gg.n_new.sum(), 'nl': gg.n_loss.sum(),
                    'a0': pd.to_numeric(d0s.adpt_dt.astype(str).str[:4], errors='coerce').groupby(d0s.k).min(),
                    'ic': gg.indcd.first(), 'ld': gg.ldong.first()}).reindex(u.index)
dp = nps('202312'); Ep = dp.groupby('k').n_emp.sum()
cmp = {}
cmp['lE0'] = float(np.nanmax(np.abs(np.log1p(re_.E) - np.log1p(u.E))))
cmp['n_wkpl'] = float(np.nanmax(np.abs(re_.nw - u.n_wkpl)))
cmp['r_new'] = float(np.nanmax(np.abs(re_.nn - u.n_new)))
cmp['r_loss'] = float(np.nanmax(np.abs(re_.nl - u.n_loss)))
cmp['age'] = float(np.nanmax(np.abs((2024 - re_.a0).clip(0, 80) - u.age)))
ep = u.index.map(Ep)
cmp['demp'] = float(np.nanmax(np.abs(np.log((re_.E + 1) / (pd.Series(ep, index=u.index).fillna(0) + 1)) - u.demp)))
cmp['newkey'] = float(np.nanmax(np.abs(pd.Series(ep, index=u.index).isna().astype(float) - u.newkey)))
# 침투율(독립): 같은 업종 6자리 E0≥5 법인 중 연결 비율, 2자리 쪽 m=50 평활
ff = pd.DataFrame({'E': gg.n_emp.sum(), 'ic': gg.indcd.first(), 'ld': gg.ldong.first()}); ff = ff[ff.E >= 5]
ic = ff.ic.fillna('').astype(str).str.strip()
ff['i6'] = np.where(ic.str.fullmatch(r'\d{1,6}') & ~ic.isin(['', '000000']), ic.str.zfill(6), 'NA')
ff['i2'] = np.where(ff.i6 == 'NA', 'NA', ff.i6.str[:2])
ld = ff.ld.fillna('').astype(str).str.strip(); ff['sg'] = np.where(ld.str.len() >= 5, ld.str[:5], 'NA'); ff['sd'] = np.where(ld.str.len() >= 5, ld.str[:2], 'NA')
ff['L'] = ff.index.isin(vk).astype(int)
fvk = pd.Series(ff.index.str.split('|').str[1], index=ff.index).map(fv)
ff['F'] = ((ff.L == 1) & (fvk > 20231231) & (fvk <= 20241231)).astype(int)
def smo(a, p, c, m=50):
    st = ff.groupby(a)[c].agg(['sum', 'size']); pr = ff.groupby(p)[c].mean(); par = ff.groupby(a)[p].first().map(pr)
    return ff[a].map((st['sum'] + m * par) / (st['size'] + m))
cmp['pen_ind6'] = float(np.nanmax(np.abs(np.log(smo('i6', 'i2', 'L') + 1e-6).reindex(u.index) - u.pen_ind6)))
cmp['pen_sgg'] = float(np.nanmax(np.abs(np.log(smo('sg', 'sd', 'L') + 1e-6).reindex(u.index) - u.pen_sgg)))
cmp['flow_ind6'] = float(np.nanmax(np.abs(np.log(smo('i6', 'i2', 'F') + 1e-6).reindex(u.index) - u.flow_ind6)))
# KR 성장 제공 라벨(다리 재계산 검사는 g_KR 표만 독립으로; 다리 분포 전체는 s2 코드 검토로 대신)
kv = kp[kp.v <= REF_L]
OUT['ii_docs_max_vis_used_le_ref'] = int(kv.v.max())
OUT['ii_feature_maxabs_diff'] = cmp
print('(ii) feature diffs', cmp, flush=True)
# 로드 기록
ll = open(os.path.join(CA_, 'nps_load_log.txt'), encoding='utf-8').read().split('\n')
snaps = sorted(set(x.split('\t')[1] for x in ll if '\t' in x))
OUT['ii_nps_snapshots_loaded'] = snaps
OUT['ii_locked_snapshot_loaded'] = any(s in ('202512', '202608') for s in snaps)
print('(ii) loaded', snaps, flush=True)
json.dump(OUT, open(os.path.join(WD, 'verify_0', 'v1_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print('saved')
