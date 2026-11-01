# C03-T1 s6 (사후 보조, 판정 기준 변경 없음): 이름 매칭 '의심' 양성의 세부 분해(짧은 이름 vs 동명 다른 코드)와
#            최악 가정(의심 양성이 모두 거짓) 재채점 민감도. 동결 예측만 재채점.
import sys, re, json, time, sqlite3, pathlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()
ROOT = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전")
T1 = ROOT/'분석/모델경쟁_20260927/T1_산업대기실'; TC = T1/'cache'
W = ROOT/'분석/검증루프_20261012/C03/C03-T1'; WC = W/'cache'
def ro(p): return sqlite3.connect(pathlib.Path(p).as_uri() + '?mode=ro', uri=True)
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda x: STRIP.sub('', str(x)).lower()
BCOLS = ['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u']
ev = pd.read_parquet(WC/'eval_o2019.parquet'); ev['k'] = ev.k.astype(int)
ap = pd.read_parquet(WC/'app_pub_src_2020_23.parquet'); nm = pd.read_parquet(WC/'kr_names_2020_23.parquet'); al = pd.read_parquet(WC/'alias_src.parquet')
reg = pd.read_sql('select distinct applicantCd, applicantName from applicant', ro(ROOT/'수집파이프라인/registry_full.sqlite'))
aq = pd.read_sql('select applicant_cd, applicant_name from applicant_query', ro(ROOT/'수집파이프라인/business_rights.sqlite'))
allnm = pd.concat([reg.rename(columns={'applicantCd': 'cd', 'applicantName': 'nm'}), aq.rename(columns={'applicant_cd': 'cd', 'applicant_name': 'nm'})]); del reg
allnm['n'] = allnm.nm.map(norm); ncnt = allnm.drop_duplicates(['n', 'cd']).groupby('n').cd.nunique()
pos = ev[(ev.yA_pub == 1) & (ev.yA_reg == 0)][['firm', 'k']]
m = ap.merge(pos, left_on=['applicant_cd', 'k'], right_on=['firm', 'k'])[['firm', 'k', 'appl_no']]
h = m.merge(nm, on='appl_no').merge(al[['applicant_cd', 'n']].rename(columns={'applicant_cd': 'firm', 'n': 'nn'}), on=['firm', 'nn']).drop_duplicates(['firm', 'k', 'appl_no', 'nn'])
h['short'] = h.nn.str.len() <= 3; h['homonym'] = h.nn.map(ncnt).fillna(1) > 1
rl = pd.read_parquet(TC/'rl_pat.parquet', columns=['applicant_cd', 'appl_no']).drop_duplicates()
rlset = set(zip(rl.applicant_cd, rl.appl_no)); h['rl'] = [(f_, a_) in rlset for f_, a_ in zip(h.firm, h.appl_no)]
g = h.groupby(['firm', 'k']).agg(rl=('rl', 'any'), all_short=('short', 'all'), any_homonym=('homonym', 'any')).reset_index()
g['cls'] = np.where(g.rl, 'rl확인', np.where(g.any_homonym, '동명다른코드', np.where(g.all_short, '짧은이름만', '이름규칙정확')))
print(g.cls.value_counts().to_dict(), len(pos), round(time.time() - t0), flush=True)
nc = ev.newc.values.astype(bool); s = ev.s.values; y0 = np.maximum(ev.yA_reg.values, ev.yA_pub.values)
def sc(y):
    a = roc_auc_score(y, s); an = roc_auc_score(y[nc], s[nc])
    b = max(roc_auc_score(y, ev['b_' + c].values) for c in BCOLS); bn = max(roc_auc_score(y[nc], ev['b_' + c].values[nc]) for c in BCOLS)
    return dict(pos=int(y.sum()), auc_all=a, auc_new=an, strongest_all=b, strongest_new=bn, skill_all=(a - b) / (1 - b), skill_new=(an - bn) / (1 - bn))
base = sc(y0); out = dict(classes=g.cls.value_counts().to_dict(), pub_only_pairs=int(len(pos)), official=base)
key = ev.set_index(['firm', 'k']).index
for nm_, cl in (('drop_suspicious_all', ['동명다른코드', '짧은이름만']), ('drop_homonym_only', ['동명다른코드'])):
    bad = set(map(tuple, g[g.cls.isin(cl)][['firm', 'k']].values)); mask = np.fromiter(((f_, k_) in bad for f_, k_ in zip(ev.firm.values, ev.k.values)), bool, len(ev))
    y = y0.copy(); y[mask] = 0; r = sc(y); r['removed'] = int(mask.sum()); r['pos_change_rate'] = (r['pos'] - base['pos']) / base['pos']
    r['d_auc_all'] = r['auc_all'] - base['auc_all']; r['d_skill_all'] = r['skill_all'] - base['skill_all']; r['d_auc_new'] = r['auc_new'] - base['auc_new']; r['d_skill_new'] = r['skill_new'] - base['skill_new']
    out[nm_] = r
json.dump(out, open(WC/'s6_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float)); print('done', round(time.time() - t0))
