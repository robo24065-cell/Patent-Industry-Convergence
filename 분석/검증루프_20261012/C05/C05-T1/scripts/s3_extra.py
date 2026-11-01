# C05-T1 s3 (보고용 보충, 판정 기준 불변): (a) 기존 원고 값 '신설 IPC 셀 제외 0.406'(VERIFY_R5) 재현, (b) 신설 하위클래스만 제외,
# (c) H10 셀 CN 모멘텀(부분 소급의 최근 편중) 점검, (d) 원점 2024 감시 목록의 H01L/H10 셀, (e) KR 과거·결과창 소급률, (f) 상위 100 개정 셀 목록.
import os, sys, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
W = os.path.join(ROOT, r"검증루프_20261012\C05\C05-T1"); C = os.path.join(W, 'cache')
R3 = os.path.join(ROOT, r"모델경쟁_20260927\T3_기술셀부상예측\R3")
sys.path.insert(0, os.path.join(W, 'scripts'))
rc = pd.read_csv(os.path.join(W, 'revision_cells.csv'), dtype={'ksic': str})
P = pd.read_parquet(os.path.join(R3, 'preds', 'FINAL_R3_t0_2021.parquet')); P['p_base'] = P.p_base.astype(float)
sk = lambda d: (lambda a, b: (a - b) / (1 - b))(roc_auc_score(d.yb, d.p), roc_auc_score(d.yb, d.p_base))
SK0 = sk(P); out = dict(skill0=SK0)
NEWSC = {'G06V', 'B64U', 'H10B', 'H10K', 'H10N', 'H10D', 'H10F', 'H10H'}; NEW_MG = {'H04N 23', 'H04N 25', 'G06F 18'}
ex82 = P.mg.str[:4].isin(NEWSC) | P.mg.isin(NEW_MG)
out['a_R5_newcode'] = dict(n_excl=int(ex82.sum()), posrate_excl=float(P[ex82].yb.mean()), posrate_rest=float(P[~ex82].yb.mean()),
                           skill=sk(P[~ex82]), d=sk(P[~ex82]) - SK0)
ex72 = P.mg.str[:4].isin(NEWSC)
out['b_newsub_only'] = dict(n_excl=int(ex72.sum()), skill=sk(P[~ex72]), d=sk(P[~ex72]) - SK0)
# 같은 대응표 기준으로 2022–25 신설 메인그룹(기존 하위클래스 안)도 확인
nw = pd.read_csv(os.path.join(W, 'wipo', 'wipo_title_diff.csv'), dtype=str)
newmg = set(nw[(nw.kind == 'new_maingroup') & nw.version.isin(['20220101', '20230101', '20240101', '20250101'])].subclass)
newmg_old_sub = {m for m in newmg if m[:4] not in NEWSC}
exmg = P.mg.isin(newmg_old_sub)
out['b2_new_mg_in_old_subclass'] = dict(n_newmg=len(newmg_old_sub), cells=int(exmg.sum()), pos=int(P[exmg].yb.sum()),
                                        cells_list=P[exmg].mg.value_counts().head(15).to_dict(), R5_list_in_conc=sorted(NEW_MG & newmg))
ex_all = ex72 | exmg
out['b3_newsub_or_newmg'] = dict(n_excl=int(ex_all.sum()), posrate=float(P[ex_all].yb.mean()), skill=sk(P[~ex_all]), d=sk(P[~ex_all]) - SK0)
# (c) CN 모멘텀(공개 2016–18 → 2019–21): A1 대 프로필 매칭 칸(비A2)
pan = pd.read_parquet(os.path.join(C, 'panel.parquet')); pan['cell'] = pan.ksic + '|' + pan.mg
def win(cc, a, b): return pan[(pan.cc == cc) & pan.year.between(a, b)].groupby('cell').n.sum()
for cc in ['CN', 'KR', 'EP']:
    P[f'{cc}_mom_p'] = np.log((P.cell.map(win(cc, 2019, 2021)).fillna(0) + 1) / (P.cell.map(win(cc, 2016, 2018)).fillna(0) + 1))
P['A1'] = P.mg.str.startswith('H10'); P['A2'] = P.mg.str[:4].isin(NEWSC | {'H01L', 'G06K'})
o = {}
for cc in ['CN', 'KR', 'EP']:
    o[cc] = dict(A1_median=float(P[P.A1][f'{cc}_mom_p'].median()), H01L_median=float(P[P.mg.str.startswith('H01L')][f'{cc}_mom_p'].median()),
                 other_median=float(P[~P.A2][f'{cc}_mom_p'].median()))
out['c_mom_2016_18_to_2019_21'] = o
ob = pd.read_csv(os.path.join(W, 'detect_obsolete_symbol_share.csv'), dtype={'version': str})
def wshare(v, cc, a, b):
    x = ob[(ob.version == v) & (ob.cc == cc) & ob.year.between(a, b)]
    return float(x.n_old_sym.sum() / (x.n_old_sym.sum() + x.n_new_sym.sum())) if len(x) else None
out['e_old_symbol_share'] = {v: {cc: dict(dev_2013_18=wshare(v, cc, 2013, 2018), past_2019_21=wshare(v, cc, 2019, 2021), fut_2022_24=wshare(v, cc, 2022, 2024))
                                 for cc in ['KR', 'CN', 'US', 'JP', 'EP']} for v in ['20220101', '20230101', '20250101']}
# (d) 원점 2024 감시 목록
ff = pd.read_csv(os.path.join(R3, 'future_forecast_r3.csv'), dtype={'ksic': str})
fl = ff[ff.pred_breakout_top10 == 1]
sc = fl.mg.str[:4]
out['d_future_list'] = dict(n=int(len(fl)), H01L=int((sc == 'H01L').sum()), H10=int(sc.str.startswith('H10').sum()),
                            H10_by_sub=sc[sc.str.startswith('H10')].value_counts().to_dict(), G06V=int((sc == 'G06V').sum()), B64U=int((sc == 'B64U').sum()),
                            G06K=int((sc == 'G06K').sum()), newsub_2022_25=int(sc.isin(NEWSC).sum()),
                            ksic_with_H=fl[sc.isin(NEWSC | {'H01L'})].ksic.value_counts().to_dict())
# (f) 상위 100·상위 10% 개정 셀
t = rc[(rc.top100) & (rc.A1 | rc.new_sub)][['cell', 'kr_W', 'kr_fut', 'tot_W', 'yb', 'rank_p']]
out['f_top100_revision_cells'] = t.to_dict('records')
json.dump(out, open(os.path.join(W, 's3_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
