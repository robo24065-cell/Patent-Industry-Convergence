# T1 R2 최종시험 독립 재계산 — final_run_r2.py의 함수(pub_lab·prior_app·mets·sklearn AUC)를 쓰지 않고
# 원천 캐시(app_pub.parquet, ins_firm.parquet)에서 레이블·위험집합 제외를 다시 만들고, 순위합(Mann–Whitney) AUC를 직접 계산해 final_result.json과 대조한다.
# 실행: python R2/verify_final_r2.py   (최종시험 실행 후에만)
import json, pathlib, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
R2 = pathlib.Path(__file__).resolve().parent; T1 = R2.parent; CACHE = T1/'cache'
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
C, H, END = 2017, 5, '20250822'

res = json.load(open(R2/'final_result.json', encoding='utf-8'))
o = pd.read_parquet(R2/'runs'/'FROZEN_final'/f'o{C}.parquet', columns=['firm', 'k', 'newc', 'px', 'pl'])
K = sorted(pd.read_parquet(CACHE/'ins_pat.parquet', columns=['ksic']).ksic.unique()); kidx = {k: i for i, k in enumerate(K)}

# (1) 선행 공개·등록 출원 쌍(위험집합 추가 제외): 기준시점 이하 출원, 데이터 끝까지 공개 또는 등록된 것
a = pd.read_parquet(CACHE/'app_pub.parquet', columns=['applicant_cd', 'ay', 'pub_date', 'rgst_date', 'ksic'])
a = a[a.ay >= 2000].copy(); a['k'] = a.ksic.astype(str).map(kidx); a = a.dropna(subset=['k']); a['k'] = a.k.astype(int)
pdt = a.pub_date.fillna('').astype(str); rdt = a.rgst_date.fillna('').astype(str)
vis_end = ((pdt.str.len() == 8) & (pdt <= END)) | ((rdt.str.len() == 8) & (rdt <= END))
key = lambda f, k: pd.MultiIndex.from_arrays([f, k])
okey = key(o.firm.values, o.k.values)
prior_set = pd.MultiIndex.from_frame(a[(a.ay <= C)][['applicant_cd', 'k']].drop_duplicates())   # prior_app(te, None): 가시 필터 없음
prior = okey.isin(prior_set)

# (2) 진입 레이블(공개 기준 합집합): (2017, 2022] 출원 — 공개(별칭 매칭)는 데이터 끝까지 가시분, 등록은 원출원인 코드
pubset = pd.MultiIndex.from_frame(a[vis_end & (a.ay > C) & (a.ay <= C + H)][['applicant_cd', 'k']].drop_duplicates())
ins = pd.read_parquet(CACHE/'ins_firm.parquet', columns=['applicant_cd', 'ksic', 'ay'])
ins = ins[(ins.ay > C) & (ins.ay <= C + H)].copy(); ins['k'] = ins.ksic.map(kidx)
regset = pd.MultiIndex.from_frame(ins[['applicant_cd', 'k']].dropna().astype({'k': int}).drop_duplicates())
y_reg = okey.isin(regset).astype(np.int8); y_pub = okey.isin(pubset).astype(np.int8); y = np.maximum(y_reg, y_pub)

keep = ~prior; y = y[keep]; yr = y_reg[keep]; nc = o.newc.values[keep].astype(bool)
def auc(yy, ss):
    r = rankdata(ss); n1 = yy.sum(); n0 = len(yy) - n1
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
s = 0.9 * rankdata(o.px.values[keep]) / keep.sum() + 0.1 * rankdata(o.pl.values[keep]) / keep.sum()
bs = pd.read_parquet(R2/'cache_final'/f'o{C}'/'baseline_scores.parquet').loc[keep].reset_index(drop=True)
mine = dict(rows=int(keep.sum()), pos=int(y.sum()), pos_reg=int(yr.sum()), new_rows=int(nc.sum()), new_pos=int(y[nc].sum()), removed_prior_pairs=int(prior.sum()),
            auc_all=auc(y, s), auc_new=auc(y[nc], s[nc]), auc_all_reglabel=auc(yr, s))
bm = {kk: dict(auc_all=auc(y, bs[kk].values), auc_new=auc(y[nc], bs[kk].values[nc])) for kk in bs.columns}
st_all = max(v['auc_all'] for v in bm.values()); st_new = max(v['auc_new'] for v in bm.values())
mine.update(strongest_auc_all=st_all, strongest_auc_new=st_new, skill_all=(mine['auc_all'] - st_all) / (1 - st_all), skill_new=(mine['auc_new'] - st_new) / (1 - st_new))
ref = dict(rows=res['n']['rows'], pos=res['n']['pos'], pos_reg=res['n']['pos_reg'], new_rows=res['n']['new_rows'], new_pos=res['n']['new_pos'],
           removed_prior_pairs=res['n']['removed_prior_pairs'], auc_all=res['model']['auc_all'], auc_new=res['model']['auc_new'],
           auc_all_reglabel=res['model']['auc_all_reglabel'], strongest_auc_all=res['baseline_strongest']['auc_all'],
           strongest_auc_new=res['baseline_strongest']['auc_new'], skill_all=res['skill']['auc_all'], skill_new=res['skill']['auc_new'])
rows = [dict(item=kk, final_run=ref[kk], independent=mine[kk], abs_diff=abs(float(ref[kk]) - float(mine[kk]))) for kk in ref]
df = pd.DataFrame(rows); print(df.to_string(index=False))
for kk, v in bm.items(): print('baseline', kk, round(v['auc_all'], 5), round(res['baselines'][kk]['auc_all'], 5))
out = dict(compare=rows, baselines_independent=bm, max_abs_diff_metrics=float(df[~df.item.isin(['rows', 'pos', 'pos_reg', 'new_rows', 'new_pos', 'removed_prior_pairs'])].abs_diff.max()),
           counts_equal=bool((df[df.item.isin(['rows', 'pos', 'pos_reg', 'new_rows', 'new_pos', 'removed_prior_pairs'])].abs_diff == 0).all()))
json.dump(out, open(R2/'final_verify.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('max_abs_diff_metrics', out['max_abs_diff_metrics'], 'counts_equal', out['counts_equal'])
