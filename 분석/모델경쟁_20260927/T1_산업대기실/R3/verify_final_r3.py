# T1 R3 최종시험 독립 재계산 — final_run_r3.py·r3.py·r2.py의 레이블/위험집합/지표 함수(reg_lab·pub_lab·prior_app·mets·sklearn AUC)를 쓰지 않고
# 원천 캐시(app_pub.parquet, ins_firm.parquet)에서 레이블(2020–2023 진입)과 선행출원 제외를 다시 만들고, 순위합(Mann–Whitney) AUC로 final_result_r3.json과 대조한다.
# 비전문가용 지표(상위 K 적중률·포착률, 산업별 상위 20 적중률)도 함께 낸다(모형·판정 변경 없음).
# 실행: python R3/verify_final_r3.py   (최종시험 채점 후에만)
import json, pathlib, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
R3 = pathlib.Path(__file__).resolve().parent; T1 = R3.parent; CACHE = T1/'cache'
EXTD = T1.parent/'_shared'/'ext'
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
C, H, END = 2019, 4, '20250822'

res = json.load(open(R3/'final_result_r3.json', encoding='utf-8'))
o = pd.read_parquet(R3/'runs'/'FROZEN_final'/f'o{C}.parquet', columns=['firm', 'k', 'newc', 'px', 'pl'])
K = sorted(pd.read_parquet(CACHE/'ins_pat.parquet', columns=['ksic']).ksic.unique()); kidx = {k: i for i, k in enumerate(K)}

a = pd.read_parquet(CACHE/'app_pub.parquet', columns=['applicant_cd', 'ay', 'pub_date', 'rgst_date', 'ksic'])
a = a[a.ay >= 2000].copy(); a['k'] = a.ksic.astype(str).map(kidx); a = a.dropna(subset=['k']); a['k'] = a.k.astype(int)
pdt = a.pub_date.fillna('').astype(str); rdt = a.rgst_date.fillna('').astype(str)
vis_end = ((pdt.str.len() == 8) & (pdt <= END)) | ((rdt.str.len() == 8) & (rdt <= END))
okey = pd.MultiIndex.from_arrays([o.firm.values, o.k.values])
prior = okey.isin(pd.MultiIndex.from_frame(a[a.ay <= C][['applicant_cd', 'k']].drop_duplicates()))
pubset = pd.MultiIndex.from_frame(a[vis_end & (a.ay > C) & (a.ay <= C + H)][['applicant_cd', 'k']].drop_duplicates())
ins = pd.read_parquet(CACHE/'ins_firm.parquet', columns=['applicant_cd', 'ksic', 'ay'])
ins = ins[(ins.ay > C) & (ins.ay <= C + H)].copy(); ins['k'] = ins.ksic.map(kidx)
regset = pd.MultiIndex.from_frame(ins[['applicant_cd', 'k']].dropna().astype({'k': int}).drop_duplicates())
y_reg = okey.isin(regset).astype(np.int8); y_pub = okey.isin(pubset).astype(np.int8); y = np.maximum(y_reg, y_pub)

keep = ~prior; y = y[keep]; yr = y_reg[keep]; nc = o.newc.values[keep].astype(bool); firm = o.firm.values[keep]
def auc(yy, ss):
    r = rankdata(ss); n1 = yy.sum(); n0 = len(yy) - n1
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
n = keep.sum(); s = 0.9 * rankdata(o.px.values[keep]) / n + 0.1 * rankdata(o.pl.values[keep]) / n
bs = pd.read_parquet(R3/'cache_final'/f'o{C}'/'baseline_scores.parquet').loc[keep].reset_index(drop=True)
mine = dict(rows=int(n), pos=int(y.sum()), pos_reg=int(yr.sum()), new_rows=int(nc.sum()), new_pos=int(y[nc].sum()), removed_prior_pairs=int(prior.sum()),
            auc_all=auc(y, s), auc_new=auc(y[nc], s[nc]), auc_all_reglabel=auc(yr, s))
bm = {kk: dict(auc_all=auc(y, bs[kk].values), auc_new=auc(y[nc], bs[kk].values[nc])) for kk in bs.columns}
st_all = max(v['auc_all'] for v in bm.values()); st_new = max(v['auc_new'] for v in bm.values())
mine.update(strongest_auc_all=st_all, strongest_auc_new=st_new, skill_all=(mine['auc_all'] - st_all) / (1 - st_all), skill_new=(mine['auc_new'] - st_new) / (1 - st_new))
# 국세청 폐업 정제 변형(독립: 폐업일 ≤ 2019-12-31)
st = pd.read_parquet(EXTD/'r3nts_applicant_status.parquet', columns=['applicantCd', 'end_dt']).dropna(subset=['end_dt'])
closed = set(st[st.end_dt <= pd.Timestamp(f'{C}-12-31')].applicantCd.astype(str)); cm = pd.Series(firm).astype(str).isin(closed).values
mine['nts_closed_pairs'] = int(cm.sum()); mine['skill_nts'] = (auc(y, np.where(cm, -1.0, s)) - st_all) / (1 - st_all)
ref = dict(rows=res['n']['rows'], pos=res['n']['pos'], pos_reg=res['n']['pos_reg'], new_rows=res['n']['new_rows'], new_pos=res['n']['new_pos'],
           removed_prior_pairs=res['n']['removed_prior_pairs'], auc_all=res['model']['auc_all'], auc_new=res['model']['auc_new'],
           auc_all_reglabel=res['model']['auc_all_reglabel'], strongest_auc_all=res['baseline_strongest']['auc_all'],
           strongest_auc_new=res['baseline_strongest']['auc_new'], skill_all=res['skill']['auc_all'], skill_new=res['skill']['auc_new'],
           nts_closed_pairs=res['n']['nts_closed_pairs'], skill_nts=res['secondary_preregistered']['model_nts']['skill_auc_all'])
CNT = ['rows', 'pos', 'pos_reg', 'new_rows', 'new_pos', 'removed_prior_pairs', 'nts_closed_pairs']
rows = [dict(item=kk, final_run=ref[kk], independent=mine[kk], abs_diff=abs(float(ref[kk]) - float(mine[kk]))) for kk in ref]
df = pd.DataFrame(rows); print(df.to_string(index=False))
for kk, v in bm.items(): print('baseline', kk, round(v['auc_all'], 6), round(res['baselines'][kk]['auc_all'], 6))


# ---- 비전문가용 지표 ----
def topk(score, yy, frac=None, k=None):
    k = k or int(round(len(yy) * frac)); idx = np.argsort(-score, kind='stable')[:k]
    return dict(k=int(k), hits=int(yy[idx].sum()), precision=float(yy[idx].mean()), recall=float(yy[idx].sum() / yy.sum()))
best = max(bm, key=lambda q: bm[q]['auc_all'])
plain = dict(base_rate=float(y.mean()), n=int(n), pos=int(y.sum()), strongest_baseline=best)
for nm, sc in (('model', s), (f'baseline_{best}', bs[best].values)):
    plain[nm] = {'top100': topk(sc, y, k=100), 'top1000': topk(sc, y, k=1000), 'top1pct': topk(sc, y, 0.01), 'top5pct': topk(sc, y, 0.05), 'top10pct': topk(sc, y, 0.10)}
d = pd.DataFrame({'k': o.k.values[keep], 's': s, 'b': bs[best].values, 'y': y})
plain['per_ksic_top20_precision_model'] = float(d.sort_values('s', ascending=False).groupby('k').head(20).y.mean())
plain['per_ksic_top20_precision_baseline'] = float(d.sort_values('b', ascending=False).groupby('k').head(20).y.mean())
plain['lift_top1000_vs_random'] = plain['model']['top1000']['precision'] / plain['base_rate']

out = dict(compare=rows, baselines_independent=bm,
           max_abs_diff_metrics=float(df[~df.item.isin(CNT)].abs_diff.max()), counts_equal=bool((df[df.item.isin(CNT)].abs_diff == 0).all()), plain=plain)
json.dump(out, open(R3/'final_verify_r3.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('max_abs_diff_metrics', out['max_abs_diff_metrics'], 'counts_equal', out['counts_equal'])
print(json.dumps(plain, indent=1, ensure_ascii=False))
