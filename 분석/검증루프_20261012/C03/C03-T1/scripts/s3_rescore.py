# C03-T1 s3: 복원판 레이블로 동결 예측 재채점(점추정) — 최종(원점 2019)과 개발 원점(2008–12, 보조 2015) (P2·P4)
import sys, json, time, pathlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()
ROOT = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전")
T1 = ROOT/'분석/모델경쟁_20260927/T1_산업대기실'; R3 = T1/'R3'; TC = T1/'cache'
W = ROOT/'분석/검증루프_20261012/C03/C03-T1'; WC = W/'cache'
BCOLS = ['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u']
skill = lambda m_, b_: (m_ - b_) / (1 - b_)
rank = lambda s: pd.Series(s).rank(pct=True).values
bp = pd.read_parquet(WC/'blank_pairs_2009_2024.parquet', columns=['appl_no', 'applicant_cd', 'ay', 'k', 'self', 'pubmatch']).dropna(subset=['k'])
bp['k'] = bp.k.astype(int)

def rand_auc(y, s, seed):
    rs = np.random.RandomState(seed); o = np.lexsort((rs.rand(len(s)), s)); r = np.empty(len(s)); r[o] = np.arange(len(s)); return float(roc_auc_score(y, r))
def topk(y, s, kind, seed=None, ks=(100, 1000, 10000)):
    if kind == 'stable': o = np.argsort(-s, kind='stable')
    else: rs = np.random.RandomState(seed); o = np.lexsort((rs.rand(len(s)), -s))
    return {str(k): int(y[o[:k]].sum()) for k in ks}

def score_set(y, yr, s, B, nc, full=True):
    m = dict(auc_all=float(roc_auc_score(y, s)), auc_new=float(roc_auc_score(y[nc], s[nc])),
             pr_all=float(average_precision_score(y, s)), pr_new=float(average_precision_score(y[nc], s[nc])))
    bm = {c: dict(auc_all=float(roc_auc_score(y, B[c])), auc_new=float(roc_auc_score(y[nc], B[c][nc]))) for c in BCOLS}
    st = {q: max(v[q] for v in bm.values()) for q in ('auc_all', 'auc_new')}; arg = max(bm, key=lambda q: bm[q]['auc_all'])
    out = dict(pos=int(y.sum()), pos_reg=int(yr.sum()), new_pos=int(y[nc].sum()), model=m, baseline=bm, strongest=st, strongest_arg=arg,
               skill_all=skill(m['auc_all'], st['auc_all']), skill_new=skill(m['auc_new'], st['auc_new']))
    if full:
        out['auc_all_randtie'] = rand_auc(y, s, 1); out['auc_new_randtie'] = rand_auc(y[nc], s[nc], 2)
        out['topk_model_stable'] = topk(y, s, 'stable'); rr = [topk(y, s, 'random', sd) for sd in range(5)]
        out['topk_model_random_mean'] = {k: float(np.mean([r[k] for r in rr])) for k in rr[0]}; out['topk_model_random_range'] = {k: [min(r[k] for r in rr), max(r[k] for r in rr)] for k in rr[0]}
        out['topk_base_stable'] = topk(y, B[arg], 'stable'); rb = [topk(y, B[arg], 'random', sd) for sd in range(5)]
        out['topk_base_random_mean'] = {k: float(np.mean([r[k] for r in rb])) for k in rb[0]}
        out['topk_model_new_stable'] = topk(y[nc], s[nc], 'stable', ks=(100, 1000))
    return out

def restore(firm, k, yu, lo, hi, which):
    sub = bp[(bp.ay >= lo) & (bp.ay <= hi)]
    if which == 'R_self': sub = sub[sub.self]
    key = set(zip(sub.applicant_cd, sub.k))
    add = np.fromiter(((f_, int(k_)) in key for f_, k_ in zip(firm, k)), bool, len(firm))
    return add

AP = pd.read_parquet(TC/'app_pub.parquet'); Kl = sorted(pd.read_parquet(TC/'ins_pat.parquet', columns=['ksic']).ksic.astype(str).unique()); _ki = {k: i for i, k in enumerate(Kl)}
AP['k'] = AP.ksic.astype(str).map(_ki); AP = AP.dropna(subset=['k']); AP['k'] = AP.k.astype(int); AP = AP[AP.ay >= 2000]
_pd = AP.pub_date.fillna('').astype(str); _rd = AP.rgst_date.fillna('').astype(str); AP = AP[((_pd.str.len() == 8) & (_pd <= '20250822')) | ((_rd.str.len() == 8) & (_rd <= '20250822'))]
def pair_pub(o, c):
    e = AP[(AP.ay > c) & (AP.ay <= c + 4)][['applicant_cd', 'k']].drop_duplicates(); key = set(zip(e.applicant_cd, e.k))
    return np.fromiter(((f_, int(k_)) in key for f_, k_ in zip(o.firm.values, o.k.values)), bool, len(o)).astype(np.int8)
res = {}
# ---------- 최종 ----------
ev = pd.read_parquet(WC/'eval_o2019.parquet')
s = ev.s.values; nc = ev.newc.values.astype(bool); B = {c: ev['b_' + c].values for c in BCOLS}
yr0 = ev.yA_reg.values; yp = ev.yA_pub.values; y0 = np.maximum(yr0, yp)
print('ties in s: unique', len(np.unique(s)), 'of', len(s), flush=True)
F = {'official': score_set(y0, yr0, s, B, nc)}
lab = {}
for which in ('R_self', 'R_any'):
    add = restore(ev.firm.values, ev.k.values, y0, 2020, 2023, which)
    yr = np.maximum(yr0, add.astype(np.int8)); y = np.maximum(yr, yp); lab[which] = y
    r = score_set(y, yr, s, B, nc); r['added_rows_total'] = int(add.sum()); r['new_pos'] = int(((y == 1) & (y0 == 0)).sum()); r['new_pos_newc'] = int(((y == 1) & (y0 == 0) & nc).sum())
    r['pos_change_rate'] = (int(y.sum()) - int(y0.sum())) / int(y0.sum())
    # 새 양성의 동결 점수 위치(백분위)와 공식 양성 대비
    pr_ = rank(s); r['new_pos_score_pct_median'] = float(np.median(pr_[(y == 1) & (y0 == 0)])); r['official_pos_score_pct_median'] = float(np.median(pr_[y0 == 1]))
    F[which] = r
np.save(WC/'lab_R_self.npy', lab['R_self'].astype(np.int8)); np.save(WC/'lab_R_any.npy', lab['R_any'].astype(np.int8))
off = F['official']
for which in ('R_self', 'R_any'):
    r = F[which]
    r['delta'] = dict(auc_all=r['model']['auc_all'] - off['model']['auc_all'], auc_new=r['model']['auc_new'] - off['model']['auc_new'],
                      skill_all=r['skill_all'] - off['skill_all'], skill_new=r['skill_new'] - off['skill_new'],
                      strongest_auc_all=r['strongest']['auc_all'] - off['strongest']['auc_all'], strongest_auc_new=r['strongest']['auc_new'] - off['strongest']['auc_new'],
                      pr_all=r['model']['pr_all'] - off['model']['pr_all'], pr_new=r['model']['pr_new'] - off['model']['pr_new'],
                      top1000_stable=r['topk_model_stable']['1000'] - off['topk_model_stable']['1000'], top10000_stable=r['topk_model_stable']['10000'] - off['topk_model_stable']['10000'])
res['final'] = F
# 공개 경로 재현율(진단): 공식 y_reg=1 쌍 중 y_pub=1 비율
res['final_pub_recall_of_reg'] = float(yp[yr0 == 1].mean()); res['final_pub_only_share'] = float(((yp == 1) & (yr0 == 0)).sum() / y0.sum())
print('final', json.dumps({w: (F[w]['pos'], F[w].get('pos_change_rate'), F[w].get('delta')) for w in F}, ensure_ascii=False), round(time.time() - t0), flush=True)

# ---------- 개발 원점 ----------
D = {}
for c in (2008, 2009, 2010, 2011, 2012, 2015):
    o = pd.read_parquet(R3/f'runs/FROZEN_dev/o{c}.parquet'); bs = pd.read_parquet(R3/f'cache/o{c}/baseline_scores.parquet')
    keep = ~o.prior_any.values; o = o[keep].reset_index(drop=True); bs = bs.loc[keep].reset_index(drop=True)
    sd = 0.9 * rank(o.px.values) + 0.1 * rank(o.pl.values); ncd = o.newc.values.astype(bool); Bd = {cc: bs[cc].values for cc in BCOLS}
    yu = o.y_u.values.astype(np.int8); yrd = o.y_reg.values.astype(np.int8)
    ypd = pair_pub(o, c)
    assert int(np.maximum(yrd, ypd).sum()) == int(yu.sum()), '개발 y_u 재구성 불일치'
    base = score_set(yu, yrd, sd, Bd, ncd, full=False)
    add = restore(o.firm.values, o.k.values, yu, c + 1, c + 4, 'R_self'); y2 = np.maximum(yu, add.astype(np.int8)); yr2 = np.maximum(yrd, add.astype(np.int8))
    rs_ = score_set(y2, yr2, sd, Bd, ncd, full=False)
    blank_in_window = int(bp[(bp.ay > c) & (bp.ay <= c + 4)].appl_no.nunique())
    D[c] = dict(pos=base['pos'], pos_reg=base['pos_reg'], pub_only_share=float(((yu == 1) & (yrd == 0)).sum() / yu.sum()), pub_recall_of_reg=float(ypd[yrd == 1].mean()),
                skill=base['skill_all'], skill_new=base['skill_new'], auc_all=base['model']['auc_all'], strongest=base['strongest']['auc_all'],
                blank_patents_window_rl=blank_in_window, new_pos_R_self=int(((y2 == 1) & (yu == 0)).sum()), pos_change_rate=float((y2.sum() - yu.sum()) / yu.sum()),
                dskill=rs_['skill_all'] - base['skill_all'], dskill_new=rs_['skill_new'] - base['skill_new'], dauc=rs_['model']['auc_all'] - base['model']['auc_all'])
    # 개발 원점 공개경로 재현율: o에는 y_pub 열이 없어 y_u−y_reg로 공개 전용만 계산
    print('dev', c, D[c], round(time.time() - t0), flush=True)
res['dev'] = D
mean_dev = float(np.mean([D[c]['dskill'] for c in (2008, 2009, 2010, 2011, 2012)]))
res['P4'] = dict(dskill_final_R_self=F['R_self']['delta']['skill_all'], mean_dskill_dev=mean_dev, diff=F['R_self']['delta']['skill_all'] - mean_dev,
                 threshold=0.0126, verdict='개발–최종 차이에 기여' if abs(F['R_self']['delta']['skill_all'] - mean_dev) >= 0.0126 else '기여 미미',
                 dev_mean_skill_recomputed=float(np.mean([D[c]['skill'] for c in (2008, 2009, 2010, 2011, 2012)])))
json.dump(res, open(WC/'s3_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
rows = []
for w in ('official', 'R_self', 'R_any'):
    r = F[w]; rows.append(dict(set='final_o2019', label=w, pos=r['pos'], pos_reg=r['pos_reg'], new_pos_sub=r['new_pos'] if w != 'official' else '', pos_change_rate=r.get('pos_change_rate', 0.0),
                               auc_all=r['model']['auc_all'], auc_new=r['model']['auc_new'], auc_all_randtie=r['auc_all_randtie'], pr_all=r['model']['pr_all'],
                               strongest_auc_all=r['strongest']['auc_all'], strongest_arg=r['strongest_arg'], strongest_auc_new=r['strongest']['auc_new'],
                               skill_all=r['skill_all'], skill_new=r['skill_new'],
                               top100_stable=r['topk_model_stable']['100'], top1000_stable=r['topk_model_stable']['1000'], top10000_stable=r['topk_model_stable']['10000'],
                               top1000_random_mean=r['topk_model_random_mean']['1000'], top10000_random_mean=r['topk_model_random_mean']['10000'],
                               base_top1000_stable=r['topk_base_stable']['1000'], newc_pos=r['new_pos'] if w == 'official' else r['new_pos_newc'],
                               newc_top100_stable=r['topk_model_new_stable']['100'], newc_top1000_stable=r['topk_model_new_stable']['1000']))
for c in D:
    rows.append(dict(set=f'dev_o{c}', label='official→R_self', pos=D[c]['pos'], pos_reg=D[c]['pos_reg'], new_pos_sub=D[c]['new_pos_R_self'], pos_change_rate=D[c]['pos_change_rate'],
                     auc_all=D[c]['auc_all'], strongest_auc_all=D[c]['strongest'], skill_all=D[c]['skill'], skill_new=D[c]['skill_new'],
                     d_auc_all=D[c]['dauc'], d_skill_all=D[c]['dskill'], d_skill_new=D[c]['dskill_new']))
pd.DataFrame(rows).to_csv(WC/'rescore_point.csv', index=False, encoding='utf-8-sig')
print(json.dumps(res['P4'], ensure_ascii=False)); print('done', round(time.time() - t0))
