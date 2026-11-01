# -*- coding: utf-8 -*-
"""선택 편향 기술(PREREG §5): N1 원점 가시(주 증거) 대 비가시 신규진입자 법인 비교 + 학습 패널 신규진입자 수 + RC 대 OK 비교.
 → sel_bias.csv, cache/sel_bias.json"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
out = {}; K, ki = kidx()
O = official_frame(); Ok = O[~O.prior_any.values].reset_index(drop=True)
Ok['bl'] = (0.9 * pd.Series(Ok.px.values).rank(pct=True) + 0.1 * pd.Series(Ok.pl.values).rank(pct=True)).values
Ok['pct'] = pd.Series(Ok.bl.values).rank(pct=True).values
E = pd.read_parquet(CC/'evid_all.parquet'); E = E[E.is_newc_off].set_index('firm')
nw = Ok[Ok.newc.values]
f = nw.groupby('firm').agg(rows=('y_off', 'size'), pos=('y_off', 'sum'), pct_max=('pct', 'max'), pct_mean=('pct', 'mean'))
f['entry_any'] = (f.pos > 0).astype(int)
pp = nw[nw.y_off == 1].groupby('firm').pct.mean(); f['pct_pos_mean'] = pp.reindex(f.index)
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'lsize', 'lout']).drop_duplicates('firm').set_index('firm'); f['lsize'] = te.lsize.reindex(f.index).values; f['lout'] = te.lout.reindex(f.index).values
oos = pd.read_parquet(CACHE/'oos.parquet', columns=['applicant_cd', 'rgst_no', 'ay', 'self']); oo = oos[oos.self & (oos.ay >= 2000) & (oos.ay <= CUT) & oos.applicant_cd.isin(set(f.index))].drop_duplicates(['applicant_cd', 'rgst_no'])
g = oo.groupby('applicant_cd').agg(first_ay=('ay', 'min'), last_ay=('ay', 'max'), n_oos=('rgst_no', 'size')); f = f.join(g)
f['foreign'] = (pd.Series(f.index, index=f.index).str[0] == '5').astype(int)
f['vis'] = E.N1_primary.reindex(f.index).astype(int).values
for c in ['E_pub', 'E_own']: f[c] = E[c].reindex(f.index).astype(int).values
app = pd.read_parquet(CACHE/'ins_applicant.parquet', columns=['rgst_no', 'applicantCd']); pat = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['rgst_no', 'appl_date', 'ksic'])
x = app[app.applicantCd.isin(set(f.index))].merge(pat, on='rgst_no').sort_values(['applicantCd', 'appl_date', 'rgst_no'])
first_k = x.drop_duplicates('applicantCd').set_index('applicantCd').ksic; f['first_k'] = first_k.reindex(f.index).values


def smd(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); s = np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2); return float((np.nanmean(a) - np.nanmean(b)) / s) if s > 0 else 0.0


def tvd(a, b):
    pa = pd.Series(a).value_counts(normalize=True); pb = pd.Series(b).value_counts(normalize=True); idx = pa.index.union(pb.index)
    return float(0.5 * np.abs(pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).sum())


V = f[f.vis == 1]; I = f[f.vis == 0]; rows = []
for v in ['entry_any', 'pos', 'rows', 'first_ay', 'last_ay', 'n_oos', 'lsize', 'lout', 'foreign', 'pct_max', 'pct_mean', 'pct_pos_mean']:
    rows.append(dict(var=v, n_vis=len(V), n_invis=len(I), mean_vis=float(V[v].mean()), mean_invis=float(I[v].mean()), median_vis=float(V[v].median()), median_invis=float(I[v].median()), smd=smd(V[v], I[v])))
pr_v = V.pos.sum() / V.rows.sum(); pr_i = I.pos.sum() / I.rows.sum()
rows.append(dict(var='pair_pos_rate', n_vis=int(V.rows.sum()), n_invis=int(I.rows.sum()), mean_vis=float(pr_v), mean_invis=float(pr_i)))
S = pd.DataFrame(rows); S.to_csv(HERE/'sel_bias.csv', index=False, encoding='utf-8-sig')
posk_v = nw[nw.firm.isin(set(V.index)) & (nw.y_off == 1)].k; posk_i = nw[nw.firm.isin(set(I.index)) & (nw.y_off == 1)].k
out['tvd_first_ins_ksic'] = tvd(V.first_k.dropna(), I.first_k.dropna()); out['tvd_pos_pair_k'] = tvd(posk_v, posk_i)
# 참고 기준: 무작위 반분 TVD(같은 크기, 200회) — 표본 크기만으로 생기는 TVD
rng = np.random.RandomState(17); allk = f.first_k.dropna().values; nv = int(V.first_k.notna().sum()); null = []
for _ in range(200):
    p = rng.permutation(len(allk)); null.append(tvd(allk[p[:nv]], allk[p[nv:]]))
out['tvd_first_ins_ksic_null95'] = float(np.percentile(null, 95))
allp = np.concatenate([posk_v.values, posk_i.values]); npv = len(posk_v); null2 = []
for _ in range(200):
    p = rng.permutation(len(allp)); null2.append(tvd(allp[p[:npv]], allp[p[npv:]]))
out['tvd_pos_pair_k_null95'] = float(np.percentile(null2, 95))
out['vis_firms'] = int(len(V)); out['invis_firms'] = int(len(I)); out['vis_pos'] = int(V.pos.sum()); out['invis_pos'] = int(I.pos.sum())
out['pair_pos_rate_vis'] = float(pr_v); out['pair_pos_rate_invis'] = float(pr_i)
out['incumbent_pair_pos_rate'] = float(Ok.y_off.values[~Ok.newc.values].mean()); out['newc_pair_pos_rate'] = float(nw.y_off.mean())
out['entry_by_first_ins_ay'] = {}
fi = pd.read_parquet(CC/'newc_evidence.parquet').set_index('firm').first_ins_ay; f['first_ins_ay'] = fi.reindex(f.index).values
out['entry_by_first_ins_ay'] = f.groupby('first_ins_ay').agg(firms=('pos', 'size'), entry_any=('entry_any', 'mean'), vis=('vis', 'mean')).round(4).reset_index().astype(str).to_dict('records')
# 학습 패널(원점 2012–15)의 신규진입자: 같은 applicant_query 선택을 받는다
tr = pd.read_parquet(FC/'prep_tr.parquet', columns=['firm', 'cut', 'newc', 'y'])
out['train_newc'] = tr.groupby('cut').apply(lambda d: dict(firms=int(d.firm.nunique()), newc_firms=int(d.firm[d.newc].nunique()), newc_rows=int(d.newc.sum()),
                                                            newc_pos_rate=float(d.y[d.newc].mean()), inc_pos_rate=float(d.y[~d.newc].mean()))).to_dict()
out['train_newc'] = {int(k): v for k, v in out['train_newc'].items()}
# 학습 원점 신규진입자 법인이 범위안 등록 원출원인 첫 기록을 원점 이후에 갖는가(= 원점 이후 정보로 선택)
xa = app.merge(pd.read_parquet(CACHE/'ins_pat.parquet', columns=['rgst_no', 'ay', 'rgst_date']), on='rgst_no')
fst = xa.groupby('applicantCd').agg(first_ay=('ay', 'min'), first_rg=('rgst_date', 'min'))
out['train_newc_first_ins'] = {}
for c in sorted(tr.cut.unique()):
    nf_ = tr.firm[(tr.cut == c) & tr.newc].unique(); q = fst.reindex(nf_)
    out['train_newc_first_ins'][int(c)] = dict(firms=int(len(nf_)), first_ay_gt_cut=float((q.first_ay > c).mean()), first_rgst_after_cutyear=float((q.first_rg.astype(str) > f'{c}1231').mean()),
                                              first_rgst_after_20191231=float((q.first_rg.astype(str) > '20191231').mean()))
out['mem_peak'] = G.rss_peak(); jdump(out, CC/'sel_bias.json'); print(S.round(4).to_string()); print(json.dumps({k: v for k, v in out.items() if k != 'entry_by_first_ins_ay'}, ensure_ascii=False, indent=1))
