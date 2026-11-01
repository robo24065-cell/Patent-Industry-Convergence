# -*- coding: utf-8 -*-
"""verify_2 반박 검증(독립): C16-T1 S2 발견 재계산. 작업 스크립트(c16common·s_score·popdef) import 없음.
원천: 동결 o2019.parquet(firm,k,y_u,prior_any,px,pl), baseline_scores, T1 cache ins_firm·oos·app_pub, kr_registered 공개일자.
AUC: scipy rankdata Mann-Whitney. 출력: verify_2/v2_s2_out.json"""
import sys, json, pathlib
import numpy as np, pandas as pd
from scipy.stats import rankdata
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; CA = T1/'cache'
OUT = pathlib.Path(__file__).resolve().parent/'v2_s2_out.json'
BK = ['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u']


def d8(x):
    s = pd.Series(x).fillna('').astype(str).str.replace('-', '', regex=False).str.replace('.', '', regex=False).str.strip().str[:8]
    v = pd.to_numeric(s.where(s.str.len() == 8), errors='coerce')
    return v.fillna(99999999).astype(np.int64).values


def plus18(ad):
    d = pd.to_datetime(pd.Series(ad).astype(str).str.replace('-', '', regex=False).str[:8], format='%Y%m%d', errors='coerce') + pd.DateOffset(months=18)
    return pd.to_numeric(d.dt.strftime('%Y%m%d'), errors='coerce').fillna(99999999).astype(np.int64).values


def mw_auc(s, y):
    y = np.asarray(y).astype(bool); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pct(s): return rankdata(s) / len(s)


out = {}
fz = pd.read_parquet(T1/'R3/runs/FROZEN_final/o2019.parquet', columns=['firm', 'k', 'newc', 'y_u', 'prior_any', 'px', 'pl'])
bo = pd.read_parquet(T1/'R3/cache_final/o2019/baseline_scores.parquet')
out['frozen_y_u_sum'] = int(fz.y_u.sum())
FL = pd.read_parquet(B/'분석/검증루프_20261012/C15/C15-T2/cache/t1/final_labels.parquet')
assert (FL.firm.values == fz.firm.values).all() and (FL.k.values == fz.k.values).all() and (FL.prior_any.values == fz.prior_any.values).all()
fz['y_u'] = FL.y_u.values; out['label_src'] = 'C15-T2 final_labels y_u'
for c in BK: fz[c] = bo[c].values
ins = pd.read_parquet(CA/'ins_firm.parquet', columns=['rgst_no', 'applicant_cd', 'ay', 'appl_date', 'rgst_date'])
oos = pd.read_parquet(CA/'oos.parquet', columns=['applicant_cd', 'rgst_no', 'ay', 'appl_date', 'rgst_date', 'self'])
oos = oos[oos.self].drop(columns='self')
kr = pd.read_parquet(B/'공모전데이터/작업데이터/kr_registered.parquet', columns=['등록번호', '공개일자'])
pubmap = dict(zip(kr['등록번호'].astype(str), d8(kr['공개일자'].values)))
ins['pub'] = ins.rgst_no.astype(str).map(pubmap).fillna(99999999).astype(np.int64)
ins['r'] = d8(ins.rgst_date.values); oos['r'] = d8(oos.rgst_date.values)
ins['v_rgst'] = ins.r; ins['v_pub'] = np.minimum(ins.r, ins.pub); ins['v_p18'] = np.minimum(ins.v_pub, plus18(ins.appl_date.values))
oos['v_rgst'] = oos.r; oos['v_pub'] = np.minimum(oos.r, plus18(oos.appl_date.values)); oos['v_p18'] = oos.v_pub
U = pd.concat([ins.assign(src='i'), oos.assign(src='o')], ignore_index=True)
U = U[(U.ay <= 2019)]
out['overlap_rgst_in_both_ins_oos_same_firm'] = int(U.duplicated(['applicant_cd', 'rgst_no']).sum() - ins[ins.ay <= 2019].duplicated(['applicant_cd', 'rgst_no']).sum() - oos[oos.ay <= 2019].duplicated(['applicant_cd', 'rgst_no']).sum())
te_firms = set(fz.firm.unique())
res = {}
for flt in ('ay>=2000', 'all'):
    V = U[U.ay >= 2000] if flt == 'ay>=2000' else U
    V = V.drop_duplicates(['applicant_cd', 'rgst_no'])
    coff = V.applicant_cd.value_counts(); Foff = set(coff[coff >= 3].index)
    res[flt] = dict(Foff=len(Foff), te_firms=len(te_firms), te_eq_Foff=Foff == te_firms, te_minus_Foff=len(te_firms - Foff), Foff_minus_te=len(Foff - te_firms))
out['official_firmset_check'] = res
V = U[U.ay >= 2000].drop_duplicates(['applicant_cd', 'rgst_no'])
kept = ~fz.prior_any.values; y = fz.y_u.values.astype(int)
# 공식 blend (공식 채점쌍 전체에서 순위) — 같은 점수를 부분집합에 그대로
s_off_rows = np.full(len(fz), np.nan); s_off_rows[kept] = 0.9 * pct(fz.px.values[kept]) + 0.1 * pct(fz.pl.values[kept])
sk = {}
a = mw_auc(s_off_rows[kept], y[kept]); ba = {c: mw_auc(fz[c].values[kept], y[kept]) for c in BK}; m0 = max(ba.values())
sk['OFF'] = dict(n=int(kept.sum()), pos=int(y[kept].sum()), auc=a, M0=m0, M0_arg=max(ba, key=ba.get), skill=(a - m0) / (1 - m0))
first_reg = pd.concat([ins[['applicant_cd', 'ay']], oos[['applicant_cd', 'ay']]]).groupby('applicant_cd').ay.min()
ap = pd.read_parquet(CA/'app_pub.parquet', columns=['applicant_cd', 'ay'])
first_any = pd.concat([ins[['applicant_cd', 'ay']], oos[['applicant_cd', 'ay']], ap[['applicant_cd', 'ay']]]).groupby('applicant_cd').ay.min()
grp = {}
for vcol in ('v_pub', 'v_p18', 'v_rgst'):
    cv = V[V[vcol] <= 20191231].applicant_cd.value_counts(); Fvis = set(cv[cv >= 3].index)
    fin = fz.firm.isin(Fvis).values; m = kept & fin; dm = kept & ~fin
    # (1) blend 부분집합 재순위(작업 방식)
    s_re = 0.9 * pct(fz.px.values[m]) + 0.1 * pct(fz.pl.values[m])
    a1 = mw_auc(s_re, y[m]); ba = {c: mw_auc(fz[c].values[m], y[m]) for c in BK}; m0 = max(ba.values()); arg = max(ba, key=ba.get)
    # (2) 공식 blend 점수 그대로
    a2 = mw_auc(s_off_rows[m], y[m])
    # (3) M0 고정(xgb_y_u, 공식 arg)
    m0f = ba[sk['OFF']['M0_arg']]
    sk[vcol] = dict(n=int(m.sum()), firms=int(fz.firm[m].nunique()), pos=int(y[m].sum()), D_pairs=int(dm.sum()), D_firms=int(fz.firm[dm].nunique()), D_pos=int(y[dm].sum()),
                    auc_rerank=a1, M0=m0, M0_arg=arg, skill_rerank=(a1 - m0) / (1 - m0), skill_offblend=(a2 - m0) / (1 - m0), skill_rerank_M0fixed=(a1 - m0f) / (1 - m0f),
                    delta_rerank=(a1 - m0) / (1 - m0) - sk['OFF']['skill'])
    if vcol == 'v_pub':
        for nm, mm in (('kept', m), ('dropped', dm)):
            f = pd.Series(fz.firm.values[mm]).drop_duplicates()
            grp[nm] = dict(pairs=int(mm.sum()), firms=int(len(f)), pos=int(y[mm].sum()), pos_rate=float(y[mm].mean()), newc_share=float(fz.newc.values[mm].mean()),
                           first_reg_median=float(f.map(first_reg).median()), first_reg_ge2017=float((f.map(first_reg) >= 2017).mean()),
                           first_any_median=float(f.map(first_any).median()), first_any_ge2017=float((f.map(first_any) >= 2017).mean()),
                           off_count_median=float(f.map(V.applicant_cd.value_counts()).median()), vis_count_median=float(f.map(cv).fillna(0).median()))
        # 빠지는 법인을 쌍 수 대신 법인 단위로 본 진입률(법인당 양성 쌍 수 평균 / 50 근사 회피: 법인별 pos_rate 평균)
        for nm, mm in (('kept', m), ('dropped', dm)):
            g = pd.DataFrame(dict(f=fz.firm.values[mm], y=y[mm])).groupby('f').y.mean(); grp[nm]['firm_mean_pos_rate'] = float(g.mean()); grp[nm]['firm_any_pos'] = float((g > 0).mean())
        # 빠짐 쌍의 hidden 특허 출원연도
        Dset = set(fz.firm[dm].unique()); hid = V[V.applicant_cd.isin(Dset) & (V.v_pub > 20191231)]
        grp['dropped_hidden_ay'] = {str(int(k)): int(v) for k, v in hid.ay.value_counts().sort_index().items()}; grp['dropped_hidden_pubmissing_share'] = float((hid.pub == 99999999).mean()) if 'pub' in hid else None
out['skill'] = sk; out['groups_v_pub'] = grp
# C 쌍 점수 백분위(S1full 재적합 점수) — 산술 재확인
S = pd.read_parquet(B/'분석/검증루프_20261012/C16/C16-T1/cache/build/s1F/scores.parquet', columns=['firm', 'k', 'px', 'pl', 'xgb_y_u'])
import pickle
PP = pickle.load(open(B/'분석/검증루프_20261012/C16/C16-T1/cache/build/pop.pkl', 'rb'))
Fv = set(PP['Fvis']); pv = set(map(tuple, PP['prior_vis'])); Cs = set(map(tuple, PP['C']))
Sk = list(zip(S.firm.values, S.k.values))
isC = np.array([t in Cs for t in Sk]); off_key = set(zip(fz.firm.values, fz.k.values))
pa = dict(zip(zip(fz.firm.values, fz.k.values), fz.prior_any.values))
inoff = np.array([t in off_key for t in Sk]); fin = S.firm.isin(Fv).values
paS = np.array([pa.get(t, False) for t in Sk]); pvS = np.array([t in pv for t in Sk])
sel = (inoff & fin & (~paS | (paS & ~pvS))) | isC
s = 0.9 * pct(S.px.values[sel]) + 0.1 * pct(S.pl.values[sel]); r = pct(s); rb = pct(S.xgb_y_u.values[sel]); c = isC[sel]
out['C_pct'] = dict(n_sel=int(sel.sum()), n_C=int(c.sum()), model_median=float(np.median(r[c])), xgb_y_u_median=float(np.median(rb[c])))
json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
