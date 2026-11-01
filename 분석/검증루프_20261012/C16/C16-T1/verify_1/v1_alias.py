# -*- coding: utf-8 -*-
"""verify_1: ALIAS 민감도 반박 시도. 원천(트랙 동결 파일)에서 직접 다시 채점(scipy rankdata, s_score 미사용).
 1) 공식 재현 2) ALIAS_i/ii/iii 재계산 3) 무작위 P 양성 220쌍 제거 귀무분포 4) 플래그 구성(e4·e2·코드) 5) C03 의심집합과 겹침"""
import sys, json, pathlib
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.stdout.reconfigure(encoding='utf-8')
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실/R3'; FC = T1/'cache_final'/'o2019'; RD = T1/'runs'/'FROZEN_final'
HERE = B/'분석/검증루프_20261012/C16/C16-T1'; CC = HERE/'cache'; OUT = HERE/'verify_1'
BK = ['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u']

te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'prior_any'])
fz = pd.read_parquet(RD/'o2019.parquet', columns=['firm', 'k', 'px', 'pl'])
assert (fz.firm.values == te.firm.values).all()
bo = pd.read_parquet(FC/'baseline_scores.parquet')
L = pd.read_parquet(CC/'labels.parquet'); L = L[L.src == 'te'].reset_index(drop=True)
assert (L.firm.values == te.firm.values).all() and (L.k.values == te.k.values).all() and (L.prior_any.values == te.prior_any.values).all()


def auc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def blend(px, pl):
    return 0.9 * pd.Series(px).rank(pct=True).values + 0.1 * pd.Series(pl).rank(pct=True).values


def skill(m, y, rows=None):
    s = blend(fz.px.values[m], fz.pl.values[m]); a = auc(s, y)
    b = {c: auc(bo[c].values[m], y) for c in BK}; M0 = max(b.values())
    return dict(auc=a, M0=M0, M0_arg=max(b, key=b.get), skill=(a - M0) / (1 - M0), pos=int(y.sum()), n=int(m.sum()))


kept = ~te.prior_any.values
y_off = L.y_u.values.astype(np.int8)
out = {}
out['OFF'] = skill(kept, y_off[kept])
key = pd.MultiIndex.from_arrays([te.firm.astype(str).values, te.k.values])
pP = pd.read_parquet(CC/'alias_pairs_P.parquet'); pR = pd.read_parquet(CC/'alias_pairs_R.parquet')
DOC = pd.read_parquet(CC/'alias_docs.parquet')
# 플래그 독립 재도출: alias_docs 근거 문서 verdict로부터 쌍 단위 '모든 문서 오매칭'
dP = DOC[DOC.grp == 'P']; g = dP.groupby(['firm', 'k']).wrong.all()
flag_pairs = set(g[g].index); assert flag_pairs == set(zip(pP.firm[pP.all_wrong], pP.k[pP.all_wrong]))
flagP = key.isin(pd.MultiIndex.from_tuples(list(flag_pairs))) & kept
Pm = kept & (L.y_pub.values == 1) & (L.y_reg.values == 0)
out['counts'] = dict(P=int(Pm.sum()), flagP=int(flagP.sum()), flagP_in_P=int((flagP & Pm).sum()), flagP_pos=int(y_off[flagP].sum()),
                     P_pairs_with_docs=int(dP.groupby(['firm', 'k']).ngroups))
m = kept & ~flagP; out['ALIAS_i'] = skill(m, y_off[m])
y2 = y_off.copy(); y2[flagP] = 0; out['ALIAS_ii'] = skill(kept, y2[kept])
dR = DOC[DOC.grp == 'R']; gR = dR.groupby(['firm', 'k']).wrong.all(); flagR = key.isin(pd.MultiIndex.from_tuples(list(gR[gR].index))) & te.prior_any.values
m3 = kept | flagR; out['ALIAS_iii'] = skill(m3, y_off[m3]); out['ALIAS_iv'] = skill(m3, y2[m3])
out['counts']['flagR'] = int(flagR.sum()); out['counts']['flagR_pos'] = int(y_off[flagR].sum())
for kk in ['ALIAS_i', 'ALIAS_ii', 'ALIAS_iii', 'ALIAS_iv']: out[kk]['delta'] = out[kk]['skill'] - out['OFF']['skill']

# 점수 위치: 공식 모집단 내 백분위
s_off = blend(fz.px.values[kept], fz.pl.values[kept]); pct = pd.Series(s_off).rank(pct=True).values
idx = np.where(kept)[0]; fl_k = flagP[kept]; P_k = Pm[kept]; pos_k = y_off[kept] == 1
out['score_pct_median'] = dict(flagged=float(np.median(pct[fl_k])), P_unflagged=float(np.median(pct[P_k & ~fl_k])), reg_pos=float(np.median(pct[pos_k & ~P_k])), neg=float(np.median(pct[~pos_k])))
for c in ['xgb_y_u', 'xgb_y']:
    pb = pd.Series(bo[c].values[kept]).rank(pct=True).values
    out[f'{c}_pct_median'] = dict(flagged=float(np.median(pb[fl_k])), P_unflagged=float(np.median(pb[P_k & ~fl_k])), reg_pos=float(np.median(pb[pos_k & ~P_k])))

# 귀무: P 양성 중 무작위 220쌍 제거(빠른 계산: 양성 제거는 음성 집합 불변 → 양성별 기여 고정; 블렌드 순위는 공식 모집단 고정 근사)
yk = y_off[kept]; neg = yk == 0
def contrib(s):
    r_all = rankdata(s); r_neg = rankdata(s[neg]); # 양성 i의 '음성보다 큰 수 + 동점 0.5' = r_all_i - r_pos_i ... 간단히 searchsorted
    sn = np.sort(s[neg]); sp = s[~neg]
    lo = np.searchsorted(sn, sp, 'left'); hi = np.searchsorted(sn, sp, 'right')
    return (lo + 0.5 * (hi - lo)) / neg.sum()
Cm = contrib(s_off); Cb = np.vstack([contrib(bo[c].values[kept]) for c in BK])
posidx = np.where(~neg)[0]  # 양성 위치(kept 기준)
isP_pos = P_k[~neg]; isF_pos = fl_k[~neg]
def sk_drop(dropmask):
    keep = ~dropmask; a = Cm[keep].mean(); M0 = Cb[:, keep].mean(1).max(); return (a - M0) / (1 - M0)
base = sk_drop(np.zeros(len(posidx), bool)); out['fast_check'] = dict(off=base, flag=sk_drop(isF_pos))
rng = np.random.RandomState(20261017); Pi = np.where(isP_pos & ~isF_pos)[0]; Pall = np.where(isP_pos)[0]; Ai = np.arange(len(posidx))
null = {'P_unflagged': [], 'P_any': [], 'all_pos': []}
for _ in range(1000):
    for nm, pool in (('P_unflagged', Pi), ('P_any', Pall), ('all_pos', Ai)):
        d = np.zeros(len(posidx), bool); d[rng.choice(pool, 220, replace=False)] = True; null[nm].append(sk_drop(d) - base)
out['null_drop220'] = {k: dict(mean=float(np.mean(v)), p2_5=float(np.percentile(v, 2.5)), p97_5=float(np.percentile(v, 97.5)), max=float(np.max(v))) for k, v in null.items()}
out['flag_delta_fast'] = out['fast_check']['flag'] - base

# 플래그 구성
fd = dP[dP.set_index(['firm', 'k']).index.isin(list(flag_pairs))]
out['flag_docs_verdict'] = fd.verdict.value_counts().to_dict()
out['flag_docs_e4_name_history_true'] = int(fd.e4_name_history.sum()); out['flag_docs'] = int(len(fd))
out['flag_docs_addr_mismatch_and_e4'] = int(((fd.e2 == 'mismatch') & fd.e4_name_history).sum())
out['flag_docs_is_reg'] = int(fd.is_reg.sum())
out['flag_name_len_le3'] = int((fd.name_len <= 3).sum())
# 플래그 쌍 중 주소불일치만으로 플래그된 쌍(이력 일치 있음)의 Δ
pair_e4 = fd.groupby(['firm', 'k']).agg(addr_only=('verdict', lambda v: v.str.contains('주소').all()), any_e4=('e4_name_history', 'any'))
strict = set(pair_e4.index[~(pair_e4.addr_only & pair_e4.any_e4)])
fs = key.isin(pd.MultiIndex.from_tuples(list(strict))) & kept
m = kept & ~fs; out['ALIAS_i_strict_excl_addr_only_with_name_history'] = dict(pairs=int(fs.sum()), **skill(m, y_off[m]))
out['ALIAS_i_strict_excl_addr_only_with_name_history']['delta'] = out['ALIAS_i_strict_excl_addr_only_with_name_history']['skill'] - out['OFF']['skill']
json.dump(out, open(OUT/'v1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
