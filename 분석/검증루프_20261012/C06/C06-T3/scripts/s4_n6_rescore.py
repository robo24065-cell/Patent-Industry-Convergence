# -*- coding: utf-8 -*-
"""C06-T3 s4: N6 공란 복원 시나리오(L/M/U) 재채점 — 모형 고정(final_preds_r5), 레이블·위험집합만 변경."""
import pandas as pd, numpy as np, os, sys, io, json, re
from scipy.stats import rankdata
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = ROOT + r"\분석\모델경쟁_20260927"
OUT = ROOT + r"\분석\검증루프_20261012\C06\C06-T3"; CA = OUT + r"\cache"
o = {}
def auc(y, p):
    y = np.asarray(y).astype(int); r = rankdata(np.asarray(p, float)); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
sk = lambda a, b: (a - b) / (1 - b)
pr = pd.read_parquet(MC + r"\N6_정부RnD기업_첫특허전환\final_preds_r5.parquet"); pr['bz'] = pr.bz.astype(str).str.zfill(10)
pr = pr.reset_index(drop=True)
cand = pd.read_parquet(CA + r"\n6_cand.parquet"); p1 = pd.read_parquet(CA + r"\n6_p1_pairs.parquet")
p1['window'] = p1.ad.between(20220101, 20231231); p1['known_le2021'] = p1.known <= 20211231
y0 = pr.y.values.astype(int); pc = pr.p_champ.values; pm = pr.p_m0.values
base = dict(auc=auc(y0, pc), auc_m0=auc(y0, pm)); base['skill'] = sk(base['auc'], base['auc_m0'])
def top1(y, p):
    k = max(1, int(round(0.01 * len(y)))); idx = np.argsort(-p, kind='mergesort')[:k]; return k, int(y[idx].sum())
base['top1'] = top1(y0, pc); base['top1_m0'] = top1(y0, pm)
o['base'] = base
# 시나리오별 쌍 집합
okc = cand[~cand.rl_other_only]
S = {
    'L': p1,
    'L_strict(이름 일치 P1만)': p1[p1.name_eq],
    'M': pd.concat([p1, okc[okc.addr_sg_eq & ~okc.comp_same_sg]], ignore_index=True),
    'M_sido(시도만 일치·경쟁 주소 없음)': pd.concat([p1, okc[okc.addr_sido_eq & ~okc.comp_same_sg]], ignore_index=True),
    'U': pd.concat([p1, okc], ignore_index=True),
    'U_window_only(동명 전부, 위험집합 제외 없음)': pd.concat([p1[p1.window], okc[okc.window]], ignore_index=True),
}
bzidx = {b: i for i, b in enumerate(pr.bz)}
def apply(pairs, drop=True):
    flip = set(pairs[pairs.window].bz) & set(pr.bz[pr.y == 0])
    drp = set(pairs[pairs.known_le2021].bz) if drop else set()
    y = y0.copy(); y[[bzidx[b] for b in flip]] = 1
    keep = np.ones(len(y), bool); keep[[bzidx[b] for b in drp if b in bzidx]] = False
    return y, keep, flip, drp
rng = np.random.default_rng(20261012)
NB = 1000
boots = [rng.integers(0, len(y0), len(y0)) for _ in range(NB)]
def dskill_boot(y, keep):
    out = []
    for i in boots:
        yb0 = y0[i]; a = sk(auc(yb0, pc[i]), auc(yb0, pm[i]))
        j = i[keep[i]]; yb = y[j]; b = sk(auc(yb, pc[j]), auc(yb, pm[j])); out.append(b - a)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
R = {}
for nm, pairs in S.items():
    for drop in (True, False):
        if not drop and nm.startswith('U_window'): continue
        y, keep, flip, drp = apply(pairs, drop)
        a1, a0 = auc(y[keep], pc[keep]), auc(y[keep], pm[keep]); s1 = sk(a1, a0)
        r = dict(pairs=len(pairs), flips_0to1=len(flip), drops=len(drp), drops_y1=int(sum(y0[bzidx[b]] for b in drp if b in bzidx)),
                 n=int(keep.sum()), pos=int(y[keep].sum()), auc=a1, auc_m0=a0, skill=s1, dskill=s1 - base['skill'], dauc=a1 - base['auc'],
                 top1=top1(y[keep], pc[keep]), top1_m0=top1(y[keep], pm[keep]),
                 flips_pctile_champ=float(np.mean([rankdata(pc)[bzidx[b]] / len(pc) for b in flip])) if flip else None)
        r['dskill_ci'] = dskill_boot(y, keep)
        R[f'{nm}|drop={drop}'] = r
o['scenarios'] = R
# 판정
W = 0.185 - 0.115
def verdict(d, s1, ci_lo_pos=True):
    a = abs(d)
    if a < 0.1 * W: return '영향 없음'
    if a <= 0.25 * W: return '단서 필요'
    return '수정 필요'
o['judge'] = {k: dict(dskill=v['dskill'], share_of_ciwidth=abs(v['dskill']) / W, verdict=verdict(v['dskill'], v['skill'])) for k, v in R.items()}
# 등록 가중(보조): 권리목록 조회 코드의 공란 문서 중 실제 등록 확인 비율
ca = pd.read_parquet(CA + r"\ca.parquet"); rl = pd.read_parquet(CA + r"\rl_blank.parquet")
import duckdb
o['note'] = 'P2u(유일 상호) 후보 0쌍: app_firm 이름 경로가 이미 잡음. 후보는 모두 동명(P2h).'
json.dump(o, open(OUT + r"\n6_rescore_blank.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))
# 산출 CSV: 위험집합 기업별 시나리오 레이블
out = pr[['bz', 'y', 'p_champ', 'p_m0']].copy()
for nm in ('L', 'M', 'U'):
    y, keep, flip, drp = apply(S[nm], True)
    out[f'y_{nm}'] = y; out[f'in_risk_{nm}'] = keep.astype(int)
cnt = pd.concat([S['U'].assign(src=lambda d: np.where(d.index < len(p1), 'P1', 'P2h'))]).groupby('bz').agg(n_pairs=('appl_no', 'size'))
out = out.merge(cnt, left_on='bz', right_index=True, how='left').fillna({'n_pairs': 0})
out.to_csv(OUT + r"\n6_blank_full.csv", index=False, encoding='utf-8-sig')
