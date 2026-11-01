# -*- coding: utf-8 -*-
"""C06-T3 s6: (1) 정밀도 표본 수기 판정 집계 (2) 정밀도 반영 시나리오 M_strict 재채점 (3) 선택 편향(새 양성 대 공식 양성) (4) 보안과제 연도 비중."""
import pandas as pd, numpy as np, sys, io, json, pyarrow.parquet as pq
from scipy.stats import rankdata
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = ROOT + r"\분석\모델경쟁_20260927"; SH = MC + r"\_shared\ext"
OUT = ROOT + r"\분석\검증루프_20261012\C06\C06-T3"; CA = OUT + r"\cache"
o = {}
# ---- (1) 수기 판정 (precision_sample_print.txt를 사람이 읽고 매김: 근거 = 문서 출원인명·주소 대 기업 주소, 발명 명칭 대 NTIS 과제명, 권리목록 보유자)
# 코드: OK=같은 기업으로 판단, TRANSFER=권리목록상 보유하나 출원인은 다른 주체(양도·공동), DOUBT=주소는 맞으나 기술 분야가 과제와 동떨어짐, WRONG=주소·분야 모두 불일치(다른 동명 기업)
J = {i: 'OK' for i in range(140)}
for i in (0, 7, 63, 98): J[i] = 'TRANSFER'
for i in (1, 20, 21, 25, 44, 48, 49, 52, 55, 85): J[i] = 'DOUBT'
for i in range(100, 140):
    J[i] = 'WRONG'
for i in (101, 112, 115, 137): J[i] = 'OK'
J[133] = 'DOUBT'
s = pd.read_csv(CA + r"\precision_sample_raw.csv", dtype=str)
s['judge'] = [J[i] for i in range(len(s))]
s.to_csv(OUT + r"\cache\precision_sample_judged.csv", index=False, encoding='utf-8-sig')
o['precision'] = {smp: s[s['sample'] == smp].judge.value_counts().to_dict() for smp in ('M100', 'U40')}
o['precision_M100_by_path'] = s[s['sample'] == 'M100'].groupby('path').judge.value_counts().unstack(fill_value=0).to_dict('index')
from math import sqrt
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return [(c - h) / d, (c + h) / d]
bad = int((s[s['sample'] == 'M100'].judge != 'OK').sum())
o['M100_bad_share'] = bad / 100; o['M100_bad_wilson'] = wilson(bad, 100)
o['M100_doc_window_share'] = float((pd.to_numeric(s[s['sample'] == 'M100'].ad) >= 20220101).mean())
# ---- (2) 정밀도 반영 시나리오
def auc(y, p):
    y = np.asarray(y).astype(int); r = rankdata(np.asarray(p, float)); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
sk = lambda a, b: (a - b) / (1 - b)
pr = pd.read_parquet(MC + r"\N6_정부RnD기업_첫특허전환\final_preds_r5.parquet"); pr['bz'] = pr.bz.astype(str).str.zfill(10)
y0 = pr.y.values.astype(int); pc = pr.p_champ.values; pm = pr.p_m0.values; base = sk(auc(y0, pc), auc(y0, pm))
bzi = {b: i for i, b in enumerate(pr.bz)}
cand = pd.read_parquet(CA + r"\n6_cand.parquet"); p1 = pd.read_parquet(CA + r"\n6_p1_pairs.parquet")
p1['window'] = p1.ad.between(20220101, 20231231); p1['known_le2021'] = p1.known <= 20211231
ok = cand[~cand.rl_other_only]; Mh = ok[ok.addr_sg_eq & ~ok.comp_same_sg]
Ms = pd.concat([p1[p1.name_eq], Mh], ignore_index=True)
def run(pairs, drop):
    flip = set(pairs[pairs.window].bz) & set(pr.bz[pr.y == 0]); drp = set(pairs[pairs.known_le2021].bz) if drop else set()
    y = y0.copy(); y[[bzi[b] for b in flip]] = 1; keep = np.ones(len(y), bool); keep[[bzi[b] for b in drp]] = False
    return y, keep, flip, drp
rng = np.random.default_rng(5)
res = {}
for drop in (True, False):
    y, keep, flip, drp = run(Ms, drop)
    s1 = sk(auc(y[keep], pc[keep]), auc(y[keep], pm[keep]))
    # 정밀도 반영: P2h 전환 기업 중 14%(표본 비OK 비율)를 무작위로 되돌리는 200회
    p2flip = sorted(set(Mh[Mh.window].bz) & flip - set(p1[p1.name_eq & p1.window].bz))
    sims = []
    for _ in range(200):
        back = rng.choice(p2flip, int(round(0.14 * len(p2flip))), replace=False)
        y2 = y.copy(); y2[[bzi[b] for b in back]] = 0
        sims.append(sk(auc(y2[keep], pc[keep]), auc(y2[keep], pm[keep])) - base)
    res[f'M_strict|drop={drop}'] = dict(flips=len(flip), drops=len(drp), n=int(keep.sum()), pos=int(y[keep].sum()), skill=s1, dskill=s1 - base,
                                       dskill_precision_adj=[float(np.mean(sims)), float(np.percentile(sims, 5)), float(np.percentile(sims, 95))])
o['M_strict'] = res
# ---- (3) 선택 편향: 새 양성(M_strict 전환) 대 공식 양성 대 위험집합 음성
firm = pd.read_parquet(CA + r"\n6_firm.parquet")
y, keep, flip, drp = run(Ms, False)
firm['grp_new'] = np.where(firm.bz.isin(flip), 'new_pos', np.where(firm.y == 1, 'off_pos', 'neg'))
firm['log_emp'] = np.log1p(pd.to_numeric(firm.n_emp, errors='coerce'))
def smd(a, b):
    a = a.dropna(); b = b.dropna(); return float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))
G = {k: firm[firm.grp_new == k] for k in ('new_pos', 'off_pos', 'neg')}
o['bias'] = dict(n={k: len(v) for k, v in G.items()},
                 nps_cover={k: float(v.n_emp.notna().mean()) for k, v in G.items()},
                 log_emp_mean={k: float(v.log_emp.mean()) for k, v in G.items()},
                 smd_logemp_new_vs_off=smd(G['new_pos'].log_emp, G['off_pos'].log_emp),
                 smd_logemp_new_vs_neg=smd(G['new_pos'].log_emp, G['neg'].log_emp),
                 grp={k: v.grp.value_counts(normalize=True).round(3).to_dict() for k, v in G.items()},
                 sido_top={k: v.f_sido.value_counts(normalize=True).head(5).round(3).to_dict() for k, v in G.items()})
def tvd(a, b):
    a = a.value_counts(normalize=True); b = b.value_counts(normalize=True); ix = a.index.union(b.index)
    return float(0.5 * (a.reindex(ix, fill_value=0) - b.reindex(ix, fill_value=0)).abs().sum())
o['bias']['tvd_grp_new_vs_off'] = tvd(G['new_pos'].grp, G['off_pos'].grp)
o['bias']['tvd_sido_new_vs_off'] = tvd(G['new_pos'].f_sido.dropna(), G['off_pos'].f_sido.dropna())
# KSIC: 결과창 문서 KSIC 2자리 (새 양성 = 복원 문서, 공식 양성 = app_firm 결과창 문서)
_KP = ROOT + r"\공모전데이터\작업데이터\kr_patents.parquet"; _nm = pq.ParquetFile(_KP).schema_arrow.names
kk = pq.read_table(_KP, columns=[_nm[0], _nm[17]]).to_pandas(); kk.columns = ['appl_no', 'ksic']; kk['appl_no'] = kk.appl_no.astype(str)
kd = dict(zip(kk.appl_no, kk.ksic.astype(str).str[:2]))
fa = pd.read_parquet(CA + r"\n6_fa.parquet")
offdocs = fa[(fa.appl_date >= 20220101) & fa.bz.isin(set(pr.bz[pr.y == 1]))].drop_duplicates(['bz', 'appl_no'])
newdocs = Ms[Ms.window & Ms.bz.isin(flip)].drop_duplicates(['bz', 'appl_no'])
k_off = offdocs.appl_no.astype(str).map(kd); k_new = newdocs.appl_no.astype(str).map(kd)
o['bias']['ksic2_off'] = k_off.value_counts(normalize=True).round(3).to_dict(); o['bias']['ksic2_new'] = k_new.value_counts(normalize=True).round(3).to_dict()
o['bias']['tvd_ksic2_docs_new_vs_off'] = tvd(k_new.dropna(), k_off.dropna())
# 기업 단위 주 KSIC
o['bias']['tvd_ksic2_firm_main'] = tvd(newdocs.assign(k=k_new.values).groupby('bz').k.agg(lambda s: s.value_counts().index[0]),
                                       offdocs.assign(k=k_off.values).groupby('bz').k.agg(lambda s: s.value_counts().index[0]))
# 동명 수(n_same) 분포: 새 양성 기업 코드
o['bias']['new_pos_n_same'] = Ms[Ms.bz.isin(flip)].groupby('bz').n_same.max().describe().round(2).to_dict()
# 점수 백분위
rk = rankdata(pc) / len(pc)
o['bias']['score_pctile'] = {k: float(np.mean(rk[[bzi[b] for b in v.bz]])) for k, v in G.items()}
# ---- (4) 보안과제 연도별 비중(과제 수 기준)
nt = pq.read_table(SH + r"\ntis_project_raw.parquet", columns=['year', 'secret_flag']).to_pandas()
o['secret_share_by_year'] = {int(k): float(v) for k, v in nt[nt.year >= 2010].groupby('year').secret_flag.mean().round(4).items()}
json.dump(o, open(CA + r"\s6_out.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))
