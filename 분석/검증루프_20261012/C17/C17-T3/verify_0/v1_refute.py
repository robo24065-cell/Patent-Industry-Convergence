# C17-T3 verify_0 반박 시도 (PREREG 7절)
# (1) 'B2로 올린 T3-01이 사실 C(C*)다' — C* 조건(원점 시행 코드만, 비선택 근거) 시험
# (2) 'C*로 둔 KSIC 현행 라벨이 사실 B다' — 신설 IPC 문서의 KSIC 라벨이 원천 코드 문서와 다른 산업으로 옮겨 가는지(범위 안팎 이동의 간접 지표)
import os, sys, json
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
W = os.path.join(ROOT, "검증루프_20261012", "C17", "C17-T3")
pan = pd.read_parquet(os.path.join(W, 'cache', 'panel.parquet'))
rm = pd.read_csv(os.path.join(W, 'cache', 'v2_remap.csv'), encoding='utf-8-sig').dropna()
uc = pd.read_csv(os.path.join(W, 'cache', 'u_cells_final.csv'), encoding='utf-8-sig')
sens = pd.read_csv(os.path.join(W, 't3_newipc_sens.csv'))
out = {}
# (1) 선택성: U 셀 양성률 대 나머지, 국가별 U 메인그룹 문서 비중(공개 2019–21), 개발 대비 최종 Δ
P = pd.read_parquet(os.path.join(ROOT, "모델경쟁_20260927", "T3_기술셀부상예측", "R3", "preds", "FINAL_R3_t0_2021.parquet"))
P['U'] = P.cell.isin(set(uc[uc.U].cell))
out['pos_rate_U'] = float(P[P.U].yb.mean()); out['pos_rate_nonU'] = float(P[~P.U].yb.mean()); out['n_U'] = int(P.U.sum())
U_mg = set(rm.new_mg)
w = pan[pan.year.between(2019, 2021)]
w = w.assign(isU=w.mg.isin(U_mg), isSrc=w.mg.isin(set(rm.old_mg)))
by = w.groupby('cc').apply(lambda g: pd.Series(dict(U_docs=g[g.isU].n.sum(), src_docs=g[g.isSrc].n.sum(),
                                                     U_share_of_U_plus_src=g[g.isU].n.sum() / max(1, g[g.isU | g.isSrc].n.sum()))), include_groups=False)
out['cc_U_share_2019_21'] = by.round(4).to_dict('index')
d = sens.set_index(['variant', 'period']).d_skill
po = {r.variant: json.loads(r.per_origin) for _, r in sens[sens.period == 'dev_mean'].iterrows() if isinstance(r.per_origin, str)}
out['final_vs_dev'] = {v: dict(final=round(float(d[(v, 'final')]), 4), dev_mean=round(float(d[(v, 'dev_mean')]), 4),
                               dev_min=round(min(po[v].values()), 4), final_below_dev_min=bool(d[(v, 'final')] < min(po[v].values())))
                       for v in ['V1', 'V1r', 'V48', 'V48r', 'V2', 'V2s']}
# (2) KSIC 라벨: U 문서를 원천 메인그룹으로 되돌렸을 때 (ksic, 원천 mg) 셀이 원천 코드 문서로 이미 존재하는 비율, KR 공개 2019–21 KSIC 분포 TVD(신설 vs 원천)
mp = dict(zip(rm.new_mg, rm.old_mg))
ud = pan[pan.mg.isin(U_mg)].copy(); ud['old'] = ud.mg.map(mp)
src_cells = set(zip(pan[~pan.mg.isin(U_mg)].ksic, pan[~pan.mg.isin(U_mg)].mg))
ud['cell_exists_with_src'] = [(k, o) in src_cells for k, o in zip(ud.ksic, ud.old)]
out['U_docs_in_existing_src_ksic_cell'] = float(ud[ud.cell_exists_with_src].n.sum() / ud.n.sum())
tv = {}
for cc in ['KR', 'CN', 'US']:
    for grp, olds in [('H10→H01L', None)]:
        a = ud[(ud.cc == cc) & ud.year.between(2019, 2021) & ud.mg.str.startswith('H10')]
        b = pan[(pan.cc == cc) & pan.year.between(2019, 2021) & pan.mg.isin(set(a.old))]
        if a.n.sum() == 0 or b.n.sum() == 0: continue
        pa = a.groupby('ksic').n.sum() / a.n.sum(); pb = b.groupby('ksic').n.sum() / b.n.sum()
        ix = pa.index.union(pb.index)
        tv[cc] = dict(tvd=float(0.5 * (pa.reindex(ix).fillna(0) - pb.reindex(ix).fillna(0)).abs().sum()), n_new=int(a.n.sum()), n_src=int(b.n.sum()),
                      new_ksic_not_in_src=int(len(set(pa.index) - set(pb.index))), new_docs_in_ksic_not_in_src=float(pa[~pa.index.isin(pb.index)].sum()))
out['ksic_tvd_H10_vs_src'] = tv
json.dump(out, open(os.path.join(W, 'verify_0', 'v1_refute_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
