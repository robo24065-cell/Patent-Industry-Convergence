# -*- coding: utf-8 -*-
"""C17-T2 s5: 데이터 검증 의무 (b) 분포·결측·중복 (c) 판정 가능성 선택 편향 (d) 시점(이름·스냅샷)·권리이전형 분리.
 산출: cache/s5_out.json"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
t0 = time.time(); out = {}
PJ = pd.read_parquet(CC/'pair_judg.parquet'); DJ = pd.read_parquet(CC/'doc_judg.parquet')
pp = pd.read_parquet(CC/'p_point.parquet')
P = PJ[PJ.grp == 'P'].merge(pp[['firm', 'k', 'p_main']], on=['firm', 'k']).reset_index(drop=True)
Z = np.load(CC/'eval_base.npz', allow_pickle=True); pct = pd.Series(Z['s']).rank(pct=True).values
key = pd.Series(np.arange(len(pct)), index=pd.MultiIndex.from_arrays([Z['firm'], Z['k']]))
P['score_pct'] = pct[key.reindex(pd.MultiIndex.from_arrays([P.firm.values, P.k.values])).values.astype(int)]
K = sorted(pd.read_parquet(TC/'ins_pat.parquet', columns=['ksic']).ksic.astype(str).unique()); P['ksic'] = [K[int(x)] for x in P.k]
P['det'] = P.pv != 'undet'; P['wrong'] = P.pv == 'wrong'


def tab(by):
    g = P.groupby(by).agg(n=('pv', 'size'), det_rate=('det', 'mean'), code_wrong=('wrong', 'sum'), F_rate=('F', 'mean'), exp_wrong_rate=('p_main', 'mean')).reset_index()
    return g.round(4).astype(str).to_dict('records')


out['b_by_year'] = tab('ay_min'); out['b_by_foreign'] = tab('foreign'); out['b_by_size'] = tab('size')
kt = P.groupby('ksic').agg(n=('pv', 'size'), det_rate=('det', 'mean'), exp_wrong=('p_main', 'sum'), F=('F', 'sum')).reset_index()
kt['exp_wrong_rate'] = kt.exp_wrong / kt.n
out['b_by_ksic_top_expwrong'] = kt.sort_values('exp_wrong', ascending=False).head(12).round(4).astype(str).to_dict('records')
out['b_ksic_n'] = int(len(kt)); out['b_ksic_share_top5_expwrong'] = float(kt.sort_values('exp_wrong', ascending=False).head(5).exp_wrong.sum() / kt.exp_wrong.sum())
# 결측·중복
dp = DJ[DJ.grp == 'P']
out['b_missing'] = dict(docs=int(len(dp)), addr_none=int((dp.e2 == 'none').sum()), name_not_found=int((~dp.name_found).sum()) if 'name_found' in dp else None,
                        firm_region_missing=int((dp.firm_regions.fillna('') == '').sum()))
out['b_homonym_codes'] = dict(docs_name_in_ge2_codes=int((dp.homonym_codes >= 2).sum()), share=float((dp.homonym_codes >= 2).mean()),
                              pairs_ge2=int((P.hom >= 2).sum()),
                              exp_wrong_rate_hom_ge2=float(P[P.hom >= 2].p_main.mean()) if (P.hom >= 2).any() else None,
                              exp_wrong_rate_hom_1=float(P[P.hom < 2].p_main.mean()),
                              code_wrong_hom_ge2=int(P[(P.hom >= 2)].wrong.sum()), code_correct_hom_ge2=int((P[(P.hom >= 2)].pv == 'correct').sum()))
out['b_short_name'] = dict()
nl = dp.groupby(['firm', 'k']).name_len.min().rename('nlen'); P2 = P.join(nl, on=['firm', 'k'])
out['b_short_name'] = dict(pairs_len_le3=int((P2.nlen <= 3).sum()), exp_wrong_rate_le3=float(P2[P2.nlen <= 3].p_main.mean()), exp_wrong_rate_gt3=float(P2[P2.nlen > 3].p_main.mean()),
                           F_rate_le3=float(P2[P2.nlen <= 3].F.mean()), F_rate_gt3=float(P2[P2.nlen > 3].F.mean()))
# (c) 판정 가능성 선택 편향: SMD
def smd(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); s = np.sqrt((a.var() + b.var()) / 2); return float((a.mean() - b.mean()) / s) if s > 0 else 0.0
d1 = P[P.det & (P.det_src != 'A')]; d0 = P[~P.det]
feat = dict(ay_min=lambda d: d.ay_min, log_ndocs=lambda d: np.log1p(d.n_docs), foreign=lambda d: d.foreign.astype(float), size_L=lambda d: (d['size'] == 'L').astype(float),
            size_S=lambda d: (d['size'] == 'S').astype(float), F=lambda d: d.F.astype(float), coapp=lambda d: d.coapp.astype(float), hom_ge2=lambda d: (d.hom >= 2).astype(float), score_pct=lambda d: d.score_pct)
out['c_smd_det_vs_undet'] = {k: dict(det=float(f(d1).mean()), undet=float(f(d0).mean()), smd=smd(f(d1), f(d0))) for k, f in feat.items()}
out['c_n'] = dict(det_RL=int(len(d1)), det_registry=int((P.det_src == 'A').sum()), undet=int(len(d0)))
# 가시율(정확 판정 가능) 연도별: NF vs F
P['vis'] = (P.pv == 'correct') & (P.ptier == 1)
out['c_visible_correct_rate'] = P.groupby(['F', 'ay_min']).vis.mean().round(4).reset_index().astype(str).to_dict('records')
# 판정 가능 F 정확 쌍 vs NF 정확 쌍 점수 백분위
out['c_score_pct_median'] = dict(F_correct=float(P[P.F & P.vis].score_pct.median()), NF_correct=float(P[~P.F & P.vis].score_pct.median()),
                                 F_undet=float(P[P.F & ~P.det].score_pct.median()), NF_undet=float(P[~P.F & ~P.det].score_pct.median()))
# (d) 시점·이름
con = ro(REG); ra = pd.read_sql('select applicantCd, applicantName from applicant', con); con.close()
ra['applicantCd'] = ra.applicantCd.astype(str); cur = ra.drop_duplicates('applicantCd').set_index('applicantCd').applicantName.map(norm).to_dict()
bc = dp[dp.v == 'B1_정확']
out['d_name_timing'] = dict(B1_correct_docs=int(len(bc)), kr_name_eq_registry_current=float(np.mean([cur.get(f) == n for f, n in zip(bc.firm, bc.nn)])),
                            kr_name_ne_current_but_correct=int(sum(cur.get(f) not in (None, n) for f, n in zip(bc.firm, bc.nn))))
con = ro(BIZ); ts = pd.read_sql("select min(ts) a, max(ts) b from applicant_query", con).iloc[0].to_dict(); con.close()
out['d_snapshots'] = dict(kr_patents='공개 ≤ 2025-08-25, 등록 ≤ 2025-08-22(s0)', right_list_query_ts=ts, registry_full='2025-08 수집(파일 2025-08-17)')
# 권리이전형 분리(문서 단위)
for g in ('P', 'R'):
    x = DJ[(DJ.grp == g) & DJ.v.str.startswith('A_오매칭')]
    out[f'd_registry_mismatch_types_{g}'] = dict(n=int(len(x)), by=x.v.value_counts().to_dict(), transfer_share=float(x.v.str.contains('이전').mean()) if len(x) else None)
out['d_P_wrong_pair_types'] = P[P.wrong].ptype.value_counts().to_dict()
out['secs'] = round(time.time() - t0); out['peak'] = G.rss_peak()
jdump(out, CC/'s5_out.json'); print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:7000])
