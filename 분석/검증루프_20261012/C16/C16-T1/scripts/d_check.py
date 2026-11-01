# -*- coding: utf-8 -*-
"""data_check 수치: (a) 총량 대조 (b) 분포·결측 (c) 연결 선택 편향 (d) 시점 정합 → cache/d_check.json"""
import sys, pathlib, sqlite3, pickle; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard16 as G
from c16common import *
out = {}
ins = load_ins(); oos = load_oos(); a = load_app()
con = sqlite3.connect((B/'수집파이프라인/registry_full.sqlite').as_uri() + '?mode=ro', uri=True)
pt = pd.read_sql('select rgst_no, applDate, rgstDate, opnDate from patent', con); con.close()
kr = pd.read_parquet(KR, columns=['등록번호', '공개일자', '출원연도', '등록일자'])
out['a_totals'] = dict(kr_registered=len(kr), registry_patent=len(pt), ins_firm_rows=len(pd.read_parquet(CACHE/'ins_firm.parquet', columns=['rgst_no'])), oos_self_2000=len(oos),
                       kr_patents=int(pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호']).shape[0]), app_pub=len(pd.read_parquet(CACHE/'app_pub.parquet', columns=['appl_no'])))
# (d) 공개일 결측·일치
kr['pub'] = s8(kr['공개일자']); pt['opn'] = s8(pt.opnDate); m = kr.merge(pt[['rgst_no', 'opn', 'applDate']], left_on='등록번호', right_on='rgst_no', how='left')
m['ry'] = pd.to_numeric(m['등록일자'].astype(str).str[:4], errors='coerce')
both = (m.pub < 99999999) & (m.opn < 99999999)
out['d_pubdate'] = dict(kr_pub_missing=float((m.pub == 99999999).mean()), registry_opn_missing=float((m.opn == 99999999).mean()),
                        both_present_equal=float((m.pub[both] == m.opn[both]).mean()), kr_missing_but_registry_has=int(((m.pub == 99999999) & (m.opn < 99999999)).sum()),
                        kr_missing_by_ay={str(int(k)): float(v) for k, v in m.groupby('출원연도').pub.apply(lambda s: (s == 99999999).mean()).items() if 2010 <= k <= 2023},
                        registry_missing_by_ay={str(int(k)): float(v) for k, v in m.groupby('출원연도').opn.apply(lambda s: (s == 99999999).mean()).items() if 2010 <= k <= 2023},
                        appl_year_mismatch_kr_vs_registry=int((pd.to_numeric(m.applDate.astype(str).str[:4], errors='coerce') != m['출원연도']).sum()))
# 출원연도별 원점(2019-12-31) 가시 비율(범위안, v_pub)
ip = ins[(ins.ay >= 2014) & (ins.ay <= 2019)].drop_duplicates('rgst_no')
out['d_visible_share_by_ay'] = {str(y): dict(v_pub=float((g.v_pub <= CAPV).mean()), v_rgst=float((g.v_rgst <= CAPV).mean()), n=int(len(g))) for y, g in ip.groupby('ay')}
# (b)(c) 빠지는 법인(D) vs 남는 법인
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'newc', 'prior_any', 'lsize'])
L = pd.read_parquet(CC/'labels.parquet'); L = L[L.src == 'te'].reset_index(drop=True); te['y'] = L.y_u.values
Q = pd.read_parquet(CC/'pop_te_v_pub.parquet'); te['fin'] = Q.fin.values
fm = pd.read_parquet(CACHE/'firm_meta.parquet').set_index('applicant_cd')
k_ = te[~te.prior_any]
grp = {}
for nm, g in (('남음(A)', k_[k_.fin]), ('빠짐(D)', k_[~k_.fin])):
    f = g.drop_duplicates('firm')
    first = pd.concat([ins[['applicant_cd', 'ay']], oos[['applicant_cd', 'ay']]]).groupby('applicant_cd').ay.min()
    grp[nm] = dict(pairs=int(len(g)), firms=int(len(f)), pos=int(g.y.sum()), pos_rate=float(g.y.mean()), newcomer_pair_share=float(g.newc.mean()),
                   foreign_firm_share=float(f.firm.map(fm.tp).eq('외국법인').mean()), lsize_mean=float(f.lsize.mean()),
                   first_filing_year_median=float(f.firm.map(first).median()), first_filing_ge2017_share=float((f.firm.map(first) >= 2017).mean()))
out['b_dropped_vs_kept'] = grp
# C 쌍 특성
PP = pickle.load(open(CC/'build'/'pop.pkl', 'rb')); C = pd.DataFrame(PP['C'], columns=['firm', 'k'])
K, ki = kidx(); C['ksic'] = [K[i] for i in C.k]
hid = ins[ins.ay <= CUT].assign(k=lambda d: d.ksic.map(ki)).merge(C[['firm', 'k']], left_on=['applicant_cd', 'k'], right_on=['firm', 'k'])
out['b_C_pairs'] = dict(n=int(len(C)), firms=int(C.firm.nunique()), top_ksic=C.ksic.value_counts().head(5).to_dict(), ksic_top5_share=float(C.ksic.value_counts().head(5).sum() / len(C)),
                        foreign_firm_share=float(C.drop_duplicates('firm').firm.map(fm.tp).eq('외국법인').mean()),
                        hidden_filing_ay=hid.drop_duplicates(['firm', 'k', 'rgst_no']).ay.value_counts().sort_index().to_dict(),
                        hidden_filing_rgst_year=pd.to_numeric(hid.rgst_date.astype(str).str[:4]).value_counts().sort_index().to_dict())
# (c) 이름매칭 커버리지(출원연도별 kr_patents 중 법인 매칭 비율) — 선택 편향 점검
kp = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '출원연도', '등록번호'])
kp['matched'] = kp['출원번호'].astype(str).isin(set(pd.read_parquet(CACHE/'app_pub.parquet', columns=['appl_no']).appl_no.astype(str)))
kp['reg'] = kp['등록번호'].fillna('').astype(str).str.len() > 3
out['c_name_match_coverage'] = {str(y): dict(all=float(g.matched.mean()), registered=float(g.matched[g.reg].mean()), unregistered=float(g.matched[~g.reg].mean()), n=int(len(g)))
                                for y, g in kp.groupby('출원연도') if 2012 <= y <= 2023}
DOC = pd.read_parquet(CC/'alias_docs.parquet')
d = DOC[(DOC.grp == 'P') & ~DOC.is_reg]
out['c_flag_by_firmtype'] = {str(t): dict(docs=int(len(g)), wrong_rate=float(g.wrong.mean())) for t, g in d.groupby(d.firm.map(fm.tp))}
out['c_flag_by_name_len'] = {('≤3' if l <= 3 else ('4–5' if l <= 5 else '≥6')): None for l in [3, 5, 6]}
out['c_flag_by_name_len'] = {lab: dict(docs=int(len(g)), wrong_rate=float(g.wrong.mean())) for lab, g in d.groupby(pd.cut(d.name_len, [0, 3, 5, 100], labels=['≤3', '4–5', '≥6']), observed=True)}
out['peak'] = G.rss_peak()
jdump(out, CC/'d_check.json'); print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
