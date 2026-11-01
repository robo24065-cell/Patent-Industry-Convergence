# -*- coding: utf-8 -*-
"""Q1(레이블·점수 보지 않음): 공식 모집단 재구성 확인 + 가시 기준 모집단 분할 크기. 산출 cache/pop_*.parquet, cache/q1_pop.json"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from c16common import *
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'newc', 'prior_any'])
ins = load_ins(); oos = load_oos(); a = load_app(); K, ki = kidx()
out = dict(n_te=len(te), n_te_firms=int(te.firm.nunique()), n_kept=int((~te.prior_any).sum()), ins_pub_missing_frac=float(ins.pub_missing.mean()))
ip = ins[ins.ay <= CUT]; op = oos[oos.ay <= CUT]


def counts(vcol):
    i = ip if vcol is None else ip[ip[vcol] <= CAPV]; o = op if vcol is None else op[op[vcol] <= CAPV]
    return pd.concat([i.drop_duplicates(['applicant_cd', 'rgst_no']).applicant_cd, o.drop_duplicates(['applicant_cd', 'rgst_no']).applicant_cd]).value_counts()


coff = counts(None); Foff = set(coff[coff >= 3].index)
out['official_firms_rebuilt'] = len(Foff); out['te_firms_subset_of_rebuilt'] = bool(set(te.firm.unique()) <= Foff)
ipo = ip[ip.applicant_cd.isin(Foff)].assign(k=lambda d: d.ksic.map(ki))
held_off = set(zip(ipo.applicant_cd, ipo.k))
out['official_pairs_rebuilt'] = len(Foff) * 50 - len(held_off)
out['official_pairs_eq_te'] = bool(out['official_pairs_rebuilt'] == len(te))
apr = a[a.ay <= CUT]; prior_all = set(zip(apr.applicant_cd, apr.k))
pa_re = np.array([(f, k) in prior_all for f, k in zip(te.firm, te.k)]); out['prior_any_rebuilt_eq'] = bool((pa_re == te.prior_any.values).all())
res = {}
for vcol in ('v_pub', 'v_p18', 'v_rgst'):
    cv = counts(vcol); Fvis = set(cv[cv >= 3].index)
    ipv = ipo[ipo[vcol] <= CAPV]; held_vis = set(zip(ipv.applicant_cd, ipv.k))
    av = apr[apr.v <= CAPV]; prior_vis = set(zip(av.applicant_cd, av.k))
    fin = te.firm.isin(Fvis).values
    pv = np.array([(f, k) in prior_vis for f, k in zip(te.firm, te.k)])
    kept = ~te.prior_any.values
    A = kept & fin; Bm = te.prior_any.values & fin & ~pv; Bm3 = te.prior_any.values & ~pv
    C = [(f, k) for (f, k) in held_off - held_vis if f in Fvis and (f, k) not in prior_vis]
    C3 = [(f, k) for (f, k) in held_off - held_vis if (f, k) not in prior_vis]
    d = dict(F_vis=len(Fvis), F_vis_minus_off=len(Fvis - Foff), firms_dropped=len(set(te.firm.unique()) - Fvis),
             A_kept_in_Fvis=int(A.sum()), D_dropped=int((kept & ~fin).sum()), B_prior_relaxed_in_Fvis=int(Bm.sum()), B_prior_relaxed_all=int(Bm3.sum()),
             C_new_riskset=len(C), C_firms=len(set(f for f, _ in C)), C_new_riskset_if_all_firms=len(C3),
             S1_n=int(A.sum() + Bm.sum() + len(C)), S2_n=int(A.sum()), S3_n=int(kept.sum() + Bm3.sum()),
             share_pairs_depending_on_post_origin_info_S1=float((int((kept & ~fin).sum()) + int(Bm.sum()) + len(C)) / kept.sum()))
    res[vcol] = d
    pd.DataFrame(C, columns=['firm', 'k']).to_parquet(CC/f'pop_C_{vcol}.parquet', index=False)
    pd.DataFrame(dict(firm=te.firm.values, k=te.k.values, prior_any=te.prior_any.values, fin=fin, prior_vis=pv)).to_parquet(CC/f'pop_te_{vcol}.parquet', index=False)
out['by_vis'] = res; out['mem'] = mem()
jdump(out, CC/'q1_pop.json'); print(json.dumps(out, ensure_ascii=False, indent=1))
