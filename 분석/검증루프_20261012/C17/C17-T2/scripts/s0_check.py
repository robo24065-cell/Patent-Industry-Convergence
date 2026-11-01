# -*- coding: utf-8 -*-
"""C17-T2 s0: 입력 검증과 P0.
 (1) C16 labels.parquet vs C03 eval_o2019(독립 재구성) 쌍별 레이블 일치, 공개 경로 양성 수 대조(a)
 (2) C16 alias_docs(근거 문서 목록)를 labels+app_pub에서 다시 만들어 집합 대조, kr 출원인명·주소 재대조
 (3) P0: 동결 예측·기준선으로 공식 skill 0.47267044268107444 재현(scipy rankdata)
 결과: cache/s0_out.json, cache/eval_base.npz(채점용 배열)"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
from scipy.stats import rankdata
t0 = time.time(); out = {}
L = pd.read_parquet(C16/'cache/labels.parquet'); L = L[L.src == 'te'].reset_index(drop=True)
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'prior_any'])
fz = pd.read_parquet(RD/'o2019.parquet', columns=['firm', 'k', 'px', 'pl', 'y_u', 'y_reg', 'prior_any'])
assert (L.firm.values == te.firm.values).all() and (L.k.values == te.k.values).all() and (fz.firm.values == te.firm.values).all()
out['labels_rows'] = len(L)
out['labels_vs_frozen'] = dict(prior_any_diff=int((L.prior_any.values != fz.prior_any.values).sum()),
                               y_u_diff_kept=int((L.y_u.values[~L.prior_any.values] != fz.y_u.values[~L.prior_any.values]).sum()),
                               y_reg_diff_kept=int((L.y_reg.values[~L.prior_any.values] != fz.y_reg.values[~L.prior_any.values]).sum()))
kept = ~L.prior_any.values
E = pd.read_parquet(C03/'cache/eval_o2019.parquet', columns=['firm', 'k', 'yA_reg', 'yA_pub'])
Lk = L[kept].reset_index(drop=True)
m = Lk.merge(E, on=['firm', 'k'], how='outer', indicator=True)
out['C03_eval_merge'] = m._merge.value_counts().to_dict()
mb = m[m._merge == 'both']
out['C03_label_diff'] = dict(reg=int((mb.y_reg != mb.yA_reg).sum()), pub=int((mb.y_pub != mb.yA_pub).sum()))
out['counts'] = dict(eval_rows=int(kept.sum()), pos=int(Lk.y_u.sum()), pos_reg=int(Lk.y_reg.sum()), pos_pub=int(Lk.y_pub.sum()),
                     pub_only=int(((Lk.y_pub == 1) & (Lk.y_reg == 0)).sum()), reg_only=int(((Lk.y_pub == 0) & (Lk.y_reg == 1)).sum()),
                     both=int(((Lk.y_pub == 1) & (Lk.y_reg == 1)).sum()), R_pairs=int(L.prior_any.sum()))
out['counts_C03_reference'] = dict(pos=7723, pos_reg=4733, pos_pub=6675, pub_only=2990, reg_only=1048, rows=1369514)
# (2) 근거 문서 재구성
a = pd.read_parquet(TC/'app_pub.parquet'); K = sorted(pd.read_parquet(TC/'ins_pat.parquet', columns=['ksic']).ksic.unique()); ki = {k: i for i, k in enumerate(K)}
a['k'] = a.ksic.astype(str).map(ki); a = a.dropna(subset=['k']); a['k'] = a.k.astype(int); a = a[a.ay >= 2000]
def s8(x):
    s = pd.Series(x).fillna('').astype(str).str.strip().str[:8]
    return pd.to_numeric(s.where(s.str.len() == 8, '99999999'), errors='coerce').fillna(99999999).astype(np.int64).values
a['v'] = np.minimum(s8(a.pub_date.values), s8(a.rgst_date.values))
P = L[kept & (L.y_pub.values == 1) & (L.y_reg.values == 0)][['firm', 'k']]; R = L[L.prior_any.values][['firm', 'k']]
aP = a[(a.ay > 2019) & (a.ay <= 2023) & (a.v <= 20250822)].merge(P, left_on=['applicant_cd', 'k'], right_on=['firm', 'k'])
aR = a[a.ay <= 2019].merge(R, left_on=['applicant_cd', 'k'], right_on=['firm', 'k'])
mine = set(zip(['P'] * len(aP), aP.firm.astype(str), aP.k, aP.appl_no.astype(str))) | set(zip(['R'] * len(aR), aR.firm.astype(str), aR.k, aR.appl_no.astype(str)))
D = pd.read_parquet(C16/'cache/alias_docs.parquet')
theirs = set(zip(D.grp, D.firm.astype(str), D.k, D.appl_no.astype(str)))
out['alias_docs_rebuild'] = dict(mine=len(mine), c16=len(theirs), only_mine=len(mine - theirs), only_c16=len(theirs - mine),
                                 P_pairs_with_docs=int(aP.groupby(['firm', 'k']).ngroups), R_pairs_with_docs=int(aR.groupby(['firm', 'k']).ngroups))
kp = pd.read_parquet(KP, columns=['출원번호', '등록번호', '출원인명', '출원인주소', '공개일자', '출원일', '등록일자'])
kp['출원번호'] = kp['출원번호'].astype(str)
kd = kp.drop_duplicates('출원번호').set_index('출원번호')
nm = D.appl_no.astype(str).map(kd['출원인명']).fillna('')
out['name_raw_in_kr_names'] = float(np.mean([(r == '') or (r in n.split('|')) for r, n in zip(D.name_raw.fillna(''), nm)]))
out['kr_dup_appl_no'] = int(kp['출원번호'].duplicated().sum())
out['kr_snapshot'] = dict(max_pub=str(kp['공개일자'].astype(str).max()), max_appl=str(kp['출원일'].astype(str).max()), max_rgst=str(kp['등록일자'].astype(str).max()))
# (3) P0
bo = pd.read_parquet(FC/'baseline_scores.parquet')
def auc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
s = 0.9 * pd.Series(fz.px.values[kept]).rank(pct=True).values + 0.1 * pd.Series(fz.pl.values[kept]).rank(pct=True).values
y = L.y_u.values[kept].astype(np.int8)
am = auc(s, y); ab = {c: auc(bo[c].values[kept], y) for c in BK}; M0 = max(ab.values())
sk = (am - M0) / (1 - M0)
out['P0'] = dict(auc=am, M0=M0, M0_arg=max(ab, key=ab.get), skill=sk, official=0.47267044268107444, diff=sk - 0.47267044268107444, pass_=abs(sk - 0.47267044268107444) < 1e-6)
np.savez(CC/'eval_base.npz', s=s, y=y, base=np.vstack([bo[c].values[kept] for c in BK]), firm=L.firm.values[kept].astype(str), k=L.k.values[kept],
         Rfirm=L.firm.values[~kept].astype(str), Rk=L.k.values[~kept], Ry=L.y_u.values[~kept].astype(np.int8),
         Rs_px=fz.px.values[~kept], Rs_pl=fz.pl.values[~kept], Rbase=np.vstack([bo[c].values[~kept] for c in BK]),
         px=fz.px.values[kept], pl=fz.pl.values[kept], ypub=L.y_pub.values[kept].astype(np.int8), yreg=L.y_reg.values[kept].astype(np.int8))
out['secs'] = round(time.time() - t0); out['peak'] = G.rss_peak()
jdump(out, CC/'s0_out.json'); print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
