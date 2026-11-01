# T1 레드팀 — 시점 누수 재탐색(PROTOCOL_R5 §3-4). 모형은 고치지 않고 영향 크기만 추정한다. 읽기 전용(sqlite mode=ro, 동결 예측·캐시).
#  A. 권리목록(2026 스냅샷, '권리자란'=현재 권리자) 기반 피처(t_*·ot_*·디자인·실용신안)의 사후 이전(transfer-out) 편향 — 범위안 특허로 대리 측정
#  B. 법인 모집단 = 2026 권리목록 조회 성공(OK) 법인 → 조회 실패(RC:001) 법인 제외의 생존 조건부
#  C. '≥3건 법인' 위험집합이 원점 이후 등록분에 조건부
#  D. 선행출원 제외(prior_any)·목표 KSIC 등록(원점 이후 등록) 제외 — 사후 조건부 모집단
#  E. 법인 별칭이 원점 이후 이름(2026 조회명·원점 이후 등록 이름)을 포함
#  F. 같은 점검을 개발 원점 c=2010에 적용(개발→최종 격차 가설)
import os, sys, re, json, sqlite3, time, pathlib
import numpy as np, pandas as pd
from scipy.stats import rankdata
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
V = pathlib.Path(__file__).resolve().parent; T1 = V.parent; R3 = T1/'R3'; CACHE = T1/'cache'; ROOT = T1.parent.parent.parent
t_start = time.time(); OUT = {}
sk = lambda a, b: (a - b) / (1 - b)
def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
ro = lambda p: sqlite3.connect('file:' + str(p).replace(os.sep, '/') + '?mode=ro', uri=True)
K = sorted(pd.read_parquet(CACHE/'ins_pat.parquet', columns=['ksic']).ksic.unique()); kidx = {k: i for i, k in enumerate(K)}
ins = pd.read_parquet(CACHE/'ins_firm.parquet', columns=['rgst_no', 'applicant_cd', 'ksic', 'ay', 'rgst_date']); ins = ins[ins.ay >= 2000].copy(); ins['k'] = ins.ksic.map(kidx)
rl = pd.read_parquet(CACHE/'rl_pat.parquet', columns=['applicant_cd', 'rgst_no', 'pert']); rl = rl[rl.pert == '권리자란']
rlset = pd.MultiIndex.from_frame(rl[['applicant_cd', 'rgst_no']].drop_duplicates()); del rl
ins['in_rl'] = pd.MultiIndex.from_frame(ins[['applicant_cd', 'rgst_no']]).isin(rlset)
rg = ro(ROOT/'수집파이프라인'/'registry_full.sqlite')
ow = pd.read_sql("select rgst_no, seq, ownerRgstCsDate d, ownerRgstCsName nm, ownerRgstCsReason rs from owner where seq>0", rg); rg.close()
tr_mask = ow.rs.fillna('').str.contains('전부이전') | ow.nm.isin(['양도', '법인합병', '법인분할', '합병', '신탁계약에 의한 양도', '국가귀속'])
tdate = ow[tr_mask & ow.d.notna()].groupby('rgst_no').d.min()
ins['tdate'] = ins.rgst_no.map(tdate)
OUT['A0_validation'] = dict(in_rl_rate_no_transfer=float(ins[ins.tdate.isna()].in_rl.mean()), in_rl_rate_transferred=float(ins[ins.tdate.notna()].in_rl.mean()),
                            n_ins=int(len(ins)), n_transferred=int(ins.tdate.notna().sum()))
print('A0 권리목록 존재율(이전 없음/이전됨)', OUT['A0_validation'], flush=True)

def origin_block(c, o, y, s, bmax_cols, bs, tag):
    """원점 c 평가 쌍: 법인별 원점까지 가시 범위안 특허 중 원점 이후 이전(권리목록에서 빠짐) 비중과 진입·skill 관계"""
    capd = f'{c}1231'
    v = ins[(ins.ay <= c) & (ins.rgst_date <= capd)].copy()
    v['post'] = (v.tdate > capd).astype(int); v['pre'] = (v.tdate <= capd).astype(int)
    g = v.groupby('applicant_cd').agg(n_vis=('rgst_no', 'size'), n_rl=('in_rl', 'sum'), n_post=('post', 'sum'), n_pre=('pre', 'sum'))
    f = pd.DataFrame(dict(firm=o.firm.values, y=y)).groupby('firm').y.max().rename('any_entry').to_frame().join(g, how='left').fillna(0)
    f['post_sh'] = np.where(f.n_vis > 0, f.n_post / np.maximum(f.n_vis, 1), np.nan); f['snap_ret'] = np.where(f.n_vis > 0, f.n_rl / np.maximum(f.n_vis, 1), np.nan)
    f['szq'] = pd.qcut(np.log1p(f.n_vis).rank(method='first'), 5, labels=False)
    res = dict(n_firms=int(len(f)), firms_with_vis=int((f.n_vis > 0).sum()), firms_post_transfer_any=int((f.n_post > 0).sum()),
               mean_post_share=float(f.post_sh.mean()), mean_snapshot_retention=float(f.snap_ret.mean()))
    # 규모 5분위 안에서 '원점 이후 이전 있음' 법인의 진입률 대비(전체 쌍 기준 아님, 법인 단위 진입 여부)
    tab = []
    for q, gq in f[f.n_vis > 0].groupby('szq'):
        a1 = gq[gq.n_post > 0].any_entry; a0 = gq[gq.n_post == 0].any_entry
        tab.append(dict(size_q=int(q), n_post=int(len(a1)), entry_post=float(a1.mean()) if len(a1) else None, n_none=int(len(a0)), entry_none=float(a0.mean())))
    res['entry_by_size_q'] = tab
    # skill: 깨끗한 법인(원점 이후 이전 0, 가시 특허 있거나 없음) vs 원점 이후 이전 비중 >0
    fp = o.firm.map(f.n_post).fillna(0).values; fv = o.firm.map(f.n_vis).fillna(0).values; psh = np.where(fv > 0, fp / np.maximum(fv, 1), 0)
    def sub(mask):
        yy = y[mask]; ss = s[mask]; bb = max(auc(yy, bs[c_][mask]) for c_ in bmax_cols); am = auc(yy, ss)
        return dict(pairs=int(mask.sum()), pos=int(yy.sum()), auc=am, auc_strongest=bb, skill=sk(am, bb))
    res['skill_all'] = sub(np.ones(len(y), bool)); res['skill_clean_no_post_transfer'] = sub(fp == 0); res['skill_post_transfer_any'] = sub(fp > 0)
    res['skill_excl_post_share_gt10pct'] = sub(psh <= 0.10)
    # 사후 이전 신호의 정보량: 최강 기준선(+모형) 점수에 '원점 이후 이전 비중'을 더한 로지스틱(평가 표본 적합 = 상한)
    from sklearn.linear_model import LogisticRegression
    bcol = max(bmax_cols, key=lambda c_: auc(y, bs[c_]))
    for nm, base in (('기준선', bs[bcol]), ('모형', s)):
        X = np.c_[rankdata(base) / len(base), psh, (fp > 0).astype(float)]
        p = LogisticRegression(max_iter=1000).fit(X, y).predict_proba(X)[:, 1]
        res[f'auc_gain_add_post_transfer_to_{nm}'] = auc(y, p) - auc(y, base)
    print(tag, json.dumps({k_: v_ for k_, v_ in res.items() if k_ != 'entry_by_size_q'}, ensure_ascii=False), flush=True)
    return res, f

# ---------- 최종 원점 c=2019 ----------
o = pd.read_parquet(R3/'runs'/'FROZEN_final'/'o2019.parquet', columns=['firm', 'k', 'newc', 'px', 'pl', 'prior_any'])
lab = pd.read_parquet(V/'labels_c2019_official.parquet'); assert (lab.firm.values == o.firm.values).all()
bsall = pd.read_parquet(R3/'cache_final'/'o2019'/'baseline_scores.parquet')
keep = ~o.prior_any.values; ok_ = o[keep].reset_index(drop=True); y = lab.y_u.values[keep]; n = len(ok_)
s = 0.9 * rankdata(ok_.px.values) / n + 0.1 * rankdata(ok_.pl.values) / n
bs = {c_: bsall[c_].values[keep] for c_ in bsall.columns}
OUT['A_final_c2019'], ffin = origin_block(2019, ok_, y, s, list(bs), bs, 'A 최종 c=2019')

# ---------- F. 개발 원점 c=2010 ----------
od = pd.read_parquet(R3/'runs'/'FROZEN_dev'/'o2010.parquet', columns=['firm', 'k', 'newc', 'px', 'pl', 'prior_any', 'y_u'])
bd = pd.read_parquet(R3/'cache'/'o2010'/'baseline_scores.parquet')
kd = ~od.prior_any.values; odk = od[kd].reset_index(drop=True); nd = len(odk)
sd_ = 0.9 * rankdata(odk.px.values) / nd + 0.1 * rankdata(odk.pl.values) / nd
OUT['F_dev_c2010'], _ = origin_block(2010, odk, odk.y_u.values.astype(int), sd_, list(bd.columns), {c_: bd[c_].values[kd] for c_ in bd.columns}, 'F 개발 c=2010')

# ---------- B. RC:001(2026 조회 실패) 법인 제외 ----------
firms = pd.read_parquet(CACHE/'firms.parquet', columns=['applicant_cd', 'status'])
rg = ro(ROOT/'수집파이프라인'/'registry_full.sqlite')
app = pd.read_sql("select rgst_no, applicantCd from applicant", rg); pat = pd.read_sql("select rgst_no, ksic, rgstDate from patent", rg); rg.close()
kr = pd.read_parquet(ROOT/'공모전데이터'/'작업데이터'/'kr_registered.parquet', columns=['등록번호', '출원연도']); kr.columns = ['rgst_no', 'ay']; kr['rgst_no'] = kr.rgst_no.astype(str)
pp = app.merge(pat, on='rgst_no').merge(kr, on='rgst_no'); pp = pp[pp.ay >= 2000]; pp['k'] = pp.ksic.astype(str).map(kidx)
rc = set(firms[(firms.status != 'OK') & firms.applicant_cd.str[0].isin(['1', '2', '3', '5'])].applicant_cd)
vis = pp[(pp.ay <= 2019) & (pp.rgstDate <= '20191231')]
cnt = vis.groupby('applicantCd').rgst_no.nunique()
exf = cnt[(cnt >= 3) & cnt.index.isin(rc)]
held = vis[vis.applicantCd.isin(exf.index)].groupby('applicantCd').k.apply(set)
fut = pp[(pp.ay > 2019) & (pp.ay <= 2023) & pp.applicantCd.isin(exf.index)].groupby('applicantCd').k.apply(set)
rows = []
for f_, hs in held.items():
    fs_ = fut.get(f_, set())
    for kk in range(50):
        if kk not in hs: rows.append((f_, kk, int(kk in fs_)))
EX = pd.DataFrame(rows, columns=['firm', 'k', 'y_reg'])
# 패널 쌍의 등록 기준 레이블(비교용)과 규모 구간
ins_fut = ins[(ins.ay > 2019) & (ins.ay <= 2023)]; regset = pd.MultiIndex.from_frame(ins_fut[['applicant_cd', 'k']].dropna().astype({'k': int}).drop_duplicates())
y_reg_panel = pd.MultiIndex.from_arrays([ok_.firm.values, ok_.k.values]).isin(regset).astype(int)
lin_panel = ok_.firm.map(cnt).fillna(0).values; EX['nvis'] = EX.firm.map(cnt).values
bins = [3, 5, 10, 20, 50, 1e9]; bp = np.digitize(lin_panel, bins); be = np.digitize(EX.nvis.values, bins)
OUT['B_rc001_excluded'] = dict(n_rc001_corporate_codes=int(len(rc)), excluded_firms_ge3_inscope=int(len(exf)), excluded_pairs=int(len(EX)),
                              excluded_entry_rate_reg=float(EX.y_reg.mean()), excluded_firms_any_entry=float(EX.groupby('firm').y_reg.max().mean()),
                              panel_entry_rate_reg_same_size=float(y_reg_panel[lin_panel >= 3].mean()),
                              panel_firms_any_entry_reg_same_size=float(pd.Series(y_reg_panel[lin_panel >= 3]).groupby(ok_.firm.values[lin_panel >= 3]).max().mean()))
# 영향 추정: 제외 법인 쌍을 '같은 KSIC × 같은 범위안 규모 구간' 패널 쌍의 평균 순위 점수로 대입해 넣었을 때(실시간이라면 권리목록 결손 없이 비슷한 규모 법인처럼 보였을 것)
dfp = pd.DataFrame(dict(k=ok_.k.values, b=bp, s=s, **{c_: rankdata(v_) / n for c_, v_ in bs.items()}))
gm = dfp.groupby(['k', 'b']).mean()
EXs = gm.reindex(pd.MultiIndex.from_arrays([EX.k.values, be])).values
yy = np.r_[y, EX.y_reg.values]
sc = np.r_[s, EXs[:, 0]]; am = auc(yy, sc)
bb = max(auc(yy, np.r_[rankdata(v_) / n, EXs[:, j + 1]]) for j, (c_, v_) in enumerate(bs.items()))
OUT['B_rc001_excluded']['imputed_add_back'] = dict(auc=am, auc_strongest=bb, skill=sk(am, bb), skill_official=sk(auc(y, s), max(auc(y, v_) for v_ in bs.values())))
print('B RC:001', OUT['B_rc001_excluded'], flush=True)

# ---------- C. ≥3건 기준의 사후 조건부 ----------
oos = pd.read_parquet(CACHE/'oos.parquet', columns=['applicant_cd', 'rgst_no', 'ay', 'rgst_date', 'self']); oos = oos[oos.self & (oos.ay >= 2000)]
v_ins = ins[(ins.ay <= 2019) & (ins.rgst_date <= '20191231')].groupby('applicant_cd').rgst_no.nunique()
v_oos = oos[(oos.ay <= 2019) & (oos.rgst_date <= '20191231')].groupby('applicant_cd').rgst_no.nunique()
vis_cnt = ok_.firm.map(v_ins).fillna(0).values + ok_.firm.map(v_oos).fillna(0).values
m3 = vis_cnt >= 3
yy = y[m3]; am = auc(yy, s[m3]); bb = max(auc(yy, v_[m3]) for v_ in bs.values())
OUT['C_ge3_postc'] = dict(firms_lt3_visible_at_c=int(pd.Series(ok_.firm.values[~m3]).nunique()), pairs_lt3=int((~m3).sum()), pos_lt3=int(y[~m3].sum()),
                          rate_lt3=float(y[~m3].mean()), rate_ge3=float(yy.mean()), skill_visible_ge3_only=sk(am, bb))
print('C ≥3', OUT['C_ge3_postc'], flush=True)

# ---------- D. 선행출원 제외·사후 등록 제외 쌍 ----------
yall = lab.y_u.values; sall = 0.9 * rankdata(o.px.values) / len(o) + 0.1 * rankdata(o.pl.values) / len(o)
am = auc(yall, sall); bb = max(auc(yall, bsall[c_].values) for c_ in bsall.columns)
ap_ = pd.read_parquet(CACHE/'app_pub.parquet', columns=['applicant_cd', 'ay', 'pub_date', 'rgst_date', 'ksic']); ap_ = ap_[ap_.ay >= 2000].copy()
ap_['k'] = ap_.ksic.astype(str).map(kidx); ap_ = ap_.dropna(subset=['k']); ap_['k'] = ap_.k.astype(int)
pdt = ap_.pub_date.fillna('').astype(str); rdt = ap_.rgst_date.fillna('').astype(str)
ap_['vis_c'] = ((pdt.str.len() == 8) & (pdt <= '20191231')) | ((rdt.str.len() == 8) & (rdt <= '20191231'))
pri_vis = pd.MultiIndex.from_frame(ap_[(ap_.ay <= 2019) & ap_.vis_c][['applicant_cd', 'k']].drop_duplicates())
kk_all = pd.MultiIndex.from_arrays([o.firm.values, o.k.values])
prior_vis = kk_all.isin(pri_vis)
# 목표 KSIC 범위안 등록이 원점 이후에야 생긴(출원 ≤2019, 등록 >2019) 쌍 — 패널에서 아예 빠짐(위험집합 Mrisk)
late = ins[(ins.ay <= 2019) & (ins.rgst_date > '20191231')][['applicant_cd', 'k']].drop_duplicates()
early = set(map(tuple, ins[(ins.ay <= 2019) & (ins.rgst_date <= '20191231')][['applicant_cd', 'k']].drop_duplicates().values))
late = late[[(a_, b_) not in early for a_, b_ in late[['applicant_cd', 'k']].values]]
late = late[late.applicant_cd.isin(set(o.firm))]
late_y = pd.MultiIndex.from_frame(late.astype({'k': int})).isin(pd.MultiIndex.from_frame(pd.concat([ins_fut[['applicant_cd', 'k']].dropna().astype({'k': int}),
                                                   ap_[(ap_.ay > 2019) & (ap_.ay <= 2023)][['applicant_cd', 'k']]]).drop_duplicates()))
OUT['D_postc_population'] = dict(prior_pairs_removed=int((~keep).sum()), prior_pairs_visible_at_c=int((prior_vis & ~keep).sum()), prior_pairs_invisible_at_c=int((~prior_vis & ~keep).sum()),
                                 prior_pairs_entry_rate=float(yall[~keep].mean()), skill_including_prior_pairs=sk(am, bb), auc_incl=am, auc_strongest_incl=bb,
                                 late_registered_pairs_excluded_from_panel=int(len(late)), late_pairs_entry_rate=float(late_y.mean()))
# 실시간 모집단 근사: 원점 시점 보이지 않던 선행출원 쌍은 위험집합에 남긴다(가시 선행출원 쌍만 제외)
mrt = ~prior_vis
am = auc(yall[mrt], sall[mrt]); bb = max(auc(yall[mrt], bsall[c_].values[mrt]) for c_ in bsall.columns)
OUT['D_postc_population']['skill_realtime_prior_rule'] = sk(am, bb)
print('D 사후 조건부 모집단', OUT['D_postc_population'], flush=True)

# ---------- E. 별칭의 원점 이후 이름 ----------
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda x: STRIP.sub('', str(x)).lower()
rg = ro(ROOT/'수집파이프라인'/'registry_full.sqlite'); an = pd.read_sql("select a.rgst_no, a.applicantCd, a.applicantName, p.rgstDate from applicant a join patent p using(rgst_no)", rg); rg.close()
fq = pd.read_parquet(CACHE/'firms.parquet', columns=['applicant_cd', 'applicant_name', 'status']); fq = fq[fq.status == 'OK']
an = an[an.applicantCd.isin(set(fq.applicant_cd))]; an['n'] = an.applicantName.map(norm)
pre = set(zip(an[an.rgstDate <= '20191231'].applicantCd, an[an.rgstDate <= '20191231'].n))
kp = pd.read_parquet(ROOT/'공모전데이터'/'작업데이터'/'kr_patents.parquet', columns=['출원번호', '출원연도', '공개일자', '등록일자', '출원인명'])
kp.columns = ['appl_no', 'ay', 'pub', 'rg', 'names']; kp = kp[(kp.ay >= 2000) & (kp.ay <= 2019)]
vc = ((kp.pub.fillna('').astype(str).str.len() == 8) & (kp.pub.fillna('').astype(str) <= '20191231')) | ((kp.rg.fillna('').astype(str).str.len() == 8) & (kp.rg.fillna('').astype(str) <= '20191231'))
kp = kp[vc]; kp['nm'] = kp.names.fillna('').str.split('|'); kp = kp.explode('nm'); kp['n'] = kp.nm.map(norm)
al = pd.concat([fq[['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'nm'}), an[['applicantCd', 'applicantName']].rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})])
al['n'] = al.nm.map(norm); al = al[al.n.str.len() >= 2].drop_duplicates(['applicant_cd', 'n']); amb = al.groupby('n').applicant_cd.nunique(); al = al[al.n.map(amb) == 1]
n2c = dict(zip(al.n, al.applicant_cd)); kp['cd'] = kp.n.map(n2c); mm = kp.dropna(subset=['cd']).drop_duplicates(['appl_no', 'cd'])
mm['pre_alias'] = [(c_, n_) in pre for c_, n_ in zip(mm.cd, mm.n)]
pf = set(o.firm)
OUT['E_alias_postc'] = dict(feature_window_matches=int(len(mm)), share_via_postc_alias_only=float(1 - mm.pre_alias.mean()),
                            share_via_postc_alias_only_panel_firms=float(1 - mm[mm.cd.isin(pf)].pre_alias.mean()))
print('E 별칭', OUT['E_alias_postc'], flush=True)
OUT['secs'] = round(time.time() - t_start)
json.dump(OUT, open(V/'leak_t1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
print('저장 leak_t1.json', OUT['secs'])
