# T1 레드팀 재검증(PROTOCOL_R5 §3-1) — 최종시험(c=2019 → 진입 2020–23) 레이블·선행출원 제외·신규진입자 표시를
# T1 캐시(app_pub·ins_firm·firms·ins_applicant)를 쓰지 않고 원천에서 다시 만든다:
#   등록원부 registry_full.sqlite(patent·applicant, mode=ro), 권리목록 business_rights.sqlite(applicant_query, mode=ro), 제공 메타 _shared/meta(KR)
# 그리고 동결 예측(runs/FROZEN_final/o2019.parquet)·동결 기준선 점수로 AUC·skill을 순위합으로 다시 계산한다.
# 산출: VERIFY_R5/labels_c2019_raw.parquet, VERIFY_R5/repro_t1.json
import os, sys, re, json, sqlite3, time, pathlib
import numpy as np, pandas as pd, duckdb
from scipy.stats import rankdata
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
V = pathlib.Path(__file__).resolve().parent; T1 = V.parent; MC = T1.parent; ROOT = MC.parent.parent
RD = T1/'R3'/'runs'/'FROZEN_final'; FC = T1/'R3'/'cache_final'/'o2019'
C, H, END = 2019, 4, '20250822'
t0 = time.time()
ro = lambda p: sqlite3.connect('file:' + str(p).replace(os.sep, '/') + '?mode=ro', uri=True)
# 1) 법인 모집단(조회 성공 OK) — business_rights 원천
br = ro(ROOT/'수집파이프라인'/'business_rights.sqlite')
firms = pd.read_sql("select applicant_cd, applicant_name, status from applicant_query", br); br.close()
okf = firms[firms.status == 'OK']
# 2) 등록원부: 범위안 등록특허 + 원출원인 코드
rg = ro(ROOT/'수집파이프라인'/'registry_full.sqlite')
pat = pd.read_sql("select rgst_no, ksic, applDate, rgstDate from patent", rg)
app = pd.read_sql("select rgst_no, seq, applicantCd, applicantName from applicant", rg); rg.close()
pat['ay'] = pd.to_numeric(pat.applDate.str[:4], errors='coerce')
K = sorted(pat.ksic.astype(str).unique()); assert len(K) == 50; kidx = {k: i for i, k in enumerate(K)}
ins = app[app.applicantCd.isin(set(okf.applicant_cd))][['rgst_no', 'applicantCd']].drop_duplicates().merge(pat, on='rgst_no')
ins = ins[ins.ay >= 2000].copy(); ins['k'] = ins.ksic.astype(str).map(kidx)
print('ins rows', len(ins), round(time.time() - t0), flush=True)
# 3) 제공 메타 KR 출원 → 법인 별칭(조회명 + 등록원부 같은 코드 이름, 정규화, 모호명 제거) 매칭
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda s: STRIP.sub('', str(s)).lower()
al = pd.concat([okf[['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'nm'}),
                app[app.applicantCd.isin(set(okf.applicant_cd))][['applicantCd', 'applicantName']].rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})])
al['n'] = al.nm.map(norm); al = al[al.n.str.len() >= 2].drop_duplicates(['applicant_cd', 'n'])
amb = al.groupby('n').applicant_cd.nunique(); al = al[al.n.map(amb) == 1]; n2c = dict(zip(al.n, al.applicant_cd))
META = str(MC/'_shared'/'meta').replace(os.sep, '/') + '/*.parquet'
con = duckdb.connect(); con.execute("set memory_limit='2GB'; set threads=2")
k = con.sql(f"select appl_no, appl_year::int ay, pub_date, rgst_date, applicant, ksic from read_parquet('{META}') where cc='KR'").df()
k['nm'] = k.applicant.fillna('').str.split('|'); k = k.explode('nm'); k['applicant_cd'] = k.nm.map(norm).map(n2c)
ap = k.dropna(subset=['applicant_cd']).drop_duplicates(['appl_no', 'applicant_cd']); del k
ap = ap[ap.ay >= 2000].copy(); ap['k'] = ap.ksic.astype(str).map(kidx); ap = ap.dropna(subset=['k']); ap['k'] = ap.k.astype(int)
pdt = ap.pub_date.fillna('').astype(str); rdt = ap.rgst_date.fillna('').astype(str)
ap['vis_end'] = ((pdt.str.len() == 8) & (pdt <= END)) | ((rdt.str.len() == 8) & (rdt <= END))
print('matched apps', ap.appl_no.nunique(), 'firms', ap.applicant_cd.nunique(), round(time.time() - t0), flush=True)
# 4) 동결 평가 쌍에 레이블 부착
o = pd.read_parquet(RD/'o2019.parquet', columns=['firm', 'k', 'newc', 'px', 'pl', 'prior_any'])
key = pd.MultiIndex.from_arrays([o.firm.values, o.k.values])
mi = lambda d: pd.MultiIndex.from_frame(d[['applicant_cd', 'k']].astype({'k': int}).drop_duplicates())
reg = ins[(ins.ay > C) & (ins.ay <= C + H)].rename(columns={'applicantCd': 'applicant_cd'})
y_reg = key.isin(mi(reg)).astype(np.int8)
y_pub = key.isin(mi(ap[ap.vis_end & (ap.ay > C) & (ap.ay <= C + H)])).astype(np.int8)
prior = key.isin(mi(ap[ap.ay <= C]))
newc_raw = ~pd.Series(o.firm.values).isin(set(ins[ins.ay <= C].applicantCd)).values
risk_viol = key.isin(mi(ins[ins.ay <= C].rename(columns={'applicantCd': 'applicant_cd'})))   # 위험집합 정의 위반(목표 KSIC 등록 보유) 검사
y_u = np.maximum(y_reg, y_pub)
lab = pd.DataFrame(dict(firm=o.firm.values, k=o.k.values, y_reg=y_reg, y_pub=y_pub, y_u=y_u, prior_raw=prior, newc_raw=newc_raw))
lab.to_parquet(V/'labels_c2019_raw.parquet', index=False)
keep = ~prior; n = int(keep.sum())
def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
yk = y_u[keep]; nc = o.newc.values[keep].astype(bool)
s = 0.9 * rankdata(o.px.values[keep]) / n + 0.1 * rankdata(o.pl.values[keep]) / n
bs = pd.read_parquet(FC/'baseline_scores.parquet').loc[keep].reset_index(drop=True)
bm = {c: dict(auc_all=auc(yk, bs[c].values), auc_new=auc(yk[nc], bs[c].values[nc])) for c in bs.columns}
sa = max(v['auc_all'] for v in bm.values()); sn = max(v['auc_new'] for v in bm.values())
mine = dict(rows=n, pos=int(yk.sum()), pos_reg=int(y_reg[keep].sum()), new_rows=int(nc.sum()), new_pos=int(yk[nc].sum()), removed_prior_pairs=int(prior.sum()),
            auc_all=auc(yk, s), auc_new=auc(yk[nc], s[nc]), strongest_auc_all=sa, strongest_auc_new=sn,
            skill_all=(auc(yk, s) - sa) / (1 - sa), skill_new=(auc(yk[nc], s[nc]) - sn) / (1 - sn))
ref = json.load(open(T1/'R3'/'final_result_r3.json', encoding='utf-8'))
refd = dict(rows=ref['n']['rows'], pos=ref['n']['pos'], pos_reg=ref['n']['pos_reg'], new_rows=ref['n']['new_rows'], new_pos=ref['n']['new_pos'],
            removed_prior_pairs=ref['n']['removed_prior_pairs'], auc_all=ref['model']['auc_all'], auc_new=ref['model']['auc_new'],
            strongest_auc_all=ref['baseline_strongest']['auc_all'], strongest_auc_new=ref['baseline_strongest']['auc_new'], skill_all=ref['skill']['auc_all'], skill_new=ref['skill']['auc_new'])
cmp = {kk: dict(official=refd[kk], raw_rebuild=mine[kk], diff=float(mine[kk]) - float(refd[kk])) for kk in refd}
extra = dict(prior_flag_agree=float((prior == o.prior_any.values).mean()), prior_flag_mismatch=int((prior != o.prior_any.values).sum()),
             newc_flag_agree=float((newc_raw == o.newc.values.astype(bool)).mean()), risk_set_violations=int(risk_viol.sum()),
             panel_firms=int(o.firm.nunique()), panel_firms_in_OK=float(pd.Series(o.firm.unique()).isin(set(okf.applicant_cd)).mean()),
             n_ok_firms=int(len(okf)), n_rc001=int((firms.status != 'OK').sum()))
out = dict(compare=cmp, checks=extra, baselines_raw=bm, secs=round(time.time() - t0))
json.dump(out, open(V/'repro_t1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for kk, v in cmp.items(): print(f"{kk:22s} 공식 {v['official']}  원천재구성 {v['raw_rebuild']}  차 {v['diff']:.3g}")
print(extra, out['secs'])
