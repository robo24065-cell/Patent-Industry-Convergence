# -*- coding: utf-8 -*-
"""verify_1 v3: 권리목록 미조회 위험집합 기업의 공란 등록 기대 효과(동명 확률·등록 비율 보정) — 판정 외 참고 추정."""
import duckdb, pandas as pd, numpy as np, os, sys, io, re, json, sqlite3, time
from sklearn.metrics import roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = B + r"\분석\모델경쟁_20260927"; SH = MC + r"\_shared"
V = B + r"\분석\검증루프_20261012\C05\C05-T3\verify_1"
P = (SH + r"\ext\ntis_project_raw.parquet").replace(os.sep, "/")
M = (SH + r"\meta\*.parquet").replace(os.sep, "/")
o = {}
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def nrm(s):
    s = SUF.sub("", str(s or "")).upper(); return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
pr = pd.read_parquet(MC + r"\N6_정부RnD기업_첫특허전환\final_preds_r5.parquet"); pr['bz'] = pr.bz.astype(str).str.zfill(10)
ns = pd.read_parquet(SH + r"\ext\r3nts_applicant_status.parquet", columns=["applicantCd", "bizno", "aname"])
b2a = ns.groupby("bizno").applicantCd.apply(list).to_dict()
# y0 후보 (v1과 같은 정의) — 사업자번호 연결 코드 기준으로 다시
cand = set(pd.read_csv(V + r"\v1_y0_name_candidates.csv", dtype=str).bz)
con = sqlite3.connect('file:' + B + r"\수집파이프라인\business_rights.sqlite" + '?mode=ro', uri=True)
aq = pd.read_sql("select applicant_cd cd, status from applicant_query", con)
rl = pd.read_sql("select applicant_cd cd, rgst_no, appl_no from right_list where substr(rgst_no,1,2)='10'", con)
con.close()
aq_ok = set(aq[aq.status == 'OK'].cd)
rows = []
with open(B + r"\공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt", encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("¶")
        if len(p) >= 2:
            rows.append((p[0].strip(), p[1].strip()))
ca = pd.DataFrame(rows, columns=['cd', 'aname']); ca['n'] = ca.aname.map(nrm)
cnt = ca.groupby('n').cd.nunique().to_dict(); cdn = dict(zip(ca.cd, ca.n))
kp = pd.read_parquet(B + r"\공모전데이터\작업데이터\kr_patents.parquet", columns=["출원번호", "출원일", "등록번호", "출원인명"])
kp.columns = ['appl_no', 'ad', 'rg', 'names']
kp = kp[(kp.ad >= 20220101) & (kp.ad <= 20231231)].copy()
kp['blank'] = kp.rg.isna() | (kp.rg.astype(str).str.strip() == '')
o['window_apps'] = len(kp); o['window_blank_share'] = float(kp.blank.mean())
o['window_empty_applicant_name'] = int((kp.names.fillna('').str.strip() == '').sum())
# 등록 비율 추정: 조회(OK) 코드가 이름으로 유일하게 붙는 결과창 출원 중, 공란이면서 권리목록 등록 확인 비율
kp['nm'] = kp.names.fillna('').str.split('|'); ke = kp.explode('nm'); ke['n'] = ke.nm.map(nrm)
uniq_n2c = {n: c for c, n in cdn.items() if cnt.get(n) == 1}
ke['cd'] = ke.n.map(uniq_n2c)
q = ke[ke.cd.isin(aq_ok) & ke.blank]
rlpair = set(zip(rl.cd, rl.appl_no))
q = q.assign(reg=[(c, a) in rlpair for c, a in zip(q.cd, q.appl_no)])
o['reg_share_blank_window_queried'] = float(q.drop_duplicates(['appl_no', 'cd']).reg.mean())
o['reg_share_n'] = int(q.drop_duplicates(['appl_no', 'cd']).shape[0])
r_reg = o['reg_share_blank_window_queried']
# 미조회 후보 기업별 확률: P(적어도 한 출원이 내 것) × P(등록) — 동명 k개 균등 가정
f = []
for bz in cand:
    cds = [c for c in b2a.get(bz, [])]
    if not cds:
        continue
    if any(c in aq_ok for c in cds):
        continue
    names = {cdn.get(c) for c in cds if cdn.get(c)}
    apps = ke[ke.n.isin(names) & ke.blank]
    if len(apps) == 0:
        continue
    k = apps.n.map(cnt).values
    pr_none = np.prod(1 - r_reg / k)  # 각 출원이 내 것이면서 등록될 확률 r/k
    f.append((bz, 1 - pr_none, len(apps)))
fp = pd.DataFrame(f, columns=['bz', 'p', 'napps'])
o['unqueried_cand_with_blank_apps'] = len(fp); o['expected_flips'] = float(fp.p.sum())
o['napps_desc'] = fp.napps.describe().to_dict()
def sk(d, ycol):
    A = roc_auc_score(d[ycol], d.p_champ); Bv = roc_auc_score(d[ycol], d.p_m0); return (A - Bv) / (1 - Bv)
base = sk(pr, 'y')
# v1의 22곳(권리목록 확인) 포함
s6new = None
v1 = json.load(open(V + r"\v1_out.json", encoding='utf-8'))
rng = np.random.default_rng(1)
pm = fp.set_index('bz').p
res = []
for i in range(300):
    flip = set(pm.index[rng.random(len(pm)) < pm.values])
    d = pr.copy(); d['y3'] = (d.y | d.bz.isin(flip)).astype(int)
    res.append((len(flip), sk(d, 'y3') - base))
res = np.array(res)
o['sim_unqueried_only'] = {'flips_mean': float(res[:, 0].mean()), 'dskill_mean': float(res[:, 1].mean()),
                           'dskill_p5_p95': [float(np.percentile(res[:, 1], 5)), float(np.percentile(res[:, 1], 95))],
                           'share_abs_ge_0.005': float((np.abs(res[:, 1]) >= 0.005).mean())}
# 모형 점수 분포: 후보 기업 vs y0 전체 (후보가 저점수면 skill 하락)
yy = pr.set_index('bz')
o['cand_p_champ_pctile_mean'] = float(yy.p_champ.rank(pct=True).reindex(fp.bz).mean())
o['cand_p_m0_pctile_mean'] = float(yy.p_m0.rank(pct=True).reindex(fp.bz).mean())
o['pos_p_champ_pctile_mean'] = float(yy.p_champ.rank(pct=True)[yy.y == 1].mean())
o['base_skill'] = base
json.dump(o, open(V + r"\v3_out.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))
