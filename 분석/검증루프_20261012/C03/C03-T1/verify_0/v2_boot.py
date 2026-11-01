# verify_0 v2: 독립 법인 군집 부트스트랩(시드 777, R=200) — Δskill_all·ΔAUC_all 구간, 복원판 skill 하한, 모형>최강 기준선 비율
# v1과 같은 레이블을 다시 만든다(원천: T1 공식 입력 캐시 + business_rights right_list). 쓰기: verify_0\ 만.
import sys, re, json, time, sqlite3, pathlib
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()
ROOT = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전")
R3 = ROOT/'분석/모델경쟁_20260927/T1_산업대기실/R3'; TC = ROOT/'분석/모델경쟁_20260927/T1_산업대기실/cache'
V = ROOT/'분석/검증루프_20261012/C03/C03-T1/verify_0'; DATA = ROOT/'공모전데이터/작업데이터'; END = '20250822'
def ro(p): return sqlite3.connect(pathlib.Path(p).as_uri() + '?mode=ro', uri=True)
o = pd.read_parquet(R3/'runs/FROZEN_final/o2019.parquet'); keep = ~o.prior_any.values
o = o[keep].reset_index(drop=True); bs = pd.read_parquet(R3/'cache_final/o2019/baseline_scores.parquet').loc[keep].reset_index(drop=True)
rk = lambda x: pd.Series(x).rank(pct=True).values
s = 0.9*rk(o.px.values) + 0.1*rk(o.pl.values); nc = o.newc.values.astype(bool)
K = sorted(pd.read_parquet(DATA/'kr_registered.parquet', columns=['세세분류코드'])['세세분류코드'].astype(str).unique()); kidx = {k: i for i, k in enumerate(K)}
pairs0 = list(zip(o.firm.values, o.k.values.astype(int)))
ins = pd.read_parquet(TC/'ins_firm.parquet', columns=['applicant_cd', 'ksic', 'ay']); e = ins[(ins.ay > 2019) & (ins.ay <= 2023)][['applicant_cd', 'ksic']].drop_duplicates()
e['k'] = e.ksic.astype(str).map(kidx); e = e.dropna(subset=['k']); E = set(zip(e.applicant_cd, e.k.astype(int))); del ins
a = pd.read_parquet(TC/'app_pub.parquet'); a['k'] = a.ksic.astype(str).map(kidx); a = a.dropna(subset=['k']); a['k'] = a.k.astype(int)
pdt = a.pub_date.fillna('').astype(str); rdt = a.rgst_date.fillna('').astype(str)
a = a[(((pdt.str.len() == 8) & (pdt <= END)) | ((rdt.str.len() == 8) & (rdt <= END))) & (a.ay > 2019) & (a.ay <= 2023)]; P = set(zip(a.applicant_cd, a.k)); del a
y0 = np.array([(p in E) or (p in P) for p in pairs0], np.int8)
kp = pd.read_parquet(DATA/'kr_patents.parquet', columns=['출원번호', '출원연도', '등록번호', '세세분류코드']); kp.columns = ['appl_no', 'ay', 'rgst_no', 'ksic']; kp['ay'] = kp.ay.astype(int)
kp = kp[(kp.ay >= 2020) & (kp.ay <= 2023) & (kp.rgst_no.fillna('').astype(str).str.len() <= 3)]; kp['k'] = kp.ksic.astype(str).map(kidx)
c = ro(ROOT/'수집파이프라인/business_rights.sqlite')
rl = pd.read_sql("select applicant_cd, appl_no, rgst_date, json_extract(raw,'$.pertDiv') pert from right_list where rgst_no like '10%' and substr(appl_no,3,4) >= '2020'", c)
ok = set(pd.read_sql("select applicant_cd from applicant_query where status='OK'", c).applicant_cd)
rl = rl[(rl.rgst_date <= END) & (rl.pert == '권리자란') & rl.applicant_cd.isin(ok)].merge(kp[['appl_no', 'k']], on='appl_no').dropna(subset=['k'])
B = set(zip(rl.applicant_cd, rl.k.astype(int)))   # R_any 판(상한)
y1 = np.maximum(y0, np.array([p in B for p in pairs0], np.int8))
print('pos', y0.sum(), y1.sum(), round(time.time()-t0), flush=True)

def prep(x):
    o_ = np.argsort(x, kind='stable'); xs = x[o_]; st = np.r_[0, np.nonzero(np.diff(xs))[0] + 1]; return o_, st
def wauc(pre, y, w):
    o_, st = pre; ww = w[o_]; yy = y[o_]
    wp = np.add.reduceat(ww*yy, st); wn = np.add.reduceat(ww*(1-yy), st)
    cn = np.cumsum(wn) - wn
    return float((wp*cn).sum() + 0.5*(wp*wn).sum()) / (wp.sum()*wn.sum())
cols = ['xgb_y_u', 'xgb_y', 'logit_stack_y_u']
pre = {'m': prep(s)}; pre.update({cc: prep(bs[cc].values) for cc in cols})
fi = pd.factorize(o.firm.values)[0]; nf = fi.max()+1
rng = np.random.RandomState(777); R = 200; rows = []
for r in range(R):
    cnt = np.bincount(rng.randint(0, nf, nf), minlength=nf).astype(float); w = cnt[fi]
    d = {}
    for lab, y in (('0', y0), ('1', y1)):
        m = wauc(pre['m'], y, w); b = max(wauc(pre[cc], y, w) for cc in cols)
        d['auc' + lab] = m; d['base' + lab] = b; d['skill' + lab] = (m-b)/(1-b)
    rows.append(d)
D = pd.DataFrame(rows); D['dauc'] = D.auc1 - D.auc0; D['dskill'] = D.skill1 - D.skill0
q = lambda v: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))]
res = dict(R=R, seed=777, label='R_any 상한판(권리자란, OK 법인)', pos0=int(y0.sum()), pos1=int(y1.sum()), skill0=q(D.skill0), skill1=q(D.skill1), dauc=q(D.dauc), dskill=q(D.dskill),
           model_gt_base_share0=float((D.auc0 > D.base0).mean()), model_gt_base_share1=float((D.auc1 > D.base1).mean()), sec=round(time.time()-t0))
json.dump(res, open(V/'v2_boot.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1); print(json.dumps(res, ensure_ascii=False))
