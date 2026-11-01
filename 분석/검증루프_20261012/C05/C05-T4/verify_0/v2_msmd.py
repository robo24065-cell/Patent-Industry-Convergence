# verify_0 v2: CA 사업자번호 경로 사후 의존 M SMD 재확인(C04 s5 정의, 10자리 차단 반영 양성). 개발 원점 2015–2021만(관측 상한 2023-12-31).
import sys, os, re, json, calendar
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
ROOT = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
WD = os.path.join(ROOT, '분석', '검증루프_20261012', 'C05', 'C05-T4')
SUFX = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def nz(s):
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", SUFX.sub("", str(s or "")).upper())
def addm(ymd, m):
    y, mo, d = ymd // 10000, ymd // 100 % 100, ymd % 100
    mo += m; y += (mo - 1) // 12; mo = (mo - 1) % 12 + 1
    return y * 10000 + mo * 100 + min(d, calendar.monthrange(y, mo)[1])
names = set()
with open(os.path.join(ROOT, '공모전데이터', '외부데이터', 'Corporate_20260720', 'CORP_APPLICANT.txt'), encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\r\n').split('¶')
        if len(p) >= 5:
            names.add(nz(p[1].strip()))
kp = pd.read_parquet(os.path.join(ROOT, '공모전데이터', '작업데이터', 'kr_patents.parquet'), columns=['공개일자', '등록일자', '출원인명', '우선권국가'])
pub = kp['공개일자'].fillna('').astype(str).str.strip(); rg = kp['등록일자'].fillna('').astype(str).str.strip()
pubi = pub.where(pub.str.len() == 8, '99999999').astype(np.int64); rgi = rg.where(rg.str.len() == 8, '99999999').astype(np.int64)
kp['v'] = np.minimum(pubi, rgi); kp['p'] = pubi
pr = kp['우선권국가'].fillna('')
dom = (pr == '') | pr.str.split('|').map(lambda L: all(x == 'KR' for x in L if x))
kc = kp['출원인명'].fillna('').str.contains(r'주식회사|\(주\)|㈜|유한회사') & dom
kp['n'] = kp['출원인명'].map(nz)
k = kp[kc & (kp.v < 99999999)]
fv = k.groupby('n').v.min()
pos = pd.read_parquet(os.path.join(WD, 'cache', 'dev_pos.parquet'))
REF = {'201512': 20151130, '201612': 20161130, '201712': 20171130, '201812': 20181130, '201912': 20191130, '202012': 20201231, '202112': 20211231}
G = []
for s0, ref in REF.items():
    end = addm(ref, 6); fu = addm(end, 18)
    assert fu <= 20231231
    w = k[(k.p < 99999999) & (k.p > ref) & (k.p <= end)]
    w = w[w.n.map(fv) > ref]
    nm = pd.Index(w.n.unique()); nm = nm[nm.isin(names) & (nm.str.len() >= 2)]
    later = set(k[(k.v > end) & (k.v <= fu)].n)
    det = set(pos[pos.s0 == s0].key.str.split('|').str[1])
    G.append(pd.DataFrame({'n': nm, 'M': nm.isin(later).astype(int), 'det': nm.isin(det), 's0': s0}))
G = pd.concat(G)
a, b = G[G.det].M.astype(float), G[~G.det].M.astype(float)
sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
res = dict(n=int(len(G)), n_det=int(G.det.sum()), smd=round(float((a.mean() - b.mean()) / sp), 4),
           det_rate_M1=round(float(G[G.M == 1].det.mean()), 4), det_rate_M0=round(float(G[G.M == 0].det.mean()), 4))
res['det_ratio'] = round(res['det_rate_M1'] / res['det_rate_M0'], 4)
res['pass_abs_smd_lt_0.2'] = abs(res['smd']) < 0.2
print(res)
json.dump(res, open(os.path.join(WD, 'verify_0', 'v2_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
