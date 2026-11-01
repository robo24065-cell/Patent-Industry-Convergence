# verify_0 v2: C04-B 양성 중 NTIS 차단(biz6+상호)을 빠져나간 수행 기업 — CA 10자리 사업자번호로 정확 대조.
import sys, re, json
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
EXT = B + r'\분석\모델경쟁_20260927\_shared\ext'
W = B + r'\분석\검증루프_20261012\C04\C04-T4'
rx_suf = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
rx_p = re.compile(r"[\s\(\)\[\]·\.,\-&'\"/]")
nz = lambda s: rx_p.sub('', rx_suf.sub('', str(s or '')).upper())
n2bz = {}
with open(B + r'\공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\r\n').split('¶')
        if len(p) >= 5:
            bz = re.sub(r'\D', '', p[4])
            if len(bz) == 10: n2bz.setdefault(nz(p[1]), set()).add(bz)
nt = pd.read_parquet(EXT + r'\ntis_project_raw.parquet', columns=['year', 'corp_rgst_no'])
nt['bz'] = nt.corp_rgst_no.fillna('').str.replace(r'\D', '', regex=True); nt = nt[nt.bz.str.len() == 10]
pos = pd.read_parquet(W + r'\cache\B_dev_pos.parquet')
pos['nm'] = pos.key.str.split('|').str[1]
res = {}
for s0, g in pos.groupby('s0'):
    t0 = int(s0[:4])
    a = set(nt[nt.year.between(t0 - 5, t0 - 3)].bz); b = set(nt[nt.year.between(t0 - 2, t0)].bz)
    bzs = g.nm.map(lambda n: n2bz.get(n, set()))
    res[s0] = dict(pos=len(g), resid_ntis_t0m5_m3_fullbiz=int(sum(1 for s in bzs if s & a)),
                   ntis_t0m2_t0_fullbiz=int(sum(1 for s in bzs if s & b)))
    print(s0, res[s0])
tot = {k: sum(v[k] for v in res.values()) for k in ['pos', 'resid_ntis_t0m5_m3_fullbiz', 'ntis_t0m2_t0_fullbiz']}
res['total'] = tot; print(tot)
json.dump(res, open(W + r'\verify_0\v2_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
