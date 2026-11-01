# verify_2 r1: C04-B 반박 재계산(libA 미사용, 독립 구현). 개발 원점 P1·잠금 위험집합 크기·P5-a 정밀값·'진짜 첫 특허' 여부(같은 10자리 사업자번호의 다른 CA 상호 선행 문서).
# 잠금창(2025-01~06) 레이블은 집계하지 않는다. 잠금 S0 스냅샷은 위험집합 크기만 센다.
import sys, re, json, calendar
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
EXT = B + r'\분석\모델경쟁_20260927\_shared\ext'
OUT = B + r'\분석\검증루프_20261012\C04\C04-T4\verify_2'
TC = {str(x) for x in range(81, 89)}
SUFX = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def nz(s):
    s = SUFX.sub('', str(s or '')).upper(); return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", '', s)
def addm(ymd, m):
    y, mo, d = int(ymd[:4]), int(ymd[4:6]) + m, int(ymd[6:]); y += (mo - 1) // 12; mo = (mo - 1) % 12 + 1
    return f'{y:04d}{mo:02d}{min(d, calendar.monthrange(y, mo)[1]):02d}'
ORIG = {'201512': '20151130', '201612': '20161130', '201712': '20171130', '201812': '20181130', '201912': '20191130',
        '202012': '20201231', '202112': '20211231', '202212': '20221231', '202312': '20231231', '202412': '20241231'}
# CA
rows = []
with open(B + r'\공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\r\n').split('¶')
        if len(p) >= 5: rows.append((p[0].strip(), p[1].strip(), re.sub(r'\D', '', p[4])))
ca = pd.DataFrame(rows, columns=['cd', 'nm', 'bz']); ca['n'] = ca.nm.map(nz); ca['hb'] = ca.bz.str.len() == 10
ca['b6'] = ca.bz.where(ca.hb).str[:6]; ca['tc'] = ca.hb & ca.bz.str[3:5].isin(TC)
c2 = ca[ca.n.str.len() >= 2]
st = c2.groupby('n').agg(N=('cd', 'size'), nb=('hb', 'sum'), u6=('b6', 'nunique'), b6=('b6', 'first'), tc=('tc', 'max'), u10=('bz', 'nunique'), bz=('bz', 'first'))
ok = st[(st.N == st.nb) & (st.u6 == 1) & st.tc]
n2b6 = ok.b6.to_dict(); n2bz = ok[ok.u10 == 1].bz.to_dict()
# docs
k = pd.read_parquet(B + r'\공모전데이터\작업데이터\kr_patents.parquet', columns=['출원번호', '공개일자', '등록일자', '출원인명', '세세분류코드'])
pub = k['공개일자'].fillna('').astype(str).str.strip(); rg = k['등록일자'].fillna('').astype(str).str.strip()
pub = pub.where(pub.str.len() == 8); rg = rg.where(rg.str.len() == 8)
vis = pd.concat([pub.fillna('99999999'), rg.fillna('99999999')], axis=1).min(axis=1).replace('99999999', np.nan)
D = pd.DataFrame({'a': k['출원번호'].values, 'pub': pub.values, 'vis': vis.values, 'n': k['출원인명'].map(nz).values, 'ks': k['세세분류코드'].astype(str).values})
D['b6'] = D.n.map(n2b6)
fv = D[D.vis.notna()].groupby('n').vis.min()
# 10자리 사업자번호 → 그 번호를 가진 모든 CA 상호의 첫 보이는 날짜(다른 출원인코드·표기로 낸 선행 문서 탐지)
ca['fv'] = ca.n.map(fv)
bz_fv = ca[ca.hb & ca.fv.notna()].groupby('bz').fv.min()
nt = pd.read_parquet(EXT + r'\ntis_project_raw.parquet', columns=['year', 'corp_rgst_no', 'research_agency'])
nt['bz'] = nt.corp_rgst_no.fillna('').str.replace(r'\D', '', regex=True)
nt = nt[(nt.bz.str.len() == 10) & nt.bz.str[3:5].isin(TC)]
nt['key'] = nt.bz.str[:6] + '|' + nt.research_agency.map(nz)
res = {}; POS = []
for s0, ref in ORIG.items():
    t0 = int(s0[:4]); end = addm(ref, 6)
    d = pd.read_parquet(EXT + rf'\firm_nps_wkpl_{s0}.parquet', columns=['name', 'biz6', 'stat', 'n_emp'])
    d = d[(d.stat.astype(str) == '1') & d.biz6.astype(str).str[3:5].isin(TC)]
    d = d.assign(key=d.biz6 + '|' + d.name.map(nz))
    E = d.groupby('key').n_emp.sum(); U = set(E[E >= 5].index)
    visk = {n2b6[n] + '|' + n for n in fv[fv <= ref].index if n in n2b6}
    ntk = set(nt[nt.year.between(t0 - 5, t0 - 3)].key)
    R = (U - visk) - ntk
    r = dict(E0ge5=len(U), risk=len(R))
    if s0 != '202412':
        w = D[D.pub.notna() & (D.pub > ref) & (D.pub <= end) & D.b6.notna()]
        w = w[w.n.map(fv) > ref]
        kk = w.b6 + '|' + w.n
        hit = w[kk.isin(R)].assign(key=kk[kk.isin(R)])
        pk = hit.groupby('key').agg(n=('n', 'first'), ks=('ks', lambda s: '|'.join(sorted(set(s))))).reset_index(); pk['s0'] = s0
        # 진짜 첫 특허인가: 그 상호의 10자리 번호(유일할 때)를 가진 다른 CA 상호가 기준일 전에 보이는 문서를 가졌는가
        bzs = pk.n.map(n2bz); pk['bz'] = bzs
        pk['prior_other_name'] = bzs.map(bz_fv).fillna('99999999') <= ref
        # 10자리 NTIS 잔여(t0-5..t0-3)
        nb10 = set(nt[nt.year.between(t0 - 5, t0 - 3)].bz)
        pk['ntis10_resid'] = bzs.isin(nb10)
        r.update(pos=len(pk), prior_other_name=int(pk.prior_other_name.sum()), ntis10_resid=int(pk.ntis10_resid.sum()), bz_unique_share=round(float(bzs.notna().mean()), 4))
        POS.append(pk)
    res[s0] = r; print(s0, r, flush=True)
P = pd.concat(POS, ignore_index=True)
res['P1_unique'] = int(P.drop_duplicates('key').shape[0]); res['P1_sum'] = int(len(P))
res['prior_other_name_total'] = int(P.prior_other_name.sum()); res['ntis10_resid_total'] = int(P.ntis10_resid.sum())
P.to_csv(OUT + r'\r1_pos.csv', index=False, encoding='utf-8-sig')
# 원작업 양성과 비교
o = pd.read_parquet(B + r'\분석\검증루프_20261012\C04\C04-T4\cache\B_dev_pos.parquet')
res['match_original'] = dict(mine=len(set(zip(P.s0, P.key))), orig=len(set(zip(o.s0, o.key))), inter=len(set(zip(P.s0, P.key)) & set(zip(o.s0, o.key))))
json.dump(res, open(OUT + r'\r1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False))
