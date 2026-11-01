# verify_2 r3: C04-B P2-c 운영 표지 M 재계산. s5는 first_vis를 '국내법인 표지·국내우선 문서'만으로 계산했다(s4·s6은 전체 문서).
# 여기서는 (i) s5와 같은 정의 재현 (ii) 전체 문서 first_vis(s4 정의와 일치)로 다시 계산. 개발 원점 2015–2021만, 관측 상한 2023-12-31.
import sys, re, json, calendar
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
OUT = B + r'\분석\검증루프_20261012\C04\C04-T4\verify_2'
CAP = '20231231'
SUFX = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def nz(s):
    s = SUFX.sub('', str(s or '')).upper(); return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", '', s)
def addm(ymd, m):
    y, mo, d = int(ymd[:4]), int(ymd[4:6]) + m, int(ymd[6:]); y += (mo - 1) // 12; mo = (mo - 1) % 12 + 1
    return f'{y:04d}{mo:02d}{min(d, calendar.monthrange(y, mo)[1]):02d}'
ORIG = {'201512': '20151130', '201612': '20161130', '201712': '20171130', '201812': '20181130', '201912': '20191130',
        '202012': '20201231', '202112': '20211231'}
names = set()
with open(B + r'\공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\r\n').split('¶')
        if len(p) >= 5: names.add(nz(p[1].strip()))
k = pd.read_parquet(B + r'\공모전데이터\작업데이터\kr_patents.parquet', columns=['출원번호', '공개일자', '등록일자', '출원인명', '우선권국가'])
pub = k['공개일자'].fillna('').astype(str).str.strip(); rg = k['등록일자'].fillna('').astype(str).str.strip()
pub = pub.where(pub.str.len() == 8); rg = rg.where(rg.str.len() == 8)
vis = pd.concat([pub.fillna('99999999'), rg.fillna('99999999')], axis=1).min(axis=1).replace('99999999', np.nan)
pr = k['우선권국가'].fillna('')
dom = (pr == '') | pr.str.split('|').map(lambda L: all(x == 'KR' for x in L if x))
kc = k['출원인명'].fillna('').str.contains(r'주식회사|\(주\)|㈜|유한회사') & dom
D = pd.DataFrame({'pub': pub, 'vis': vis, 'n': k['출원인명'].map(nz), 'kc': kc})
pos = pd.read_parquet(B + r'\분석\검증루프_20261012\C04\C04-T4\cache\B_dev_pos.parquet')
def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sp)
out = {}
for label, fvsrc in [('s5_def_kc_only', D[D.kc & D.vis.notna()]), ('all_docs_fv', D[D.vis.notna()])]:
    fv = fvsrc.groupby('n').vis.min(); Dk = D[D.kc & D.vis.notna()]
    G = []
    for s0, ref in ORIG.items():
        end = addm(ref, 6); fu = addm(end, 18)
        w = Dk[Dk.pub.notna() & (Dk.pub > ref) & (Dk.pub <= end)]
        w = w[w.n.map(fv) > ref]
        g = pd.DataFrame(index=pd.Index(w.n.unique(), name='n'))
        g = g[g.index.isin(names) & (g.index.str.len() >= 2)]
        later = set(Dk[(Dk.vis > end) & (Dk.vis <= fu)].n)
        g['M'] = g.index.isin(later).astype(int)
        g['det'] = g.index.isin(set(pos[pos.s0 == s0].key.str.split('|').str[1]))
        G.append(g)
    G = pd.concat(G)
    d1, d0 = G[G.M == 1].det.mean(), G[G.M == 0].det.mean()
    out[label] = dict(n=len(G), smd=round(smd(G[G.det].M, G[~G.det].M), 4), det_ratio=round(float(d1 / d0), 4), det_rate=round(float(G.det.mean()), 4))
    print(label, out[label], flush=True)
json.dump(out, open(OUT + r'\r3_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
