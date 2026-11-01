# verify_2 r2: C04-B P5-a 독립 재구현(정밀값) + 귀무 순위 + 시드 민감도(판정 밖: 200회).
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
OUT = B + r'\분석\검증루프_20261012\C04\C04-T4\verify_2'
T0 = time.time()
P = pd.read_csv(OUT + r'\r1_pos.csv', dtype=str)
ORIG = {'201512': '20151130', '201612': '20161130', '201712': '20171130', '201812': '20181130', '201912': '20191130',
        '202012': '20201231', '202112': '20211231', '202212': '20221231', '202312': '20231231'}
import calendar
def addm(ymd, m):
    y, mo, d = int(ymd[:4]), int(ymd[4:6]) + m, int(ymd[6:]); y += (mo - 1) // 12; mo = (mo - 1) % 12 + 1
    return f'{y:04d}{mo:02d}{min(d, calendar.monthrange(y, mo)[1]):02d}'
import re
SUFX = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def nz(s):
    s = SUFX.sub('', str(s or '')).upper(); return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", '', s)
k = pd.read_parquet(B + r'\공모전데이터\작업데이터\kr_patents.parquet', columns=['출원번호', '공개일자', '출원인명', '세세분류코드'])
k['pub'] = k['공개일자'].fillna('').astype(str).str.strip(); k['n'] = k['출원인명'].map(nz); k['ks'] = k['세세분류코드'].astype(str)
# 양성 기업의 창 문서: 상호 n이 양성 key의 상호 부분과 같고 공개일이 창 안
pos_names = {}
for r in P.itertuples():
    pos_names.setdefault(r.s0, set()).add(r.key.split('|', 1)[1])
PD = []
for s0, ref in ORIG.items():
    end = addm(ref, 6)
    w = k[(k.pub.str.len() == 8) & (k.pub > ref) & (k.pub <= end) & k.n.isin(pos_names[s0])]
    PD.append(pd.DataFrame({'s0': s0, 'key': w.n.values, 'a': w['출원번호'].values, 'ks': w.ks.values}))
PD = pd.concat(PD, ignore_index=True)
lab = pd.read_parquet(B + r'\분석\검증루프_20261012\C02\C02-T4\cache\kr_publab_src.parquet')
lab['mg'] = lab.sg.str.split('/').str[0]; lab['sc'] = lab.sg.str[:4]
lab = lab.drop_duplicates('appl_no').set_index('appl_no')
def assign(t0, levels, nmin):
    sub = lab[lab.appl_year <= t0]; out = pd.Series(np.nan, index=lab.index, dtype=object)
    for l in levels:
        s = sub[sub[l].notna()]
        tab = s.groupby([l, 'k_free']).size().rename('c').reset_index()
        tot = tab.groupby(l).c.sum(); keep = set(tot[tot >= nmin].index)
        tab = tab[tab[l].isin(keep)].sort_values([l, 'c', 'k_free'], ascending=[True, False, True]).drop_duplicates(l)
        m = lab[l].map(tab.set_index(l).k_free)
        fill = out.isna() & m.notna(); out[fill] = m[fill]
    return out
def jac(x, col):
    Ea = set(zip(x.s0, x.key, x[col])) if col else None
    return Ea
res = {}
for A in ['TXT-FREE', 'FREE-IPC', 'H-sc-1']:
    I = U = 0; NI = {20: np.zeros(20), 200: np.zeros(200)}; NU = {20: np.zeros(20), 200: np.zeros(200)}
    for s0 in ORIG:
        t0 = int(s0[:4]); x = PD[PD.s0 == s0].reset_index(drop=True).copy()
        L = lab.k_free if A == 'TXT-FREE' else assign(t0, ['sg', 'mg', 'sc'], 3) if A == 'FREE-IPC' else assign(t0, ['sc'], 1)
        x['p'] = x.a.map(L).astype(object)
        Eo = set(zip(x.s0, x.key, x.ks)); xa = x.dropna(subset=['p']); Ea = set(zip(xa.s0, xa.key, xa.p))
        I += len(Ea & Eo); U += len(Ea | Eo)
        grp = x.groupby('ks').indices
        for nrep, base in [(20, 20261012), (200, 7770000)]:
            for rep in range(nrep):
                rng = np.random.default_rng(base + rep); q = x.p.values.copy()
                for kk, ix in grp.items(): q[ix] = q[ix][rng.permutation(len(ix))]
                xn = x.assign(p=q).dropna(subset=['p']); En = set(zip(xn.s0, xn.key, xn.p))
                NI[nrep][rep] += len(En & Eo); NU[nrep][rep] += len(En | Eo)
    J = I / U; r = dict(J_pub=J, I=I, U=U)
    for nrep in (20, 200):
        nj = NI[nrep] / NU[nrep]
        r[f'null{nrep}_q05'] = float(np.quantile(nj, 0.05)); r[f'null{nrep}_min'] = float(nj.min()); r[f'null{nrep}_mean'] = float(nj.mean())
        r[f'null{nrep}_n_le_Jpub'] = int((nj <= J).sum()); r[f'pass{nrep}'] = bool(J < np.quantile(nj, 0.05))
    res[A] = r; print(A, r, round(time.time() - T0), flush=True)
res['n_docs'] = len(PD)
json.dump(res, open(OUT + r'\r2_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
