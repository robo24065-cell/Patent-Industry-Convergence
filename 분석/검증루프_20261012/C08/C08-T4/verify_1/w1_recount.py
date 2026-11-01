# C08-T4 verify_1 w1: P0 후보 A, A_UB, B, C, D 독립 재계산(작업자 캐시·lib8·v1 미사용).
# 생존은 seq 순서상 t0 이전 마지막 상태사건(소멸류 vs 회복)으로 판정(세 번째 구현). C는 원 정의(dd>0, 특허별 마지막 소멸) + 교정판(같은 날 회복 포함, 소멸 에피소드 전부).
import sqlite3, sys, os, json, re, time
import pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
T = time.time()
DB = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite'
OUT = os.path.dirname(os.path.abspath(__file__))
con = sqlite3.connect('file:' + DB + '?mode=ro', uri=True)
q = lambda s: pd.read_sql(s, con)
CUT = '20250101'
P = q("select rgst_no, rgstDate rd, cndrtExptnDate ex from patent")
for c in ['rd', 'ex']:
    P[c] = P[c].where(P[c].fillna('').str.fullmatch(r'\d{8}'))
ST = q(f"select rgst_no, seq, rgstCsDate d, rgstCsName nm from right where rgstCsName in ('소멸등록','권리의 말소등록','말소등록','직권말소등록','회복등록') and rgstCsDate < '{CUT}'")
ST = ST[ST.d.fillna('').str.fullmatch(r'\d{8}')]
ST['dead'] = (ST.nm != '회복등록').astype(int)
LIC = q(f"select rgst_no, elsLcnsStartDate d, elsLcnsPsnNm nm from elsLcnsList where elsLcnsStartDate < '{CUT}' and length(elsLcnsStartDate)=8 "
        f"union all select rgst_no, noelsLcnsStartDate d, noelsLcnsPsnNm nm from noelsLcnsList where noelsLcnsStartDate < '{CUT}' and length(noelsLcnsStartDate)=8")
PART = q(f"select rgst_no, rgstCsDate d from right where rgstCsName='권리의 일부말소등록' and instr(coalesce(rgstCsReason,''),'포기')>0 and rgstCsDate < '{CUT}'")
PAY = q(f"select rgst_no, payDate d, statAnnl sa, lastAnnl la from pay where payDate < '{CUT}' and length(payDate)=8")
PAY['sa'] = pd.to_numeric(PAY.sa, errors='coerce'); PAY['la'] = pd.to_numeric(PAY.la, errors='coerce')
MULTI = PAY[(PAY.sa >= 4) & ((PAY.la - PAY.sa) >= 1)]
# A_UB: 실시권자 비개인 & 권리자·출원인 이름과 다름 (독자 구현: 공백·법인표기 제거 후 비교, 개인 = 한글 2-4자 순수 이름)
OWN = q("select rgst_no, ownerName n from owner union all select rgst_no, applicantName n from applicant")
strip = re.compile(r"주식회사|유한회사|유한책임회사|\(주\)|㈜|（주）|株式会社|CO\.?,?\s*LTD\.?|INC\.?|CORPORATION|[\s\(\)\[\]·\.,\-&'\"/]", re.I)
nz = lambda s: strip.sub('', str(s or '')).upper()
own_pairs = set(zip(OWN.rgst_no, OWN.n.map(nz)))
corp_kw = ['대학', '산학', '연구', '과학기술원', '재단', '공사', '공단', '진흥원', '대한민국', '시청', '도청', '회사', '유한', '코리아', '테크', '산업', '전자', '기업', '조합', '법인', '(주)', '㈜']
def indiv(s):
    s = str(s or '')
    if any(k in s for k in corp_kw): return False
    return bool(re.fullmatch(r'[가-힣]{2,4}', s.replace(' ', '')))
LIC['ub'] = [(not indiv(n)) and ((r, nz(n)) not in own_pairs) for r, n in zip(LIC.rgst_no, LIC.nm)]
print('load', round(time.time() - T, 1), 'lic', len(LIC), 'ub_share', round(LIC.ub.mean(), 4), flush=True)

def alive(t0):
    s = ST[ST.d < t0].sort_values(['rgst_no', 'seq'])
    last = s.groupby('rgst_no').dead.last()
    deadset = set(last.index[last == 1])
    ok = P.rd.notna() & (P.rd < t0) & (P.ex.isna() | (P.ex >= t0)) & ~P.rgst_no.isin(deadset)
    return set(P.rgst_no[ok])

EV = {'A': LIC, 'A_UB': LIC[LIC.ub], 'B': PART, 'D': MULTI}
rows = []
for Y in list(range(2015, 2023)) + [2025]:
    t0, t1 = f'{Y}0101', f'{Y+1}0101'
    A = alive(t0); r = {'origin': Y, 'risk': len(A)}
    if Y < 2025:
        for k, e in EV.items():
            r[k] = len(set(e.rgst_no[(e.d >= t0) & (e.d < t1)]) & A)
    rows.append(r); print(r, round(time.time() - T, 1), flush=True)
R = pd.DataFrame(rows); dev = R[R.origin < 2025]; lr = int(R[R.origin == 2025].risk.iloc[0])
summ = {}
for k in EV:
    rt = dev[k] / dev.risk; m2 = float(rt[dev.origin >= 2021].mean())
    summ[k] = {'pos_total': int(dev[k].sum()), 'rate_mean': float(rt.mean()), 'rate_min': float(rt.min()), 'rate_max': float(rt.max()), 'rate_2021_22': m2,
               'lock_exp_main': lr * m2, 'lock_exp_alldev': lr * float(rt.mean()), 'P0': bool(rt.mean() >= 0.003 and lr * m2 >= 1000)}
# B 2016 스파이크 제외 민감도
rtB = dev.B / dev.risk; summ['B']['rate_mean_ex2016'] = float(rtB[dev.origin != 2016].mean())

# C: 등록료불납 소멸 → 회복
EXT = q("select rgst_no, seq, rgstCsDate d from right where rgstCsName='소멸등록' and rgstCsReason='등록료불납' and length(rgstCsDate)=8")
REC = q("select rgst_no, seq, rgstCsDate dr from right where rgstCsName like '회복등록%' and length(rgstCsDate)=8")
EXT = EXT[EXT.d.str.fullmatch(r'\d{8}')].copy(); REC = REC[REC.dr.str.fullmatch(r'\d{8}')].copy()
EXT['dt'] = pd.to_datetime(EXT.d, format='%Y%m%d', errors='coerce'); REC['dtr'] = pd.to_datetime(REC.dr, format='%Y%m%d', errors='coerce')
x = EXT.merge(REC[['rgst_no', 'seq', 'dtr']].rename(columns={'seq': 'sr'}), on='rgst_no', how='left')
x['lag'] = (x.dtr - x.dt).dt.days
x['after'] = x.sr > x.seq   # seq상 소멸 뒤의 회복
def cohort(strict, all_ep):
    e = EXT.copy()
    if not all_ep:
        e = e.sort_values('d').drop_duplicates('rgst_no', keep='last')
    xx = e.merge(x[['rgst_no', 'seq', 'lag', 'after']], on=['rgst_no', 'seq'], how='left')
    hit = (xx.lag > 0) if strict else ((xx.lag >= 0) & xx.after.fillna(False))
    xx['hit'] = hit & (xx.lag <= 365)
    g = xx.groupby(['rgst_no', 'seq', 'd']).hit.max().reset_index(); g['y'] = g.d.str[:4].astype(int)
    c = g.groupby('y').agg(risk=('hit', 'size'), pos=('hit', 'sum')); c['rate'] = c.pos / c.risk
    return c
Cres = {}
for nm, strict, allep in [('C_orig(dd>0,last)', True, False), ('C_fix(dd>=0,seq after,all episodes)', False, True)]:
    c = cohort(strict, allep); cd = c.loc[2015:2022]
    lrC = 0.5 * float(c.loc[[2023, 2024], 'risk'].mean()); m2 = float(c.loc[[2021, 2022], 'rate'].mean())
    Cres[nm] = {'pos_total': int(cd.pos.sum()), 'rate_mean': float(cd.rate.mean()), 'rate_2021_22': m2, 'lock_risk_half': lrC, 'lock_exp_main': lrC * m2,
                'lock_exp_fullyear': 2 * lrC * m2, 'P0': bool(cd.rate.mean() >= 0.003 and lrC * m2 >= 1000),
                'by_year': {int(k): [int(v.risk), int(v.pos)] for k, v in c.loc[2013:2024].iterrows()}}
    print(nm, json.dumps(Cres[nm], ensure_ascii=False), flush=True)
lagall = x.lag[x.after.fillna(False) & (x.lag >= 0)]
Cres['lag_dist_after'] = {'n': int(len(lagall)), 'share_0': float((lagall == 0).mean()), 'q50': float(lagall.quantile(.5)), 'q99': float(lagall.quantile(.99))}
R.to_csv(os.path.join(OUT, 'w1_counts.csv'), index=False)
json.dump({'lock_risk': lr, 'summary': summ, 'C': Cres, 'sec': round(time.time() - T, 1)}, open(os.path.join(OUT, 'w1_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(summ, ensure_ascii=False, indent=1)); print(json.dumps(Cres['lag_dist_after']))
