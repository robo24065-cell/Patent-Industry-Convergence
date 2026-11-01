# C08-T4 verify_0 v1: P0(A, A_UB 제외, B, D, E)·P1(E*) 건수를 작업자 캐시·lib8 없이 sqlite SQL(julianday·date 함수)로 독립 재계산.
# 생존 판정도 다른 구현(최종 소멸일 vs 최종 회복일 비교). 2025-01-01 이후 사건은 SQL 조건으로 제외.
import sqlite3, sys, os, json, time
import pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
T = time.time()
DB = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite'
OUT = os.path.dirname(os.path.abspath(__file__))
con = sqlite3.connect('file:' + DB + '?mode=ro', uri=True)
q = lambda s: pd.read_sql(s, con)
P = q("select rgst_no, rgstDate rd, cndrtExptnDate ex from patent")
P['rd'] = P.rd.where(P.rd.astype(str).str.fullmatch(r'\d{8}')); P['ex'] = P.ex.where(P.ex.astype(str).str.fullmatch(r'\d{8}'))
S = q("select rgst_no, rgstCsDate d, case when rgstCsName='회복등록' then 1 else 0 end rec from right "
      "where rgstCsName in ('소멸등록','권리의 말소등록','말소등록','직권말소등록','회복등록') and rgstCsDate < '20250101' and length(rgstCsDate)=8")
# 지연 납부행: SQLite date()로 기한 계산(2/29 + N년 → 3/1로 넘어가는 차이 있음)
LATE = q("""select p.rgst_no, p.payDate d,
       julianday(substr(p.payDate,1,4)||'-'||substr(p.payDate,5,2)||'-'||substr(p.payDate,7,2))
     - julianday(date(substr(t.rgstDate,1,4)||'-'||substr(t.rgstDate,5,2)||'-'||substr(t.rgstDate,7,2), '+'||(cast(p.statAnnl as integer)-1)||' years')) lag,
       cast(p.statAnnl as integer) sa, cast(p.lastAnnl as integer) la
  from pay p join patent t on p.rgst_no=t.rgst_no
 where p.payDate < '20250101' and length(p.payDate)=8 and cast(p.statAnnl as integer) >= 4""")
LIC = q("select rgst_no, elsLcnsStartDate d from elsLcnsList where elsLcnsStartDate < '20250101' union all select rgst_no, noelsLcnsStartDate d from noelsLcnsList where noelsLcnsStartDate < '20250101'")
PART = q("select rgst_no, rgstCsDate d from right where rgstCsName='권리의 일부말소등록' and rgstCsReason like '%포기%' and rgstCsDate < '20250101'")
print('load', round(time.time() - T, 1), flush=True)


def alive(t0):
    s = S[S.d < t0]
    md = s[s.rec == 0].groupby('rgst_no').d.max(); mr = s[s.rec == 1].groupby('rgst_no').d.max()
    dead = md.index[~(mr.reindex(md.index).fillna('0') >= md)]
    ok = P.rd.notna() & (P.rd < t0) & (P.ex.isna() | (P.ex >= t0)) & ~P.rgst_no.isin(set(dead))
    return set(P.rgst_no[ok])


rows = []
for Y in list(range(2015, 2023)) + [2025]:
    t0, t1 = f'{Y}0101', f'{Y + 1}0101'
    A = alive(t0)
    r = {'origin': Y, 'risk': len(A)}
    if Y < 2025:
        w = LATE[(LATE.d >= t0) & (LATE.d < t1)]
        for nm, thr in [('E0', 0), ('E3', 3), ('E7', 7)]:
            r[nm] = len(set(w.rgst_no[w.lag > thr]) & A)
        r['D'] = len(set(w.rgst_no[(w.la - w.sa) >= 1]) & A)
        r['A'] = len(set(LIC.rgst_no[(LIC.d >= t0) & (LIC.d < t1)]) & A)
        r['B'] = len(set(PART.rgst_no[(PART.d >= t0) & (PART.d < t1)]) & A)
    rows.append(r); print(r, round(time.time() - T, 1), flush=True)
R = pd.DataFrame(rows)
dev = R[R.origin < 2025]; lr = int(R[R.origin == 2025].risk.iloc[0])
summ = {}
for k in ['E0', 'E3', 'E7', 'D', 'A', 'B']:
    rt = dev[k] / dev.risk
    summ[k] = {'pos_total': int(dev[k].sum()), 'rate_mean': float(rt.mean()), 'rate_2021_22': float(rt[dev.origin >= 2021].mean()),
               'lock_exp_main': lr * float(rt[dev.origin >= 2021].mean()), 'min_pos': int(dev[k].min())}
R.to_csv(os.path.join(OUT, 'v1_counts.csv'), index=False)
json.dump({'lock_risk': lr, 'summary': summ, 'sec': round(time.time() - T, 1)}, open(os.path.join(OUT, 'v1_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(summ, ensure_ascii=False, indent=1))
