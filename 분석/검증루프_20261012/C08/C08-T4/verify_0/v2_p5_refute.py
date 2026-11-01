# C08-T4 verify_0 v2: P5 재현·반박.
# (i) 제공 KSIC 산업 지연률 k_late1을 작업자 캐시 없이 다시 만든다(레이블 = v1과 같은 SQL 구현, KSIC = 제공 kr_registered 세세분류코드).
# (ii) 반박: Δ_A = S_prov − S_pub(A)의 출원인 군집 부트스트랩·KSIC 군집 부트스트랩(각 200회) — 특허 단위 짝 순열 귀무가 산업 내 상관을 무시해 좁을 수 있다는 의심.
import sqlite3, sys, os, json, time
import pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
T = time.time()
B_ = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
DB = B_ + r'\수집파이프라인\registry_full.sqlite'
CW = B_ + r'\분석\검증루프_20261012\C08\C08-T4\cache'
OUT = os.path.dirname(os.path.abspath(__file__))
con = sqlite3.connect('file:' + DB + '?mode=ro', uri=True)
q = lambda s: pd.read_sql(s, con)
P = q("select rgst_no, rgstDate rd, cndrtExptnDate ex from patent")
P['rd'] = P.rd.where(P.rd.astype(str).str.fullmatch(r'\d{8}')); P['ex'] = P.ex.where(P.ex.astype(str).str.fullmatch(r'\d{8}'))
kr = pd.read_parquet(B_ + r'\공모전데이터\작업데이터\kr_registered.parquet', columns=['등록번호', '세세분류코드'])
KS = dict(zip(kr['등록번호'].astype(str), kr['세세분류코드'].astype(str)))
S = q("select rgst_no, rgstCsDate d, case when rgstCsName='회복등록' then 1 else 0 end rec from right "
      "where rgstCsName in ('소멸등록','권리의 말소등록','말소등록','직권말소등록','회복등록') and rgstCsDate < '20250101' and length(rgstCsDate)=8")
LATE = q("""select p.rgst_no, p.payDate d,
       julianday(substr(p.payDate,1,4)||'-'||substr(p.payDate,5,2)||'-'||substr(p.payDate,7,2))
     - julianday(date(substr(t.rgstDate,1,4)||'-'||substr(t.rgstDate,5,2)||'-'||substr(t.rgstDate,7,2), '+'||(cast(p.statAnnl as integer)-1)||' years')) lag
  from pay p join patent t on p.rgst_no=t.rgst_no
 where p.payDate < '20250101' and length(p.payDate)=8 and cast(p.statAnnl as integer) >= 4""")


def alive(t0):
    s = S[S.d < t0]
    md = s[s.rec == 0].groupby('rgst_no').d.max(); mr = s[s.rec == 1].groupby('rgst_no').d.max()
    dead = md.index[~(mr.reindex(md.index).fillna('0') >= md)]
    ok = P.rd.notna() & (P.rd < t0) & (P.ex.isna() | (P.ex >= t0)) & ~P.rgst_no.isin(set(dead))
    return P.rgst_no[ok].values


lab = {}
for Y in range(2017, 2023):
    t0, t1 = f'{Y}0101', f'{Y + 1}0101'
    a = alive(t0); w = LATE[(LATE.d >= t0) & (LATE.d < t1) & (LATE.lag > 7)]
    df = pd.DataFrame({'rgst_no': a}); df['y'] = df.rgst_no.isin(set(w.rgst_no)).astype(int); df['k'] = df.rgst_no.map(KS)
    lab[Y] = df


def auc(x, y):
    o = np.argsort(x, kind='mergesort'); x, y = x[o], y[o]
    u, st = np.unique(x, return_index=True); g = np.repeat(np.arange(len(st)), np.diff(np.append(st, len(x))))
    p = np.bincount(g, weights=y); n = np.bincount(g, weights=1 - y); cn = np.cumsum(n) - n
    return float((p * (cn + 0.5 * n)).sum() / (p.sum() * n.sum()))


res = {'i_auc_prov_indep': {}}
for Y in range(2018, 2023):
    r = lab[Y - 1].groupby('k').y.mean()
    d = lab[Y]; x = d.k.map(r).fillna(lab[Y - 1].y.mean()).values
    res['i_auc_prov_indep'][Y] = round(auc(x, d.y.values.astype(float)), 5)
res['i_auc_prov_indep_mean'] = float(np.mean(list(res['i_auc_prov_indep'].values())))
w5 = pd.read_csv(os.path.join(CW, 's5_p5_result.csv'))
res['i_auc_prov_worker'] = json.loads(w5.auc_prov_by_origin.iloc[0])
print(res, round(time.time() - T, 1), flush=True)
# (ii) 군집 부트스트랩 반박(작업자 피처 파일의 공개 배정 값 사용)
D = pd.concat([pd.read_parquet(os.path.join(CW, f'feat_{Y}.parquet'), columns=['origin', 'y', 'cd', 'ksic'] + [f'k_late1_{a}' for a in ['prov', 'TXT', 'CONC', 'HSC1']]) for Y in range(2018, 2023)], ignore_index=True)
for unit in ['cd', 'ksic']:
    cl = pd.Index(D[unit].unique()); ci = cl.get_indexer(D[unit])
    rng = np.random.default_rng(777)
    W = rng.multinomial(len(cl), np.ones(len(cl)) / len(cl), size=200).astype(float)
    for A in ['TXT', 'CONC', 'HSC1']:
        dl = []
        prep = []
        for v, g in D.groupby('origin'):
            y = g.y.values.astype(float); gi = ci[g.index.values]
            U, inv = np.unique(np.concatenate([g.k_late1_prov.values, g[f'k_late1_{A}'].values]), return_inverse=True)
            prep.append((y, gi, inv[:len(g)], inv[len(g):], len(U)))
        for b in range(200):
            sp, sq = [], []
            for y, gi, cp, cq, K in prep:
                w = W[b, gi]
                for code, lst in ((cp, sp), (cq, sq)):
                    p = np.bincount(code, weights=w * y, minlength=K); n = np.bincount(code, weights=w * (1 - y), minlength=K); cn = np.cumsum(n) - n
                    lst.append(abs((p * (cn + 0.5 * n)).sum() / (p.sum() * n.sum()) - 0.5))
            dl.append(np.mean(sp) - np.mean(sq))
        dl = np.array(dl)
        res[f'ii_{unit}_{A}'] = {'delta_ci_lo': float(np.quantile(dl, 0.025)), 'delta_ci_hi': float(np.quantile(dl, 0.975)), 'share_le0': float((dl <= 0).mean())}
        print(unit, A, res[f'ii_{unit}_{A}'], round(time.time() - T, 1), flush=True)
json.dump(res, open(os.path.join(OUT, 'v2_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
