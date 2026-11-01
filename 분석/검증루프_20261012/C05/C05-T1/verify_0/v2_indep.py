# C05-T1 verify_0 / 재시도(v2): 반박 목적의 2차 독립 재계산. s2·v_indep 코드·캐시 미사용.
#  - 셀 집계: duckdb 에서 창(2019-21, 2022-24)별 count(distinct cc||appl_no) 직접 (연도별 합이 아니라 창 고유)
#  - 메인그룹: 자체 파서(문자열 분해; '/' 앞 숫자)
#  - 레이블: 파이썬 sorted 키(-y, -D, md5[:8]) 로 상위 k
#  - AUC: sklearn roc_auc_score, 부트스트랩: 공식 final_run_r3.cluster_boot 알고리즘(행 복제, 같은 난수열) 그대로 재구현
#  - 시나리오: S0, S1, S1r, S2, S2r, S2mg, S2mgr, S3, S3m 전부
import os, sys, re, json, hashlib, time, zipfile
import numpy as np, pandas as pd, duckdb
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
T = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
W = os.path.join(ROOT, r"검증루프_20261012\C05\C05-T1"); V = os.path.join(W, 'verify_0')
R3 = os.path.join(ROOT, r"모델경쟁_20260927\T3_기술셀부상예측\R3")
BR = os.path.join(R3, 'cache_pub_final', 'base_rows.parquet').replace('\\', '/')
OFF = json.load(open(os.path.join(R3, 'final_result_r3.json'), encoding='utf-8'))
out = {}
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
con.execute(f"create table d as select distinct ksic, cc, year, appl_no, ipc_raw from read_parquet('{BR}') where year between 2019 and 2024")
raw = con.sql("select ipc_raw, count(*) n from d group by all").df()
def mgp(s):
    if s is None: return None
    t = s.replace(' ', '').upper()
    if len(t) < 5 or not (t[0] in 'ABCDEFGH' and t[1:3].isdigit() and t[3].isalpha()): return None
    sc, rest = t[:4], t[4:]
    if '/' in rest:
        g = rest.split('/')[0]
        return f"{sc} {int(g)}" if g.isdigit() and 1 <= len(g) <= 4 else None
    if rest.isdigit() and 3 <= len(rest) <= 6:   # 슬래시 없는 형식(메인그룹+2자리 서브그룹)
        return f"{sc} {int(rest[:-2])}"
    return None
raw['mg'] = [mgp(x) for x in raw.ipc_raw]
out['parse'] = dict(n_raw=int(len(raw)), rows=int(raw.n.sum()), unparsed_rows=int(raw.n[raw.mg.isna()].sum()),
                    unparsed_sample=raw[raw.mg.isna()].sort_values('n', ascending=False).head(8)[['ipc_raw', 'n']].astype(str).values.tolist())
print('parse', out['parse'])
con.register('mp', raw[['ipc_raw', 'mg']].dropna())
cnt = con.sql("""select d.ksic||'|'||mp.mg cell, d.ksic, mp.mg,
   count(distinct case when year between 2019 and 2021 then cc||'#'||appl_no end) tot_W,
   count(distinct case when year between 2019 and 2021 and cc='KR' then appl_no end) kr_W,
   count(distinct case when year between 2022 and 2024 and cc='KR' then appl_no end) kr_fut
 from d join mp using(ipc_raw) group by all""").df()
E = cnt[cnt.tot_W >= 50].reset_index(drop=True)
def md5k(c): return int(hashlib.md5(c.encode()).hexdigest()[:8], 16)
def lab(df):
    y = np.log((df.kr_fut.values + 1.0) / (df.kr_W.values + 1.0)); D = (df.kr_fut - df.kr_W).values; k = int(round(0.1 * len(df)))
    order = sorted(range(len(df)), key=lambda i: (-y[i], -D[i], md5k(df.cell.values[i])))
    z = np.zeros(len(df), int); z[order[:k]] = 1; return z
E['yb'] = lab(E)
P = pd.read_parquet(os.path.join(R3, 'preds', 'FINAL_R3_t0_2021.parquet'))
M = P[['cell', 'mg', 'p', 'p_base', 'yb']].rename(columns={'yb': 'ybf', 'mg': 'mgf'}).merge(E, on='cell', how='outer', indicator=True)
out['gate'] = dict(merge=M._merge.value_counts().to_dict(), yb_mis=int((M.ybf != M.yb).sum()), mg_mis=int((M.mgf != M.mg).sum()))
M = M[M._merge == 'both'].drop(columns='_merge').reset_index(drop=True); M['p_base'] = M.p_base.astype(float)
print('gate', out['gate'])

def boot(cl, y, p, pb, nb=1000, seed=0):
    rng = np.random.default_rng(seed); ums = np.unique(cl)
    g = pd.Series(np.arange(len(cl))).groupby(cl).apply(lambda s: s.values).to_dict()
    sk, dif = [], []
    for _ in range(nb):
        s = rng.choice(ums, len(ums)); ii = np.concatenate([g[u] for u in s]); yy = y[ii]
        if yy.min() == yy.max(): continue
        a, a0 = roc_auc_score(yy, p[ii]), roc_auc_score(yy, pb[ii]); sk.append((a - a0) / (1 - a0)); dif.append(a - a0)
    return [float(v) for v in np.percentile(sk, [2.5, 97.5])], [float(v) for v in np.percentile(dif, [2.5, 97.5])], len(ums)
SK0 = OFF['summary']['mean_skill']; POFF = set(M.cell[M.ybf == 1])
def sc(d, y, name, ch=None):
    y = np.asarray(y); a, b = roc_auc_score(y, d.p), roc_auc_score(y, d.p_base); s = (a - b) / (1 - b)
    ci, cd, ncl = boot(np.asarray(d.mg_cl), y, d.p.values, d.p_base.values)
    if ch is None:
        Pn = set(d.cell[y == 1]); ch = len(POFF ^ Pn) / len(POFF)
    flip = ci[0] <= 0 or s <= 0 or a <= b
    v = '수정 필요' if flip else ('단서 필요' if (abs(s - SK0) >= 0.03 or ch >= 0.05) else '영향 없음')
    r = dict(scenario=name, n=len(d), pos=int(y.sum()), auc=a, auc_base=b, skill=s, d_skill=s - SK0, skill_lo=ci[0], skill_hi=ci[1],
             aucdiff_lo=cd[0], aucdiff_hi=cd[1], n_clusters=ncl, pos_change=ch, verdict=v)
    print({k: (round(x, 6) if isinstance(x, float) else x) for k, x in r.items()}, flush=True); return r
M['mg_cl'] = M.mg
rows = [sc(M, M.yb, 'S0')]
NEWSC = ['G06V', 'B64U', 'H10B', 'H10K', 'H10N', 'H10D', 'H10F', 'H10H']
# A2 원천은 대응표 원본(전 버전 csv)에서 다시 도출: to_sc 가 NEWSC, default=Y, from_sc≠to_sc
cy = pd.read_csv(os.path.join(W, 'wipo', 'wipo_concordance_all.csv'), dtype=str)
src = cy[(cy.default == 'Y') & cy.to_sc.isin(NEWSC) & (cy.from_sc != cy.to_sc)].groupby(['to_sc', 'from_sc']).size()
out['src_from_concordance'] = {f"{a}<-{b}": int(n) for (a, b), n in src.items()}
SRC = sorted(set(src.index.get_level_values(1)))
xm = cy[(cy.default == 'Y') & cy.version.isin(['20220101', '20230101', '20240101', '20250101']) & (cy.from_mg != cy.to_mg)]
MGMOV = set(xm.from_mg) | set(xm.to_mg)
xm0 = pd.read_csv(os.path.join(W, 'wipo', 'wipo_cross_mg_moves.csv'), dtype=str); xm0 = xm0[xm0.version.isin(['20220101', '20230101', '20240101', '20250101'])]
MG0 = set(xm0.from_mg) | set(xm0.to_mg)
out['A3_check'] = dict(n_mine=len(MGMOV), n_s2=len(MG0), sym_diff=sorted(MGMOV ^ MG0)[:30])
sub = M.mg.str[:4]
A1 = sub.str.startswith('H10'); A2 = sub.isin(set(NEWSC) | set(SRC)); A3 = M.mg.isin(MGMOV)
out['sets'] = dict(SRC=SRC, A1=int(A1.sum()), A2=int(A2.sum()), A3=int(A3.sum()), A1pos=int(M.yb[A1].sum()), A2pos=int(M.yb[A2].sum()), A3pos=int(M.yb[A3].sum()))
print(out['sets'], out['src_from_concordance'], out['A3_check'])
for f, nm in [(A1, 'S1'), (A2, 'S2'), (A3, 'S2mg')]:
    d = M[~f].reset_index(drop=True)
    rows.append(sc(d, d.yb, nm))
    yr = lab(d); Pn = set(d.cell[yr == 1]); ch = len(POFF ^ Pn) / len(POFF)
    rows.append(sc(d, yr, nm + 'r', ch))
# S3
isH = sub.eq('H01L') | A1
hd = con.sql("""select distinct d.ksic, d.cc, d.year, d.appl_no from d join mp using(ipc_raw) where substr(mp.mg,1,4)='H01L' or substr(mp.mg,1,3)='H10'""").df()
g = hd.groupby('ksic').apply(lambda x: pd.Series(dict(tot_W=len(x[x.year <= 2021].drop_duplicates(['cc', 'appl_no'])),
                                                       kr_W=int(((x.year <= 2021) & (x.cc == 'KR')).sum()), kr_fut=int(((x.year >= 2022) & (x.cc == 'KR')).sum()))),
                             include_groups=False).reset_index()
g = g[g.tot_W >= 50]
Hm = M[isH].copy()
for agg, nm in [('w', 'S3'), ('max', 'S3m')]:
    if agg == 'w':
        a = Hm.groupby('ksic').apply(lambda x: pd.Series(dict(p=np.average(x.p, weights=x.tot_W), p_base=np.average(x.p_base, weights=x.tot_W), offpos=x.ybf.sum())), include_groups=False)
    else:
        a = Hm.groupby('ksic').apply(lambda x: pd.Series(dict(p=x.p.max(), p_base=x.p_base.max(), offpos=x.ybf.sum())), include_groups=False)
    gg = g.merge(a.reset_index(), on='ksic'); gg['cell'] = gg.ksic + '|H01L+H10'; gg['mg_cl'] = 'H01L+H10'
    nd = pd.concat([M[~isH][['cell', 'mg_cl', 'p', 'p_base', 'kr_W', 'kr_fut', 'ybf']], gg[['cell', 'mg_cl', 'p', 'p_base', 'kr_W', 'kr_fut', 'offpos']]], ignore_index=True)
    y3 = lab(nd)
    nonH = nd.offpos.isna(); offn = set(nd.cell[nonH & (nd.ybf == 1)]); newn = set(nd.cell[nonH & (y3 == 1)])
    mpos = (~nonH) & (y3 == 1); match = int((mpos & (nd.offpos > 0)).sum())
    ch = (len(offn ^ newn) + (int(Hm.ybf.sum()) - match) + (int(mpos.sum()) - match)) / len(POFF)
    rows.append(sc(nd, y3, nm, ch))
    out[nm + '_merged'] = dict(n_merged=int((~nonH).sum()), merged_pos=int(mpos.sum()), match=match, offH_pos=int(Hm.ybf.sum()))
    print(out[nm + '_merged'])
out['scen'] = rows
# s2 결과와 비교
s2 = pd.read_csv(os.path.join(W, 'rescore_scenarios.csv'), encoding='utf-8-sig')
s2['key'] = s2.scenario.str.replace(' 공식 재현', '').str.replace('(보조)', '').str.replace('(민감도)', '').str.replace('S2mgr', 'S2mgr')
mine = pd.DataFrame(rows).set_index('scenario')
cmp = []
for _, r in s2.iterrows():
    k = r.key
    if k not in mine.index: cmp.append(dict(s=k, missing=True)); continue
    q = mine.loc[k]
    cmp.append(dict(s=k, n=int(q.n) - int(r.n), pos=int(q.pos) - int(r.pos), d_auc=float(q.auc - r.auc), d_skill=float(q.skill - r.skill),
                    d_lo=float(q.skill_lo - r.skill_lo), d_hi=float(q.skill_hi - r.skill_hi), d_ch=float(q.pos_change - r.pos_change), verdict_same=bool(q.verdict == r.verdict)))
out['cmp_vs_s2'] = cmp
print(pd.DataFrame(cmp).to_string())
# 신설 하위클래스 독립 확인: WIPO 표제 목록 zip 에서 하위클래스 집합 차이
def subs(fn):
    z = zipfile.ZipFile(os.path.join(W, 'download', fn)); S = set()
    for nm_ in z.namelist():
        txt = z.read(nm_).decode('utf-8', 'ignore')
        S |= set(re.findall(r'(?m)^([A-H]\d{2}[A-Z])\t', txt))
    return S
try:
    vs = ['20210101', '20220101', '20230101', '20240101', '20250101', '20260101']; ss = {v: subs(f'EN_ipc_title_list_{v}.zip') for v in vs}
    out['new_sub_by_version'] = {vs[i]: dict(n=len(ss[vs[i]]), new=sorted(ss[vs[i]] - ss[vs[i - 1]]), gone=sorted(ss[vs[i - 1]] - ss[vs[i]])) for i in range(1, len(vs))}
    out['new_sub_by_version']['20210101'] = dict(n=len(ss['20210101']))
except Exception as e:
    out['new_sub_by_version'] = f'err {e!r}'
print(json.dumps(out['new_sub_by_version'], ensure_ascii=False))
json.dump(out, open(os.path.join(V, 'v2_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print('sec', round(time.time() - T, 1))
