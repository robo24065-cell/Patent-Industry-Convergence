# C05-T1 s2: 재현 관문 → 개정 영향 셀 표시(A1·A2·A3) → 최종 원점 영향 → S1–S3 재채점(공식 cluster_boot 그대로) → 피처 창 점검(F1–F3) → 개발 원점.
# 재학습 없음. 동결 예측 p, p_base 만. 쓰기는 C05-T1 안에서만.
import os, sys, json, hashlib, time
import numpy as np, pandas as pd, duckdb
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
t_start = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
W = os.path.join(ROOT, r"검증루프_20261012\C05\C05-T1"); C = os.path.join(W, 'cache')
R3 = os.path.join(ROOT, r"모델경쟁_20260927\T3_기술셀부상예측\R3")
BR = os.path.join(R3, 'cache_pub_final', 'base_rows.parquet').replace('\\', '/')
OFF = json.load(open(os.path.join(R3, 'final_result_r3.json'), encoding='utf-8'))
SK0 = OFF['summary']['mean_skill']
NB = int(os.environ.get('C05_NBOOT', '1000'))

# ---------- 패널(문서 고유) ----------
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
pf = os.path.join(C, 'panel.parquet'); hf = os.path.join(C, 'h_docs.parquet')
if not os.path.exists(pf):
    con.execute(f"""copy (select ksic, mg, cc, year, count(distinct appl_no) n from (select ksic, mgn(ipc_raw) mg, cc, year, appl_no from read_parquet('{BR}'))
      where mg is not null group by all) to '{pf.replace(chr(92), '/')}' (format parquet)""")
if not os.path.exists(hf):   # S3 통합 셀용: H01L·H10 메인그룹 문서(ksic, cc, year, appl_no, mg)
    con.execute(f"""copy (select distinct ksic, cc, year, appl_no, mg from (select ksic, mgn(ipc_raw) mg, cc, year, appl_no from read_parquet('{BR}'))
      where mg like 'H01L %' or mg like 'H10% %') to '{hf.replace(chr(92), '/')}' (format parquet)""")
pan = pd.read_parquet(pf); pan['cell'] = pan.ksic + '|' + pan.mg
Y0 = 2000; NY = 25
def arrays(p):
    cells = np.array(sorted(p.cell.unique())); ci = np.searchsorted(cells, p.cell.values)
    KR = np.zeros((len(cells), NY)); TOT = np.zeros((len(cells), NY))
    np.add.at(TOT, (ci, p.year.values - Y0), p.n.values)
    k_ = p.cc.values == 'KR'; np.add.at(KR, (ci[k_], p.year.values[k_] - Y0), p.n.values[k_])
    return cells, KR, TOT
cells, KR, TOT = arrays(pan)
S = lambda A, a, b: A[:, a - Y0:b - Y0 + 1].sum(1)
hk = lambda cs: np.array([int(hashlib.md5(s.encode()).hexdigest()[:8], 16) for s in cs], dtype=np.int64)
def topk_label(y, tie, cellv, k):
    o = np.lexsort((hk(cellv), -tie, -y)); yb = np.zeros(len(y), int); yb[o[:k]] = 1; return yb
def origin_frame(cells, KR, TOT, t0):
    d = pd.DataFrame(dict(cell=cells, tot_W=S(TOT, t0 - 2, t0), kr_W=S(KR, t0 - 2, t0), kr_fut=S(KR, t0 + 1, t0 + 3)))
    d = d[d.tot_W >= 50].reset_index(drop=True); d['t0'] = t0
    d['y'] = np.log((d.kr_fut + 1) / (d.kr_W + 1)); k = int(round(0.10 * len(d)))
    d['yb'] = topk_label(d.y.values, (d.kr_fut - d.kr_W).values, d.cell.values, k)
    return d

# ---------- 공식 cluster_boot (final_run_r3.py 110–125행 그대로; 단일 블록은 같은 난수열·같은 재표집을 벡터화한 판으로 계산) ----------
def cluster_boot(blocks, n_boot, seed=0):
    if len(blocks) == 1: return cluster_boot_fast(blocks[0], n_boot, seed)
    return cluster_boot_ref(blocks, n_boot, seed)
def cluster_boot_fast(b, n_boot, seed=0):
    rng = np.random.default_rng(seed); ums = np.unique(b[0])
    pos_ = np.searchsorted(ums, b[0]); order = np.argsort(pos_, kind='stable'); cnt = np.bincount(pos_, minlength=len(ums)); st = np.r_[0, np.cumsum(cnt)[:-1]]
    out = []
    for _ in range(n_boot):
        s = rng.choice(ums, len(ums)); q = np.searchsorted(ums, s); c = cnt[q]
        ii = order[np.repeat(st[q] - np.r_[0, np.cumsum(c)[:-1]], c) + np.arange(c.sum())]
        y = b[1][ii]
        if y.min() == y.max(): continue
        a, a0 = roc_auc_score(y, b[2][ii]), roc_auc_score(y, b[3][ii])
        out.append((a, a0, (a - a0) / (1 - a0), a - a0, (a - a0) / (1 - a0)))
    o = np.array(out); qf = lambda j: [float(v) for v in np.percentile(o[:, j], [2.5, 97.5])]
    return dict(auc=qf(0), auc_baseline=qf(1), skill=qf(2), auc_diff=qf(3), n_boot=n_boot, n_clusters=int(len(ums)))
def cluster_boot_ref(blocks, n_boot, seed=0):
    rng = np.random.default_rng(seed); ums = np.unique(np.concatenate([b[0] for b in blocks]))
    gi = [pd.Series(np.arange(len(b[0]))).groupby(b[0]).apply(lambda s: s.values).to_dict() for b in blocks]
    out = []
    for _ in range(n_boot):
        s = rng.choice(ums, len(ums)); am, ab = [], []
        for b, g in zip(blocks, gi):
            ii = np.concatenate([g[u] for u in s if u in g]); y = b[1][ii]
            if y.min() == y.max(): continue
            am.append(roc_auc_score(y, b[2][ii])); ab.append(roc_auc_score(y, b[3][ii]))
        a, a0 = np.mean(am), np.mean(ab)
        out.append((a, a0, (a - a0) / (1 - a0), a - a0, np.mean([(x - y0) / (1 - y0) for x, y0 in zip(am, ab)])))
    o = np.array(out); q = lambda j: [float(v) for v in np.percentile(o[:, j], [2.5, 97.5])]
    return dict(auc=q(0), auc_baseline=q(1), skill=q(2), auc_diff=q(3), n_boot=n_boot, n_clusters=int(len(ums)))

def auc(y, p): return float(roc_auc_score(y, p))
def topq(y, s, k): return int(np.asarray(y)[np.argsort(-np.asarray(s, float), kind='stable')[:k]].sum())

# ---------- 재현 관문 ----------
P = pd.read_parquet(os.path.join(R3, 'preds', 'FINAL_R3_t0_2021.parquet'))
te = origin_frame(cells, KR, TOT, 2021)
m = P.merge(te, on='cell', how='outer', indicator=True, suffixes=('', '_r'))
gate = dict(merge={k: int(v) for k, v in m._merge.value_counts().items()})
m = m[m._merge == 'both'].drop(columns='_merge').reset_index(drop=True)
gate.update(n=int(len(m)), pos=int(m.yb_r.sum()), yb_mismatch=int((m.yb != m.yb_r).sum()), y_maxdiff=float(np.abs(m.y - m.y_r).max()))
y = m.yb_r.values; p = m.p.values; pb = m.p_base.values.astype(float)
gate.update(auc=auc(y, p), auc_base=auc(y, pb)); gate['skill'] = (gate['auc'] - gate['auc_base']) / (1 - gate['auc_base'])
ci0 = cluster_boot([(m.mg.values, y, p, pb)], NB)
gate['ci_skill'] = ci0['skill']; gate['ci_official'] = OFF['ci95']['skill']
gate['pass'] = bool(gate['merge'].get('both') == 5041 and gate['yb_mismatch'] == 0 and gate['y_maxdiff'] < 1e-5
                    and abs(gate['auc'] - OFF['summary']['auc']) < 1e-12 and abs(gate['skill'] - SK0) < 1e-9
                    and (NB != 1000 or max(abs(a - b) for a, b in zip(ci0['skill'], OFF['ci95']['skill'])) < 5e-5))
print('GATE', json.dumps(gate, ensure_ascii=False)); assert gate['pass']
m['y'] = m.y_r; m['yb'] = m.yb_r; m['kr_W'] = m.kr_W_r
m = m[['cell', 'ksic', 'mg', 't0', 'kr_W', 'kr_fut', 'tot_W', 'y', 'yb', 'p', 'p_base']].copy()
m['p_base'] = m.p_base.astype(float); m['D'] = m.kr_fut - m.kr_W

# ---------- 개정 영향 셀 표시 ----------
NEWSC = {'G06V', 'B64U', 'H10B', 'H10K', 'H10N', 'H10D', 'H10F', 'H10H'}
nw = pd.read_csv(os.path.join(W, 'wipo', 'wipo_newsub_sources.csv'), dtype=str)
SRC = set(nw.from_sc)            # 대응표 default=Y 원천: G06K, H01L
xm = pd.read_csv(os.path.join(W, 'wipo', 'wipo_cross_mg_moves.csv'), dtype=str)
xm = xm[xm.version.isin(['20220101', '20230101', '20240101', '20250101'])]
MGMOV = set(xm.from_mg) | set(xm.to_mg)
def flags(d):
    sc = d.mg.str[:4]
    d['A1'] = sc.str.startswith('H10'); d['A2'] = sc.isin(NEWSC | SRC); d['A3'] = d.mg.isin(MGMOV)
    d['new_sub'] = sc.isin(NEWSC); d['src_sub'] = sc.isin(SRC)
    return d
m = flags(m)
m['rank_p'] = m.p.rank(ascending=False, method='first').astype(int); m['pct_p'] = m.p.rank(pct=True)
m['top10'] = m.rank_p <= 504; m['top100'] = m.rank_p <= 100
res = dict(gate=gate, sets=dict(NEWSC=sorted(NEWSC), SRC=sorted(SRC), n_mg_moved=len(MGMOV)))

# ---------- 최종 원점 영향 ----------
pos = m[m.yb == 1]; netD = pos.D.sum()
imp = {}
for f in ['A1', 'A2', 'A3', 'new_sub', 'src_sub']:
    imp[f] = dict(eligible=int(m[f].sum()), pos=int(pos[f].sum()), pos_share=float(pos[f].mean()), pos_rate=float(m[m[f]].yb.mean()) if m[f].any() else None,
                  netD_share=float(pos[pos[f]].D.sum() / netD), top10_cells=int((m.top10 & m[f]).sum()), top10_hits=int((m.top10 & m[f] & (m.yb == 1)).sum()),
                  top100_cells=int((m.top100 & m[f]).sum()), top100_hits=int((m.top100 & m[f] & (m.yb == 1)).sum()),
                  median_pct_p_pos=float(m[m[f] & (m.yb == 1)].pct_p.median()) if (m[f] & (m.yb == 1)).any() else None)
imp['all'] = dict(eligible=len(m), pos=int(len(pos)), netD=float(netD), top10_hits=int((m.top10 & (m.yb == 1)).sum()), top100_hits=int((m.top100 & (m.yb == 1)).sum()),
                  median_pct_p_pos=float(pos.pct_p.median()))
res['final_impact'] = imp
print(json.dumps(imp, ensure_ascii=False, indent=1))

# ---------- 시나리오 ----------
P_off = set(m.cell[m.yb == 1])
def score_block(d, cl, yv, name, ref_pos_cells=None, removed_pos=0):
    a, ab = auc(yv, d.p.values), auc(yv, d.p_base.values)
    sk = (a - ab) / (1 - ab); ci = cluster_boot([(cl, yv, d.p.values, d.p_base.values)], NB)
    Pn = set(d.cell[yv == 1])
    if ref_pos_cells is None: ref_pos_cells = P_off
    ch = (len(ref_pos_cells - Pn) + len(Pn - ref_pos_cells) + removed_pos) / len(P_off)
    k = int(round(0.1 * len(d)))
    out = dict(scenario=name, n=int(len(d)), pos=int(yv.sum()), auc=a, auc_base=ab, skill=sk, d_skill=sk - SK0,
               skill_lo=ci['skill'][0], skill_hi=ci['skill'][1], aucdiff_lo=ci['auc_diff'][0], aucdiff_hi=ci['auc_diff'][1], n_clusters=ci['n_clusters'],
               pos_change=ch, top10pct_hits=topq(yv, d.p.values, k), top10pct_k=k, top100_hits=topq(yv, d.p.values, 100),
               top10pct_hits_base=topq(yv, d.p_base.values, k))
    flip = (out['skill_lo'] <= 0) or (sk <= 0) or (a <= ab)
    out['verdict'] = '수정 필요' if flip else ('단서 필요' if (abs(out['d_skill']) >= 0.03 or ch >= 0.05) else '영향 없음')
    print(name, {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in out.items()}, flush=True)
    return out

rows = []
rows.append(score_block(m, m.mg.values, m.yb.values, 'S0 공식 재현'))
def excl(flag, name):
    d = m[~m[flag]].reset_index(drop=True)
    removed = len(P_off & set(m.cell[m[flag]]))
    r1 = score_block(d, d.mg.values, d.yb.values, name, ref_pos_cells=P_off - set(m.cell[m[flag]]), removed_pos=removed)
    k = int(round(0.1 * len(d))); yr = topk_label(d.y.values, d.D.values, d.cell.values, k)
    r2 = score_block(d, d.mg.values, yr, name + 'r', ref_pos_cells=P_off - set(m.cell[m[flag]]), removed_pos=removed)
    return [r1, r2]
rows += excl('A1', 'S1'); rows += excl('A2', 'S2')
aux = excl('A3', 'S2mg')
for r in aux: r['scenario'] = r['scenario'] + '(보조)'
rows += aux

# S3: KSIC × (H01L+H10) 통합 셀
hd = pd.read_parquet(hf)
def merged_counts(t0):
    h = hd[(hd.year >= t0 - 2) & (hd.year <= t0 + 3)].drop_duplicates(['ksic', 'cc', 'year', 'appl_no'])
    g = pd.DataFrame(dict(tot_W=h[h.year <= t0].groupby('ksic').size(), kr_W=h[(h.year <= t0) & (h.cc == 'KR')].groupby('ksic').size(),
                          kr_fut=h[(h.year > t0) & (h.cc == 'KR')].groupby('ksic').size())).fillna(0)
    g.index.name = 'ksic'; return g.reset_index()
def s3_frame(d_all, t0, agg):
    isH = d_all.mg.str.startswith('H01L') | d_all.mg.str.startswith('H10')
    keep = d_all[~isH].copy(); hcell = d_all[isH].copy()
    g = merged_counts(t0); g = g[g.tot_W >= 50].copy()
    def ag(x, col):
        if agg == 'wmean': return float(np.average(x[col], weights=x.tot_W)) if x.tot_W.sum() > 0 else float(x[col].mean())
        return float(x[col].max())
    sc = hcell.groupby('ksic').apply(lambda x: pd.Series(dict(p=ag(x, 'p'), p_base=ag(x, 'p_base'), n_members=len(x), off_pos=int(x.yb.sum()),
                                                                 members=';'.join(x.cell)))).reset_index()
    g = g.merge(sc, on='ksic', how='inner')         # 동결 점수가 있는 통합 셀만(구성 셀 중 적격 셀 1개 이상)
    g['cell'] = g.ksic + '|H01L+H10'; g['mg'] = 'H01L+H10'; g['y'] = np.log((g.kr_fut + 1) / (g.kr_W + 1)); g['D'] = g.kr_fut - g.kr_W
    nd = pd.concat([keep[['cell', 'ksic', 'mg', 'kr_W', 'kr_fut', 'tot_W', 'y', 'D', 'p', 'p_base', 'yb']].rename(columns={'yb': 'yb_off'}),
                    g[['cell', 'ksic', 'mg', 'kr_W', 'kr_fut', 'tot_W', 'y', 'D', 'p', 'p_base', 'off_pos', 'n_members', 'members']]], ignore_index=True)
    k = int(round(0.1 * len(nd))); nd['yb'] = topk_label(nd.y.values, nd.D.values, nd.cell.values, k)
    return nd, hcell, g
s3_info = {}
for agg, nm in [('wmean', 'S3'), ('max', 'S3m(민감도)')]:
    nd, hcell, g = s3_frame(m, 2021, agg)
    # 양성 집합 변화: 비H 셀은 셀 단위 비교, 통합 셀은 '구성 셀 중 공식 양성 있음'과 비교. 통합으로 사라진 공식 양성 = 공식 양성 H 셀 수 − 통합 양성 대응 수
    nonH = nd[nd.mg != 'H01L+H10']; Hs = nd[nd.mg == 'H01L+H10']
    off_nonH = set(nonH.cell[nonH.yb_off == 1]); new_nonH = set(nonH.cell[nonH.yb == 1])
    offH_pos = int(hcell.yb.sum()); newH_pos = int(Hs.yb.sum()); match_H = int(((Hs.yb == 1) & (Hs.off_pos > 0)).sum())
    ch_n = len(off_nonH - new_nonH) + len(new_nonH - off_nonH) + (offH_pos - match_H) + (newH_pos - match_H)
    a, ab = auc(nd.yb.values, nd.p.values), auc(nd.yb.values, nd.p_base.values)
    r = score_block(nd, nd.mg.values, nd.yb.values, nm, ref_pos_cells=set(nd.cell[nd.yb == 1]), removed_pos=0)
    r['pos_change'] = ch_n / len(P_off)
    flip = (r['skill_lo'] <= 0) or (r['skill'] <= 0) or (r['auc'] <= r['auc_base'])
    r['verdict'] = '수정 필요' if flip else ('단서 필요' if (abs(r['d_skill']) >= 0.03 or r['pos_change'] >= 0.05) else '영향 없음')
    rows.append(r)
    s3_info[nm] = dict(n_merged_cells=int(len(Hs)), n_H_member_cells=int(len(hcell)), off_pos_in_H=offH_pos, merged_pos=newH_pos, merged_pos_matching_off=match_H,
                       nonH_pos_lost=len(off_nonH - new_nonH), nonH_pos_gained=len(new_nonH - off_nonH),
                       merged_cells=Hs[['cell', 'kr_W', 'kr_fut', 'tot_W', 'y', 'yb', 'p', 'off_pos', 'n_members']].sort_values('p', ascending=False).head(20).to_dict('records'))
    if nm == 'S3':
        Hs.drop(columns=['yb_off']).to_csv(os.path.join(W, 'cache', 's3_merged_cells.csv'), index=False, encoding='utf-8-sig')
res['s3_info'] = s3_info
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'merged_cells'} for k, v in s3_info.items()}, ensure_ascii=False))

# ---------- 피처 창 점검 F1–F3 ----------
# F1: A1 셀 과거창 국가 구성
pc = pan[(pan.year >= 2019) & (pan.year <= 2021)].groupby(['cell', 'cc']).n.sum().unstack('cc', fill_value=0)
a1 = m[m.A1].merge(pc, left_on='cell', right_index=True, how='left').fillna(0)
for c_ in ['KR', 'CN', 'US', 'JP', 'EP']:
    if c_ not in a1: a1[c_] = 0
f1 = dict(n=int(len(a1)), past_share_by_cc={c_: float(a1[c_].sum() / a1.tot_W.sum()) for c_ in ['KR', 'CN', 'US', 'JP', 'EP']},
          kr0_cells=int((a1.kr_W == 0).sum()), kr0_pos=int(((a1.kr_W == 0) & (a1.yb == 1)).sum()),
          cn_ge50pct_cells=int((a1.CN / a1.tot_W >= 0.5).sum()), us_jp_past_docs=float(a1.US.sum() + a1.JP.sum()))
newc = m[m.new_sub]
f1['cannot_exist_at_t0'] = dict(cells=int(len(newc)), pos=int(newc.yb.sum()), top10_cells=int(newc.top10.sum()), top10_hits=int((newc.top10 & (newc.yb == 1)).sum()),
                                top100_cells=int(newc.top100.sum()), top100_hits=int((newc.top100 & (newc.yb == 1)).sum()),
                                by_sub=newc.groupby(newc.mg.str[:4]).agg(cells=('cell', 'size'), pos=('yb', 'sum'), top10=('top10', 'sum'), top100=('top100', 'sum')).to_dict('index'))
a1[['cell', 'kr_W', 'kr_fut', 'tot_W', 'KR', 'CN', 'US', 'JP', 'EP', 'yb', 'p', 'rank_p']].sort_values('rank_p').to_csv(os.path.join(W, 'cache', 'f1_h10_cells.csv'), index=False, encoding='utf-8-sig')
# F2: 프로필 매칭
m2 = m.merge(pc, left_on='cell', right_index=True, how='left').fillna({'KR': 0})
m2['forsh'] = 1 - m2.KR / m2.tot_W
m2['b_kr'] = pd.cut(m2.kr_W, [-1, 0, 1, 4, 19, 1e9], labels=['0', '1', '2-4', '5-19', '20+']).astype(str)
m2['b_tot'] = pd.qcut(m2.tot_W, 4, labels=False, duplicates='drop').astype(str)
m2['b_fx'] = pd.cut(m2.forsh, [-0.01, 0.5, 0.9, 1.01], labels=['<.5', '.5-.9', '>=.9']).astype(str)
m2['bk'] = m2.b_kr + '|' + m2.b_tot + '|' + m2.b_fx
m2['pct_b'] = m2.p_base.rank(pct=True)
cmp_ = []
for bk, g in m2[m2.A1].groupby('bk'):
    ref = m2[(m2.bk == bk) & ~m2.A1 & ~m2.A2]
    cmp_.append(dict(bk=bk, n_A1=len(g), n_ref=len(ref), A1_pct=g.pct_p.mean(), ref_pct=ref.pct_p.mean() if len(ref) else np.nan,
                     A1_posrate=g.yb.mean(), ref_posrate=ref.yb.mean() if len(ref) else np.nan))
cmp_ = pd.DataFrame(cmp_); w_ = cmp_.n_A1 * cmp_.n_ref.gt(0)
f2 = dict(cells=cmp_.to_dict('records'), A1_mean_pct=float(np.average(cmp_.A1_pct, weights=w_)) if w_.sum() else None,
          ref_mean_pct_matched=float(np.average(cmp_.ref_pct.fillna(0), weights=w_)) if w_.sum() else None,
          A1_posrate=float(np.average(cmp_.A1_posrate, weights=w_)) if w_.sum() else None,
          ref_posrate_matched=float(np.average(cmp_.ref_posrate.fillna(0), weights=w_)) if w_.sum() else None,
          A1_unmatched=int(cmp_.loc[cmp_.n_ref == 0, 'n_A1'].sum()))
# F3: H10 문서를 대응표 역매핑으로 H01L 메인그룹에 되돌리면 같은 KSIC H01L 셀 과거창 수가 얼마나 바뀌나(규모만)
cy = pd.read_csv(os.path.join(W, 'wipo', 'wipo_concordance_all.csv'), dtype=str)
cy = cy[(cy.default == 'Y') & cy.to_sc.str.startswith('H10', na=False) & (cy.from_sc == 'H01L') & cy.version.isin(['20230101', '20250101'])]
back = cy.groupby('to_mg').from_mg.agg(lambda s: s.value_counts().index[0]).to_dict()   # H10 메인그룹 → 가장 많은 원천 H01L 메인그룹
hp = hd[(hd.year >= 2019) & (hd.year <= 2021) & hd.mg.str.startswith('H10')].copy()
hp['mg_back'] = hp.mg.map(back)
f3 = dict(h10_mg_mapped=float(hp.mg_back.notna().mean()), n_back_map=len(back))
hb = hp[hp.mg_back.notna()].groupby(['ksic', 'mg_back', 'cc']).appl_no.nunique().unstack('cc', fill_value=0)
hb['tot_add'] = hb.sum(1); hb['kr_add'] = hb.get('KR', 0); hb = hb.reset_index(); hb['cell'] = hb.ksic + '|' + hb.mg_back
mm = m.set_index('cell')
hb['in_eval'] = hb.cell.isin(mm.index)
hb['tot_W_now'] = hb.cell.map(mm.tot_W); hb['kr_W_now'] = hb.cell.map(mm.kr_W)
e = hb[hb.in_eval]
f3.update(target_cells_in_eval=int(len(e)), median_tot_add_ratio=float((e.tot_add / e.tot_W_now).median()) if len(e) else None,
          median_kr_add_ratio=float((e.kr_add / e.kr_W_now.replace(0, np.nan)).median()) if len(e) else None,
          target_cells_not_eligible_now=int((~hb.in_eval).sum()), would_become_eligible=None,
          back_map_sample=dict(list(back.items())[:15]))
res['F'] = dict(F0='R3에 학습 모형 객체 없음(성분 예측 npy 캐시만, comp_pub/*.npy) → 기준일 분류 피처로 동결 모형 재점수화는 재학습 없이 불가',
                F1=f1, F2=f2, F3=f3)
print(json.dumps(res['F'], ensure_ascii=False, default=str)[:3000])

# ---------- revision_cells.csv ----------
rc = m[m.A1 | m.A2 | m.A3].copy()
rc['ksic_cell'] = rc.cell
rc[['cell', 'ksic', 'mg', 'A1', 'A2', 'A3', 'new_sub', 'src_sub', 'tot_W', 'kr_W', 'kr_fut', 'D', 'y', 'yb', 'p', 'p_base', 'rank_p', 'top10', 'top100']].sort_values('rank_p') \
    .to_csv(os.path.join(W, 'revision_cells.csv'), index=False, encoding='utf-8-sig')

# ---------- 개발 원점 ----------
dv = pd.read_parquet(os.path.join(R3, 'preds', 'dev_preds_r3.parquet')); dev = []
for t0, g in dv.groupby('t0'):
    o = origin_frame(cells, KR, TOT, t0)   # 결과창 ≤2021 → 같은 스냅샷에서 다시 만든 레이블
    g = g.merge(o[['cell', 'kr_fut', 'yb', 'y']], on='cell', how='left', suffixes=('', '_r'))
    mis = int((g.yb != g.yb_r).sum()); g['D'] = g.kr_fut - g.kr_W; g = flags(g); g['p_base'] = g.p_base.astype(float)
    a, ab = auc(g.yb, g.p), auc(g.yb, g.p_base); sk = (a - ab) / (1 - ab); ps = g[g.yb == 1]
    row = dict(t0=int(t0), n=len(g), pos=int(g.yb.sum()), label_mismatch=mis, skill=sk)
    for f in ['A1', 'A2', 'new_sub']:
        row[f'{f}_elig'] = int(g[f].sum()); row[f'{f}_pos'] = int(ps[f].sum()); row[f'{f}_pos_share'] = float(ps[f].mean())
        row[f'{f}_netD_share'] = float(ps[ps[f]].D.sum() / ps.D.sum()); row[f'{f}_posrate'] = float(g[g[f]].yb.mean()) if g[f].any() else np.nan
        d = g[~g[f]]; a1_, ab1_ = auc(d.yb, d.p), auc(d.yb, d.p_base); row[f'{f}_dskill_excl'] = (a1_ - ab1_) / (1 - ab1_) - sk
    # S3 개발 원점
    g2 = g.rename(columns={})[['cell', 'ksic', 'mg', 'kr_W', 'kr_fut', 'D', 'y', 'p', 'p_base', 'yb']].copy()
    g2 = g2.merge(o[['cell', 'tot_W']], on='cell', how='left')
    nd, hcell, gg = s3_frame(g2, t0, 'wmean')
    a3, ab3 = auc(nd.yb, nd.p), auc(nd.yb, nd.p_base); row['S3_dskill'] = (a3 - ab3) / (1 - ab3) - sk
    dev.append(row)
dev = pd.DataFrame(dev); print(dev.round(4).to_string())
dev.to_csv(os.path.join(W, 'dev_origins_revision.csv'), index=False, encoding='utf-8-sig')
res['dev'] = dev.to_dict('records')
fr = res['final_impact']['A1']
res['dev_compare'] = dict(final_A1_pos_share=fr['pos_share'], dev_min=float(dev.A1_pos_share.min()), dev_max=float(dev.A1_pos_share.max()),
                          final_A1_netD_share=fr['netD_share'], dev_netD_min=float(dev.A1_netD_share.min()), dev_netD_max=float(dev.A1_netD_share.max()),
                          final_A1_posrate=fr['pos_rate'], dev_posrate_range=[float(dev.A1_posrate.min()), float(dev.A1_posrate.max())])
res['dev_compare']['flag'] = bool((fr['pos_share'] > dev.A1_pos_share.max()) and (fr['pos_share'] >= 2 * dev.A1_pos_share.max()))
pd.DataFrame(rows).to_csv(os.path.join(W, 'rescore_scenarios.csv'), index=False, encoding='utf-8-sig')
res['scenarios'] = rows
json.dump(res, open(os.path.join(W, 's2_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res['dev_compare'], ensure_ascii=False))
print('sec', round(time.time() - t_start, 1))
