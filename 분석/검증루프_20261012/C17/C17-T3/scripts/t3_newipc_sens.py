# C17-T3 s1: T3 원점 이후 IPC 셀(B2) 재판정용 동결 재채점 — PREREG 4절.
# 재학습 없음. 동결 p·p_base만. 원천: T3 R3 base_rows(공개 ≤2024), preds, WIPO 표제 목록·대응표(C05-T1 download 재사용, md5 재확인).
# cluster_boot_fast는 C05-T1 scripts\s2_rescore.py(final_run_r3.py 110–125행 벡터화판)를 그대로 옮겼다.
import os, sys, json, hashlib, zipfile, re, time
import numpy as np, pandas as pd, duckdb
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
T0 = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
W = os.path.join(ROOT, "검증루프_20261012", "C17", "C17-T3"); C = os.path.join(W, 'cache'); os.makedirs(C, exist_ok=True)
R3 = os.path.join(ROOT, "모델경쟁_20260927", "T3_기술셀부상예측", "R3")
C05 = os.path.join(ROOT, "검증루프_20261012", "C05", "C05-T1")
D = os.path.join(C05, 'download')
BR = os.path.join(R3, 'cache_pub_final', 'base_rows.parquet').replace('\\', '/')
OFF = json.load(open(os.path.join(R3, 'final_result_r3.json'), encoding='utf-8')); SK0 = OFF['summary']['mean_skill']
DEVR = json.load(open(os.path.join(R3, 'dev_result_r3.json'), encoding='utf-8')); DSK0 = DEVR['summary']['mean_skill']
NB = 1000
out = dict(meta=dict(script='t3_newipc_sens.py', nboot=NB))

# ---------- WIPO: md5·하위클래스 수 재확인, 2021.01 메인그룹 집합 ----------
VERS = ['20210101', '20220101', '20230101', '20240101', '20250101', '20260101']
wchk = {}; mgs = {}; subs = {}
for v in VERS:
    tab = {}
    for l in open(os.path.join(D, f'{v}md5.txt'), encoding='utf-8', errors='replace').read().splitlines()[1:]:
        p = l.split()
        if len(p) == 2: tab[p[1]] = p[0]
    zb = open(os.path.join(D, f'EN_ipc_title_list_{v}.zip'), 'rb').read()
    cz = zipfile.ZipFile(os.path.join(D, f'ipc_concordancelist_{v}.zip')).read(f'ipc_concordancelist_{v}.xml')
    z = zipfile.ZipFile(os.path.join(D, f'EN_ipc_title_list_{v}.zip')); S, M = set(), set()
    for n in z.namelist():
        if not n.endswith('.txt'): continue
        for line in z.read(n).decode('utf-8', errors='replace').splitlines():
            code = line.split('\t')[0].strip()
            if re.match(r'^[A-H]\d\d[A-Z]$', code): S.add(code)
            mm = re.match(r'^([A-H]\d\d[A-Z])(\d{4})(\d{6})$', code)
            if mm and mm.group(3) == '000000': M.add(mm.group(1) + ' ' + str(int(mm.group(2))))
    subs[v] = S; mgs[v] = M
    wchk[v] = dict(title_zip_md5_ok=hashlib.md5(zb).hexdigest() == tab.get(f'EN_ipc_title_list_{v}.zip'),
                   conc_xml_md5_ok=hashlib.md5(cz).hexdigest() == tab.get(f'ipc_concordancelist_{v}.xml'),
                   n_subclass=len(S), n_maingroup=len(M))
out['wipo_check'] = wchk
print('WIPO', wchk, flush=True)
MG21 = mgs['20210101']; SC21 = subs['20210101']
MGLATER = set().union(*[mgs[v] for v in VERS[1:]])  # 변경 기록 (1): 원점 이후 판에만 있는 메인그룹
conc = pd.read_csv(os.path.join(C05, 'wipo', 'wipo_concordance_all.csv'), dtype=str)
conc = conc[conc.version.isin(VERS[1:]) & conc.from_mg.notna() & conc.to_mg.notna()]

# ---------- 패널 재구성(문서 고유; C05-T1 mgn 매크로와 같은 규칙) ----------
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
pf = os.path.join(C, 'panel.parquet')
if not os.path.exists(pf):
    con.execute(f"""copy (select ksic, mg, cc, year, count(distinct appl_no) n from (select ksic, mgn(ipc_raw) mg, cc, year, appl_no from read_parquet('{BR}'))
      where mg is not null group by all) to '{pf.replace(chr(92), '/')}' (format parquet)""")
pan = pd.read_parquet(pf); pan['cell'] = pan.ksic + '|' + pan.mg
c05p = pd.read_parquet(os.path.join(C05, 'cache', 'panel.parquet'))
out['panel_vs_C05'] = dict(rows_mine=int(len(pan)), rows_c05=int(len(c05p)), n_mine=int(pan.n.sum()), n_c05=int(c05p.n.sum()))
print('PANEL', out['panel_vs_C05'], flush=True)

Y0 = 2000; NY = 25
def arrays(p):
    cells = np.array(sorted(p.cell.unique())); ci = np.searchsorted(cells, p.cell.values)
    KR = np.zeros((len(cells), NY)); TOT = np.zeros((len(cells), NY))
    np.add.at(TOT, (ci, p.year.values - Y0), p.n.values)
    k_ = p.cc.values == 'KR'; np.add.at(KR, (ci[k_], p.year.values[k_] - Y0), p.n.values[k_])
    return cells, KR, TOT
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
def relabel(d):
    k = int(round(0.10 * len(d))); return topk_label(d.y.values, (d.kr_fut - d.kr_W).values, d.cell.values, k)

def cluster_boot_fast(b, n_boot, seed=0):
    rng = np.random.default_rng(seed); ums = np.unique(b[0])
    pos_ = np.searchsorted(ums, b[0]); order = np.argsort(pos_, kind='stable'); cnt = np.bincount(pos_, minlength=len(ums)); st = np.r_[0, np.cumsum(cnt)[:-1]]
    o = []
    for _ in range(n_boot):
        s = rng.choice(ums, len(ums)); q = np.searchsorted(ums, s); c = cnt[q]
        ii = order[np.repeat(st[q] - np.r_[0, np.cumsum(c)[:-1]], c) + np.arange(c.sum())]
        y = b[1][ii]
        if y.min() == y.max(): continue
        a, a0 = roc_auc_score(y, b[2][ii]), roc_auc_score(y, b[3][ii]); o.append((a - a0) / (1 - a0))
    return [float(v) for v in np.percentile(np.array(o), [2.5, 97.5])]
def sk(y, p, pb):
    a, ab = roc_auc_score(y, p), roc_auc_score(y, pb); return a, ab, (a - ab) / (1 - ab)

cells, KR, TOT = arrays(pan)
# ---------- 재현 관문 ----------
P = pd.read_parquet(os.path.join(R3, 'preds', 'FINAL_R3_t0_2021.parquet'))
te = origin_frame(cells, KR, TOT, 2021)
m = P.merge(te, on='cell', how='outer', indicator=True, suffixes=('', '_r'))
g = dict(merge={k: int(v) for k, v in m._merge.value_counts().items()})
m = m[m._merge == 'both'].reset_index(drop=True)
g.update(n=int(len(m)), pos=int(m.yb_r.sum()), yb_mismatch=int((m.yb != m.yb_r).sum()), y_maxdiff=float(np.abs(m.y - m.y_r).max()))
a, ab, s0 = sk(m.yb_r.values, m.p.values, m.p_base.values.astype(float)); g.update(auc=a, skill=s0)
ci0 = cluster_boot_fast((m.mg.values, m.yb_r.values, m.p.values, m.p_base.values.astype(float)), NB)
g['ci'] = ci0; g['ci_off'] = OFF['ci95']['skill']
g['pass'] = bool(g['merge'].get('both') == 5041 and g['yb_mismatch'] == 0 and g['y_maxdiff'] < 1e-5 and abs(a - OFF['summary']['auc']) < 1e-9
                 and abs(s0 - SK0) < 1e-9 and max(abs(x - y_) for x, y_ in zip(ci0, OFF['ci95']['skill'])) < 5e-5)
print('GATE final', g, flush=True); out['gate_final'] = g; assert g['pass']
F = m[['cell', 'ksic', 'mg', 'p', 'p_base']].copy(); F['p_base'] = F.p_base.astype(float)
F = F.merge(te[['cell', 'tot_W', 'kr_W', 'kr_fut', 'y', 'yb']], on='cell')
DP = pd.read_parquet(os.path.join(R3, 'preds', 'dev_preds_r3.parquet'))
dev = {}; gd = {}
for t0, gg in DP.groupby('t0'):
    tf = origin_frame(cells, KR, TOT, int(t0)); mm = gg.merge(tf, on='cell', how='inner', suffixes=('', '_r'))
    gd[int(t0)] = dict(n_pred=int(len(gg)), n_frame=int(len(tf)), n_both=int(len(mm)), yb_mismatch=int((mm.yb != mm.yb_r).sum()))
    d = mm[['cell', 'ksic', 'mg', 'p', 'p_base', 'tot_W', 'kr_W', 'kr_fut', 'y_r', 'yb_r']].rename(columns={'y_r': 'y', 'yb_r': 'yb'}); d['p_base'] = d.p_base.astype(float)
    dev[int(t0)] = d
dsk = np.mean([sk(d.yb.values, d.p.values, d.p_base.values)[2] for d in dev.values()])
gd['_mean_skill'] = float(dsk); gd['_official'] = DSK0
gd['pass'] = bool(all(v['yb_mismatch'] == 0 and v['n_both'] == v['n_pred'] == v['n_frame'] for k, v in gd.items() if isinstance(k, int)) and abs(dsk - DSK0) < 1e-9)
print('GATE dev', gd, flush=True); out['gate_dev'] = gd; assert gd['pass']

# ---------- U 집합 ----------
def flags(d):
    sc = d.mg.str.split(' ').str[0]
    d['LEGACY'] = ~d.mg.isin(MG21) & ~d.mg.isin(MGLATER); d['U'] = ~d.mg.isin(MG21) & d.mg.isin(MGLATER); d['U_newsub'] = d.U & ~sc.isin(SC21); d['U_newmg'] = d.U & sc.isin(SC21)
    d['T48'] = sc.str.startswith('H10') | (sc == 'B64U')
    d['S1_H10'] = sc.str.startswith('H10')
    d['S2_set'] = sc.isin({'G06V', 'B64U', 'H10B', 'H10K', 'H10N', 'H10D', 'H10F', 'H10H', 'G06K', 'H01L'})
    d['T6_82'] = sc.isin({'G06V', 'B64U', 'H10B', 'H10K', 'H10N', 'H10D', 'H10F', 'H10H'}) | d.mg.isin({'H04N 23', 'H04N 25', 'G06F 18'})
    return d
F = flags(F)
for t0 in dev: dev[t0] = flags(dev[t0])
ucells = F[F.U]
out['U_final'] = dict(n=int(F.U.sum()), newsub=int(F.U_newsub.sum()), newmg=int(F.U_newmg.sum()), pos=int(F[F.U].yb.sum()),
                      by_subclass=ucells.mg.str.split(' ').str[0].value_counts().to_dict(),
                      newmg_list=sorted(F[F.U_newmg].mg.unique().tolist()),
                      T48=int(F.T48.sum()), T48_in_U=int((F.T48 & F.U).sum()), T48_not_U=sorted(F[F.T48 & ~F.U].cell.tolist()),
                      overlap_S1=int((F.U & F.S1_H10).sum()), overlap_S2=int((F.U & F.S2_set).sum()), S2_n=int(F.S2_set.sum()),
                      overlap_T6=int((F.U & F.T6_82).sum()), T6_n=int(F.T6_82.sum()), U_not_T6=sorted(F[F.U & ~F.T6_82].mg.unique().tolist()),
                      share_cells=float(F.U.mean()), share_pos=float(F[F.yb == 1].U.mean()), legacy_n=int(F.LEGACY.sum()), legacy_mg=sorted(F[F.LEGACY].mg.unique().tolist()))
out['U_dev'] = {t0: dict(n=int(d.U.sum()), pos=int(d[d.U].yb.sum()), T48=int(d.T48.sum())) for t0, d in dev.items()}
print('U', out['U_final'], out['U_dev'], flush=True)

rows = []
def score(name, d, yv, period, boot=True):
    a, ab, s = sk(yv, d.p.values, d.p_base.values)
    r = dict(variant=name, period=period, n=int(len(d)), pos=int(yv.sum()), auc=a, auc_base=ab, skill=s)
    if boot: r['ci_lo'], r['ci_hi'] = cluster_boot_fast((d.mg.values, yv, d.p.values, d.p_base.values), NB)
    return r
def run_set(name, mask_col):
    # 최종
    d = F[~F[mask_col]].reset_index(drop=True)
    r1 = score(name, d, d.yb.values, 'final'); r1['d_skill'] = r1['skill'] - SK0
    r1['pos_change'] = (int(F[F[mask_col]].yb.sum())) / 504; rows.append(r1)
    yr = relabel(d); r2 = score(name + 'r', d, yr, 'final'); r2['d_skill'] = r2['skill'] - SK0
    P0 = set(F.cell[F.yb == 1]); Pn = set(d.cell[yr == 1]); r2['pos_change'] = (len(P0 - Pn) + len(Pn - P0)) / 504; rows.append(r2)
    # 개발(원점 평균)
    for rl in ['', 'r']:
        per = {}
        for t0, dd in dev.items():
            e = dd[~dd[mask_col]].reset_index(drop=True); yv = relabel(e) if rl else e.yb.values
            per[t0] = sk(yv, e.p.values, e.p_base.values)[2]
        ms = float(np.mean(list(per.values())))
        rows.append(dict(variant=name + rl, period='dev_mean', n=int(sum(len(dd[~dd[mask_col]]) for dd in dev.values())), skill=ms, d_skill=ms - DSK0,
                         per_origin=json.dumps({k: round(v - sk(dev[k].yb.values, dev[k].p.values, dev[k].p_base.values)[2], 4) for k, v in per.items()})))
    print(name, [ (r['variant'], r['period'], round(r['skill'], 4), round(r['d_skill'], 4)) for r in rows[-4:]], flush=True)
rows.append(dict(variant='V0', period='final', n=5041, pos=504, auc=g['auc'], skill=s0, d_skill=0.0, ci_lo=ci0[0], ci_hi=ci0[1]))
rows.append(dict(variant='V0', period='dev_mean', skill=dsk, d_skill=0.0))
run_set('V1', 'U'); run_set('V48', 'T48')

# ---------- V2 구 IPC 되돌림 ----------
allmg = set(pan.mg.unique()); Umg = sorted(m_ for m_ in allmg if m_ not in MG21 and m_ in MGLATER)
def src_of(mg_):
    c = conc[(conc.to_mg == mg_) & (conc.from_mg != mg_)]
    if len(c) == 0: return None
    cy = c[c.default == 'Y']; c = cy if len(cy) else c
    return c.from_mg.value_counts().index[0]
remap = {}
for u in Umg:
    s_, seen = src_of(u), 0
    while s_ is not None and s_ not in MG21 and seen < 5: s_ = src_of(s_); seen += 1
    remap[u] = s_ if (s_ is not None and s_ in MG21) else None
rm = pd.DataFrame(dict(new_mg=list(remap.keys()), old_mg=list(remap.values())))
rm.to_csv(os.path.join(W, 'cache', 'v2_remap.csv'), index=False, encoding='utf-8-sig')
docsU = pan[pan.mg.isin(Umg)].n.sum()
out['V2_remap'] = dict(n_Umg_in_panel=len(Umg), mapped=int(rm.old_mg.notna().sum()), unmapped=sorted(rm[rm.old_mg.isna()].new_mg.tolist()),
                       doc_rows_in_Umg=int(docsU), old_targets=rm.old_mg.value_counts().head(15).to_dict())
con.register('rmap', rm[rm.old_mg.notna()])
p2f = os.path.join(C, 'panel_v2.parquet')
con.execute(f"""copy (select b.ksic, coalesce(r.old_mg, b.mg) mg, b.cc, b.year, count(distinct b.appl_no) n
   from (select ksic, mgn(ipc_raw) mg, cc, year, appl_no from read_parquet('{BR}')) b left join rmap r on b.mg = r.new_mg
   where b.mg is not null and not (b.mg in (select new_mg from rmap) and r.old_mg is null) and b.mg not in ({','.join("'" + x + "'" for x in out['V2_remap']['unmapped']) or "''"})
   group by all) to '{p2f.replace(chr(92), '/')}' (format parquet)""")
p2 = pd.read_parquet(p2f); p2['cell'] = p2.ksic + '|' + p2.mg
c2, K2, T2 = arrays(p2)
def v2_period(t0, base):
    fr = origin_frame(c2, K2, T2, t0)
    off_cells = set(base.cell); v2_cells = set(fr.cell)
    keep = base[['cell', 'mg', 'p', 'p_base', 'U']].merge(fr[['cell', 'tot_W', 'kr_W', 'kr_fut', 'y', 'yb']], on='cell', how='inner')
    # 원점 시행판 코드 셀 중 적격이 바뀐 셀
    old_off = base[~base.U]; lost = sorted(set(old_off.cell) - v2_cells); gained = sorted(v2_cells - off_cells)
    info = dict(t0=t0, n_v2_frame=len(fr), n_scored=len(keep), lost_existing=len(lost), gained_unscored=len(gained),
                lost_examples=lost[:8], gained_examples=gained[:8],
                existing_label_flip=int((keep.yb.values != base.set_index('cell').loc[keep.cell, 'yb'].values).sum()))
    return fr, keep, info
fr, keep, info = v2_period(2021, F)
r = score('V2', keep, keep.yb.values, 'final'); r['d_skill'] = r['skill'] - SK0; rows.append(r)
ys = relabel(keep); r = score('V2s', keep, ys, 'final'); r['d_skill'] = r['skill'] - SK0; rows.append(r)
out['V2_final'] = info
per, pers, dinfo = {}, {}, {}
for t0, dd in dev.items():
    _, kp, inf = v2_period(t0, dd); dinfo[t0] = {k: v for k, v in inf.items() if 'examples' not in k}
    b0 = sk(dd.yb.values, dd.p.values, dd.p_base.values)[2]
    per[t0] = sk(kp.yb.values, kp.p.values, kp.p_base.values)[2] - b0
    pers[t0] = sk(relabel(kp), kp.p.values, kp.p_base.values)[2] - b0
rows.append(dict(variant='V2', period='dev_mean', skill=DSK0 + float(np.mean(list(per.values()))), d_skill=float(np.mean(list(per.values()))), per_origin=json.dumps({k: round(v, 4) for k, v in per.items()})))
rows.append(dict(variant='V2s', period='dev_mean', skill=DSK0 + float(np.mean(list(pers.values()))), d_skill=float(np.mean(list(pers.values()))), per_origin=json.dumps({k: round(v, 4) for k, v in pers.items()})))
out['V2_dev'] = dinfo
print('V2', info, flush=True)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(W, 't3_newipc_sens.csv'), index=False, encoding='utf-8-sig')
F[F.U | F.T48][['cell', 'ksic', 'mg', 'U', 'U_newsub', 'U_newmg', 'T48', 'S2_set', 'T6_82', 'yb', 'p']].to_csv(os.path.join(W, 'cache', 'u_cells_final.csv'), index=False, encoding='utf-8-sig')
out['elapsed_s'] = round(time.time() - T0, 1)
json.dump(out, open(os.path.join(C, 't3_sens_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(res[['variant', 'period', 'n', 'pos', 'skill', 'd_skill', 'ci_lo', 'ci_hi']].to_string())
print('elapsed', out['elapsed_s'])
