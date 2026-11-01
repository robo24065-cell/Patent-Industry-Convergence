# -*- coding: utf-8 -*-
"""C18-T1 s2: 원인 판정 보강 — 일 단위 누락 패턴, 월 내 무작위 귀무 런 검정, 연·월 분해(공식 감소 대 공란),
C17-T2 413건 대응, 옛 스냅샷 대조, 로그 흔적. 읽기 전용."""
import sys, pathlib, json, re, sqlite3, time, glob
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
t0 = time.time()
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
H = B/'분석/검증루프_20261012/C18/C18-T1'; C = H/'cache'
ro = lambda p: sqlite3.connect(pathlib.Path(p).as_uri() + '?mode=ro', uri=True)
o = {}
T = pd.read_csv(H/'monthly_capture.csv', dtype={'ym': str}).set_index('ym')
RLu = pd.read_parquet(C/'s1_rlin_unique.parquet')
rc = ro(B/'수집파이프라인/registry_full.sqlite')
RF = pd.read_sql("select rgst_no, rgstDate rd from patent where substr(rgst_no,1,2)='10' and rgstDate>='20231001'", rc); rc.close()
rfset = set(RF.rgst_no)
# ---- 1) 일 단위: 권리목록 범위안 등록(2023-10~2025-08)의 등록일별 RF 보유율
Z = RLu[(RLu.rd >= '20231001') & (RLu.rd <= '20250822')].copy(); Z['in_rf'] = Z.rgst_no.isin(rfset)
D = Z.groupby('rd').agg(n=('in_rf', 'size'), rf=('in_rf', 'sum')); D['share'] = D.rf / D.n
D['wd'] = pd.to_datetime(D.index).dayofweek
D.to_csv(C/'s2_day_share.csv', encoding='utf-8-sig')
Dg = D[D.n >= 20]
o['days_n20'] = int(len(Dg)); o['days_share_eq0'] = int((Dg.share == 0).sum()); o['days_share_eq1'] = int((Dg.share == 1).sum())
o['days_share_between'] = int(((Dg.share > 0) & (Dg.share < 1)).sum())
o['day_share_quantiles'] = Dg.share.quantile([.05, .25, .5, .75, .95]).round(3).to_dict()
gapd = Dg[(Dg.index >= '20240101') & (Dg.index <= '20240831') | (Dg.index >= '20250201')]
o['gapdays_n'] = int(len(gapd)); o['gapdays_share_eq0'] = int((gapd.share == 0).sum()); o['gapdays_share_eq1'] = int((gapd.share == 1).sum())
o['gapdays_share_q'] = gapd.share.quantile([.05, .25, .5, .75, .95]).round(3).to_dict()
# 일 내 이항 과분산: 월 평균 보유율 p_m 기준 기대 분산 대비 관측 분산
gapd = gapd.assign(ym=gapd.index.str[:6])
pm = gapd.groupby('ym').apply(lambda d: d.rf.sum() / d.n.sum())
gapd['pm'] = gapd.ym.map(pm)
chi = (((gapd.rf - gapd.n * gapd.pm) ** 2) / (gapd.n * gapd.pm * (1 - gapd.pm))).sum()
o['day_overdispersion_chi2_df'] = [float(chi), int(len(gapd) - pm.size)]
# ---- 2) 월 내 무작위 귀무로 빈칸 런 재검정(일련번호 순)
U = Z.copy(); U['seq'] = U.rgst_no.str[2:9].astype(int); U['ym'] = U.rd.str[:6]
U = U.sort_values('seq'); x = (~U.in_rf.values).astype(int); ymv = U.ym.values
def runs(v):
    d = np.diff(np.r_[0, v, 0]); st = np.where(d == 1)[0]; en = np.where(d == -1)[0]; return en - st
r = runs(x); o['run_obs'] = {'mean': float(r.mean()), 'max': int(r.max()), 'n': int(len(r)), 'ge20_share': float(r[r >= 20].sum() / r.sum())}
rng = np.random.default_rng(20261017); mx = []; mn = []
idx_by = {m: np.where(ymv == m)[0] for m in np.unique(ymv)}
for b in range(200):
    y = x.copy()
    for m, ii in idx_by.items(): y[ii] = rng.permutation(x[ii])
    rr = runs(y); mx.append(rr.max()); mn.append(rr.mean())
o['run_null_within_month'] = {'mean_p50': float(np.median(mn)), 'mean_p99': float(np.quantile(mn, .99)), 'max_p50': float(np.median(mx)), 'max_p99': float(np.quantile(mx, .99))}
# 같은 등록일 안에서 무작위 귀무(일 단위 블록을 인정한 뒤에도 남는 군집이 있는가)
rdv = U.rd.values; idx_d = {d: np.where(rdv == d)[0] for d in np.unique(rdv)}
mx2 = []; mn2 = []
for b in range(100):
    y = x.copy()
    for d, ii in idx_d.items(): y[ii] = rng.permutation(x[ii])
    rr = runs(y); mx2.append(rr.max()); mn2.append(rr.mean())
o['run_null_within_day'] = {'mean_p50': float(np.median(mn2)), 'mean_p99': float(np.quantile(mn2, .99)), 'max_p50': float(np.median(mx2)), 'max_p99': float(np.quantile(mx2, .99))}
# ---- 3) 분해: ln(RF_m/RF_ref) = ln(O_m/O_ref) + ln(R_m/R_ref)
ref_ratio = 0.2600397085620072
dec = []
for y in ['2023', '2024', '2025']:
    s = T.loc[[i for i in T.index if i.startswith(y)]]
    rfy, oy = s.RF.sum(), s.O_pat.sum()
    dec.append({'year': y, 'RF': int(rfy), 'O_pat': int(oy), 'ratio': rfy / oy, 'expected_RF_at_ref': oy * ref_ratio,
                'deficit': oy * ref_ratio - rfy, 'deficit_rate': 1 - rfy / (oy * ref_ratio),
                'deficit_in_gap_months': float(((s.O_pat * ref_ratio - s.RF)[s.gap]).sum()),
                'deficit_rate_vs_union': 1 - rfy / s.union.sum()})
DEC = pd.DataFrame(dec)
# 2023→2024 원시 감소 분해
r23, r24 = DEC.set_index('year').loc['2023'], DEC.set_index('year').loc['2024']
tot = np.log(r24.RF / r23.RF); offp = np.log(r24.O_pat / r23.O_pat); ratp = np.log(r24.ratio / r23.ratio)
o['decomp_2023_2024'] = {'raw_change': float(r24.RF / r23.RF - 1), 'official_change': float(r24.O_pat / r23.O_pat - 1),
                         'ratio_change': float(r24.ratio / r23.ratio - 1), 'share_official_log': float(offp / tot), 'share_ratio_log': float(ratp / tot),
                         'ratio_2024_vs_ref_median': float(r24.ratio / ref_ratio - 1)}
# C02-T2 정의 재현: 2019–23 연 비율 평균
ann = []
for y in range(2019, 2024):
    s = T.loc[[i for i in T.index if i.startswith(str(y))]]; ann.append(s.RF.sum() / s.O_pat.sum())
o['c02_ref_mean_2019_23'] = float(np.mean(ann)); o['c02_d_2024'] = float(r24.ratio / np.mean(ann) - 1)
DEC.to_csv(C/'s2_decomp_year.csv', index=False, encoding='utf-8-sig')
# 월 분해(공백 달): 원시 결손 대비 2023 같은 달
mrows = []
for ym in T.index[T.gap | (T.index.str[:4] == '2024')]:
    prev = str(int(ym[:4]) - 1) + ym[4:]
    if prev not in T.index: continue
    a, b = T.loc[ym], T.loc[prev]
    mrows.append({'ym': ym, 'RF': a.RF, 'RF_prev_year': b.RF, 'raw_change': a.RF / b.RF - 1, 'O_change': a.O_pat / b.O_pat - 1,
                  'R_change': a.R_m / b.R_m - 1, 'O_vs_2023_05_10_mean': a.O_pat / T.loc['202305':'202310'].O_pat.mean() - 1,
                  'C_m': a.C_m, 'd_m': a.d_m, 'RLshare_m': a.RLshare_m})
pd.DataFrame(mrows).to_csv(C/'s2_decomp_month.csv', index=False, encoding='utf-8-sig')
# 공식 감소 15% 넘는 공백 달
MR = pd.DataFrame(mrows).set_index('ym')
o['gap_months_official_drop_ge15'] = MR.index[(MR.O_vs_2023_05_10_mean <= -0.15) & MR.index.isin(T.index[T.gap])].tolist()
o['gap_months_RLshare_range'] = [float(T.loc[T.gap, 'RLshare_m'].min()), float(T.loc[T.gap, 'RLshare_m'].max())]
o['normal_months_RLshare_range_2019_2023'] = [float(T.loc['201901':'202310', 'RLshare_m'].min()), float(T.loc['201901':'202310', 'RLshare_m'].max())]
# ---- 4) C17-T2 413건 대응
C16 = B/'분석/검증루프_20261012/C16/C16-T1'
try:
    Dd = pd.read_parquet(C16/'cache/alias_docs.parquet'); Dd['appl_no'] = Dd.appl_no.astype(str)
    PU = set(Dd[(Dd.grp == 'P') & (~Dd.is_reg)].appl_no)
    RLr = pd.read_parquet(C/'s1_rlin_rows.parquet')
    hit = RLr[RLr.appl_no.isin(PU)].groupby('appl_no').rd.min()
    early = hit[hit <= '20250822']
    miss = pd.read_parquet(C/'s1_missing.parquet')
    o['c17_hits'] = int(len(hit)); o['c17_early_le_20250822'] = int(len(early))
    o['c17_early_in_my_missing'] = int(early.index.isin(set(miss.appl_no)).sum())
    o['c17_early_ym_hist'] = early.str[:6].value_counts().sort_index().to_dict()
    o['c17_early_in_gap_months'] = int(early.str[:6].isin(set(T.index[T.gap])).sum())
except Exception as e:
    o['c17_err'] = repr(e)
# ---- 5) 옛 등록원부 스냅샷(2026-08-11~13) — 같은 번호 목록인지
for f in sorted(glob.glob(str(B/'공모전데이터/작업데이터/registry_snapshot_2026081*.sqlite'))):
    try:
        c = ro(f); tabs = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
        info = {'tables': tabs}
        if 'registry' in tabs:
            info['n'] = c.execute("select count(*) from registry").fetchone()[0]
            q = "select substr(reg_date,1,6) ym, count(*) from registry where substr(rgst_no,1,2)='10' and reg_date between '20231101' and '20250831' group by 1"
            info['gap_window_month'] = dict(c.execute(q).fetchall())
        if 'pending' in tabs:
            info['pending'] = c.execute("select count(*) from pending").fetchone()[0]
        c.close(); o['snap_' + pathlib.Path(f).stem] = info
    except Exception as e:
        o['snap_' + pathlib.Path(f).stem] = repr(e)
# ---- 6) 로그 흔적: 오류·누락·재시도 키워드
L = {}
for f in sorted(glob.glob(str(B/'수집파이프라인/logs/*.log'))):
    nm = pathlib.Path(f).name
    if not re.match(r'(collect|extract|daily)_', nm) and nm != 'bad_responses.log': continue
    s = open(f, encoding='utf-8', errors='replace').read()
    L[nm] = {'lines': s.count('\n'), 'err': len(re.findall(r'오류 [1-9]|NET:|HTTP:|RC:|non-json|전멸|실패', s)),
             'session_end': re.findall(r'세션 종료[^\n]*', s)[-3:], 'target': re.findall(r'대상 [\d,]+[^\n]*', s)[-2:]}
o['logs'] = L
o['secs'] = round(time.time() - t0)
json.dump(o, open(C/'s2_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print('done', o['secs'])
