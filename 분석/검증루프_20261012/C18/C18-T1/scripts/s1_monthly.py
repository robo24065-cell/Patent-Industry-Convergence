# -*- coding: utf-8 -*-
"""C18-T1 s1: 월별 등록 집계·포착률·공식 대조·일련번호 추정·빈칸 런 분포 (pandas 경로).
원천은 읽기 전용(sqlite mode=ro, parquet 읽기). 쓰기는 C18-T1 안만."""
import sys, pathlib, json, re, sqlite3, time
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
t0 = time.time()
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
H = B/'분석/검증루프_20261012/C18/C18-T1'; C = H/'cache'
ro = lambda p: sqlite3.connect(pathlib.Path(p).as_uri() + '?mode=ro', uri=True)
o = {}
# ---------- 1) registry_full (특허 '10')
rc = ro(B/'수집파이프라인/registry_full.sqlite')
RF = pd.read_sql("select rgst_no, applNo appl_no, rgstDate rd, claimCount cc, ksic from patent where substr(rgst_no,1,2)='10'", rc)
o['rf_pat'] = len(RF); o['rf_dup'] = int(RF.rgst_no.duplicated().sum())
o['rf_rd_null'] = int(RF.rd.isna().sum()); o['rf_rd_minmax'] = [RF.rd.min(), RF.rd.max()]
rc.close()
# registry.sqlite: pending = 수집 대상 목록
gc = ro(B/'수집파이프라인/registry.sqlite')
pend = pd.read_sql("select rgst_no, appl_no from pending", gc)
regok = pd.read_sql("select rgst_no, reg_date, fetched_at from registry where status='OK'", gc)
gc.close()
o['pending'] = len(pend); o['registry_ok'] = len(regok)
o['fetched_at_range'] = [regok.fetched_at.min(), regok.fetched_at.max()]
o['fetched_day_hist'] = regok.fetched_at.str[:10].value_counts().sort_index().to_dict()
# ---------- 2) kr_registered / kr_patents
KRR = pd.read_parquet(B/'공모전데이터/작업데이터/kr_registered.parquet', columns=['출원번호', '등록번호', '등록일자'])
KRR.columns = ['appl_no', 'rgst_no', 'rd']
for c in KRR: KRR[c] = KRR[c].astype(str).str.strip()
o['krr_rows'] = len(KRR); o['krr_dup_rgst'] = int(KRR.rgst_no.duplicated().sum())
krr_all = set(KRR.rgst_no); pend_all = set(pend.rgst_no)
o['pending_vs_krr_symdiff'] = len(krr_all ^ pend_all)
KR = KRR[KRR.rgst_no.str[:2] == '10'].drop_duplicates('rgst_no')
o['krr_pat'] = len(KR)
o['rf_vs_krpat_symdiff'] = len(set(KR.rgst_no) ^ set(RF.rgst_no))
m = KR.merge(RF[['rgst_no', 'rd']], on='rgst_no', how='inner', suffixes=('_kr', '_rf'))
o['rd_mismatch_kr_rf'] = int((m.rd_kr != m.rd_rf).sum())
o['rd_mismatch_month'] = int((m.rd_kr.str[:6] != m.rd_rf.str[:6]).sum())
KP = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '등록번호', '등록일자', '출원연도'])
KP.columns = ['appl_no', 'rgst_no', 'rd', 'ay']
for c in ['appl_no', 'rgst_no', 'rd']: KP[c] = KP[c].astype(str).str.strip()
o['kp_rows'] = len(KP); o['kp_dup_appl'] = int(KP.appl_no.duplicated().sum())
P = set(KP.appl_no)
kp_blank = set(KP.loc[~KP.rgst_no.str.match(r'^\d{13}$'), 'appl_no'])
kp_reg = dict(zip(KP.appl_no, KP.rgst_no))
o['kp_blank_rgst'] = len(kp_blank)
# ---------- 3) 권리목록: 범위안(appl_no ∈ P) 특허 행 + 전체 '10' 일련번호 월별 분위
bc = ro(B/'수집파이프라인/business_rights.sqlite')
rl_in = []; seq_parts = []
cur = bc.execute("select rgst_no, rgst_date, appl_no, claim_count, applicant_cd from right_list where substr(rgst_no,1,2)='10'")
nrl = 0
while True:
    ch = cur.fetchmany(300000)
    if not ch: break
    nrl += len(ch)
    df = pd.DataFrame(ch, columns=['rgst_no', 'rd', 'appl_no', 'cc', 'h'])
    seq_parts.append(df[['rgst_no', 'rd']].drop_duplicates('rgst_no'))
    rl_in.append(df[df.appl_no.isin(P)])
bc.close()
o['rl_pat_rows'] = nrl
RL = pd.concat(rl_in, ignore_index=True)
o['rlin_rows'] = len(RL); o['rlin_unique_rgst'] = int(RL.rgst_no.nunique())
RLu = RL.sort_values(['rgst_no', 'rd']).drop_duplicates('rgst_no')
# 같은 등록번호의 등록일 불일치(권리목록 내부)
g = RL.groupby('rgst_no').rd.nunique(); o['rlin_rd_inconsistent'] = int((g > 1).sum())
# 같은 등록번호가 다른 출원번호(권리목록 내부)
g2 = RL.groupby('rgst_no').appl_no.nunique(); o['rlin_appl_inconsistent'] = int((g2 > 1).sum())
RLu.to_parquet(C/'s1_rlin_unique.parquet', index=False)
RL[['rgst_no', 'appl_no', 'h', 'rd']].to_parquet(C/'s1_rlin_rows.parquet', index=False)
SEQ = pd.concat(seq_parts).drop_duplicates('rgst_no')
# 일련번호 원천: 권리목록 전체 ∪ registry_full ∪ kr_registered (모두 '10')
SEQ = pd.concat([SEQ, RF[['rgst_no', 'rd']], KR[['rgst_no', 'rd']]]).drop_duplicates('rgst_no')
SEQ = SEQ[SEQ.rd.str.match(r'^\d{8}$', na=False)]
SEQ['seq'] = SEQ.rgst_no.str[2:9].astype(int); SEQ['ym'] = SEQ.rd.str[:6]
o['seq_src_n'] = len(SEQ)
sq = SEQ.groupby('ym').seq.agg(n='size', mx='max', p999=lambda s: np.quantile(s, 0.999), p995=lambda s: np.quantile(s, 0.995)).sort_index()
sq['ostar_max'] = sq.mx.diff(); sq['ostar_p999'] = sq.p999.diff(); sq['ostar_p995'] = sq.p995.diff()
sq.to_csv(C/'s1_seq_month.csv', encoding='utf-8-sig')
# ---------- 4) 공식 월별(연보 7.2) 파싱
dump = open(C/'s0_monthly_sheets_dump.txt', encoding='utf-8').read().split('######## ')
EDY = {'f_205_1': 2013, 'idx_2015': 2015, 'f_250_1': 2016, 'f_267_2': 2017, 'f_312_2': 2019, 'f_328_2': 2020,
       'f_360_1': 2021, 'f_380_2': 2022, 'f_400_2': 2023, 'f_416_2': 2024, 'f_437_1': 2025}
off = []
for blk in dump[1:]:
    ed = blk.split(' | ')[0].strip(); y = EDY[ed]
    my = re.search(r'Monthly registrations in (\d{4})', blk)
    if my: assert int(my.group(1)) == y, (ed, my.group(1))
    tot = None
    lines = blk.split('\n')
    for i, ln in enumerate(lines):
        mm = re.match(r'^(\d{1,2})월 \| (\d+) \| (\d+) \|', ln)
        if mm: off.append((f'{y}{int(mm.group(1)):02d}', int(mm.group(2)), int(mm.group(3)), ed))
        if ln.startswith('Total |'):
            v = ln.split(' | ')[1].strip()
            tot = int(v) if v.isdigit() else tot
    s = sum(v for k, v, u, e in off if e == ed)
    o.setdefault('official_total_check', {})[y] = [s, tot]
OFF = pd.DataFrame(off, columns=['ym', 'O_pat', 'O_um', 'edition']).set_index('ym')
OFF.to_csv(C/'s1_official_month.csv', encoding='utf-8-sig')
# ---------- 5) 월별 포착률
CUT = '20250822'   # 제공 데이터·registry_full 등록일 최대값 = 스냅샷 끝. 합집합도 이 날짜까지로 자른다
def mon(df, col='rd'):
    d = df[df[col].str.match(r'^\d{8}$', na=False)]
    d = d[d[col] <= CUT]
    return d.assign(ym=d[col].str[:6])
RFm = mon(RF); KRm = mon(KR); RLm = mon(RLu)
U = pd.concat([RFm[['rgst_no', 'ym']].assign(s='RF'), KRm[['rgst_no', 'ym']].assign(s='KR'), RLm[['rgst_no', 'ym']].assign(s='RL')])
# 합집합의 월은 우선순위 RF > KR > RL
U['pr'] = U.s.map({'RF': 0, 'KR': 1, 'RL': 2}); Uu = U.sort_values('pr').drop_duplicates('rgst_no')
rfset = set(RF.rgst_no); krset = set(KR.rgst_no)
Uu['in_rf'] = Uu.rgst_no.isin(rfset); Uu['in_kr'] = Uu.rgst_no.isin(krset)
miss = Uu[~Uu.in_rf].merge(RLu[['rgst_no', 'appl_no']], on='rgst_no', how='left')
miss['in_P'] = miss.appl_no.isin(P); miss['kp_blank'] = miss.appl_no.isin(kp_blank)
miss['kp_other_rgst'] = miss.appl_no.map(kp_reg).fillna('')
miss['kp_other_rgst'] = np.where(miss.kp_blank, '', miss.kp_other_rgst)
miss.to_parquet(C/'s1_missing.parquet', index=False)
yms = [f'{y}{m:02d}' for y in range(2015, 2026) for m in range(1, 13) if f'{y}{m:02d}' <= '202508']
T = pd.DataFrame(index=yms); T.index.name = 'ym'
T['O_pat'] = OFF.O_pat; T['ostar_p999'] = sq.ostar_p999; T['ostar_max'] = sq.ostar_max
T['RF'] = RFm.groupby('ym').size(); T['KR'] = KRm.groupby('ym').size(); T['RLin'] = RLm.groupby('ym').size()
T['union'] = Uu.groupby('ym').size()
T['RLin_not_RF'] = miss.groupby('ym').size()
T['miss_kp_blank'] = miss[miss.kp_blank].groupby('ym').size()
T['miss_kp_other_rgst'] = miss[miss.kp_other_rgst != ''].groupby('ym').size()
T = T.fillna(0)
T['C_m'] = T.RF / T.union; T['C_KR'] = T.KR / T.union; T['RF_over_KR'] = T.RF / T.KR
# 권리목록 보유 범위 안 포착률(권리목록에 있는 범위안 등록 중 RF 보유)
rl_in_rf = RLm.assign(inrf=RLm.rgst_no.isin(rfset)).groupby('ym').inrf.mean(); T['C_within_RL'] = rl_in_rf
T['R_m'] = T.RF / T.O_pat.replace(0, np.nan)
ref = T.loc[(T.index >= '201901') & (T.index <= '202310'), 'R_m']
o['R_ref_median'] = float(ref.median()); o['R_ref_mean'] = float(ref.mean()); o['R_ref_sd'] = float(ref.std()); o['R_ref_n'] = int(ref.notna().sum())
T['d_m'] = 1 - T.R_m / o['R_ref_median']
# 공식 월 인접 정상 6개월 평균 대비(정상 = 2023-05~10)
T['RLshare_m'] = T.RLin / T.O_pat.replace(0, np.nan)
T['gap'] = (T.C_m < 0.8) | (T.d_m >= 0.20)
T['partial_month'] = T.index == '202508'   # 2025-08은 22일까지(공식 O는 한 달 전체 → R_m·d_m 과소·과대 주의)
T.to_csv(H/'monthly_capture.csv', encoding='utf-8-sig', float_format='%.6g')
o['gap_months'] = list(T.index[T.gap])
# ---------- 6) 빈칸 런 분포(합집합을 일련번호 순으로 놓고 RF에 없는 연속 길이) — 2023-11~2025-08
Z = Uu[(Uu.ym >= '202311') & (Uu.ym <= '202508')].copy()  # mon()에서 이미 <= CUT
Z['seq'] = Z.rgst_no.str[2:9].astype(int)
Z = Z.sort_values('seq'); x = (~Z.in_rf.values).astype(int)
runs = []; r = 0
for v in x:
    if v: r += 1
    elif r: runs.append(r); r = 0
if r: runs.append(r)
runs = np.array(runs); p = x.mean()
o['blank_runs'] = {'n_union': int(len(x)), 'p_missing': float(p), 'n_runs': int(len(runs)), 'mean_run': float(runs.mean()),
                   'expected_mean_geom': float(1 / (1 - p)), 'max_run': int(runs.max()), 'p99_run': float(np.quantile(runs, 0.99)),
                   'share_missing_in_runs_ge20': float(runs[runs >= 20].sum() / runs.sum())}
# 무작위 순서 귀무(같은 p, 같은 길이) 최대 런 분포
rng = np.random.default_rng(20261017); mx = []
for b in range(200):
    y = rng.permutation(x); d = np.diff(np.r_[0, y, 0]); st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
    mx.append((en - st).max())
o['blank_runs']['null_max_run_p50_p99'] = [float(np.quantile(mx, .5)), float(np.quantile(mx, .99))]
# 월별 RF 일련번호 밀도: RF_m / (공식 O_m) 이미 R_m. 추가로 일련번호 구간별(1만 단위) RF 비중
Z['blk'] = Z.seq // 10000
bb = Z.groupby('blk').agg(n=('in_rf', 'size'), rf=('in_rf', 'sum'), ym_min=('ym', 'min'), ym_max=('ym', 'max'))
bb['share_rf'] = bb.rf / bb.n; bb.to_csv(C/'s1_seq_block_share.csv', encoding='utf-8-sig')
o['secs'] = round(time.time() - t0)
json.dump(o, open(C/'s1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print('done', o['secs'])
