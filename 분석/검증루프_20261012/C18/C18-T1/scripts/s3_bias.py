# -*- coding: utf-8 -*-
"""C18-T1 s3: 선택 편향 — 공백 달 누락(M) 대 보유(H), 보조 H'(2023 정상 달 보유). 무작위 동수 귀무 1,000회."""
import sys, pathlib, json, re, time
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
t0 = time.time()
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
H = B/'분석/검증루프_20261012/C18/C18-T1'; C = H/'cache'
T = pd.read_csv(H/'monthly_capture.csv', dtype={'ym': str}).set_index('ym')
G = set(T.index[T.gap])
KP = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '등록번호', '등록일자', '출원연도', '출원인명', '출원인주소', '메인IPC', '세세분류코드', '우선권국가'])
KP.columns = ['appl_no', 'rgst_no', 'rd', 'ay', 'nm', 'addr', 'mipc', 'ksic', 'prc']
for c in ['appl_no', 'rgst_no', 'rd', 'nm', 'addr', 'mipc']: KP[c] = KP[c].astype(str).str.strip()
KP['ksic'] = KP.ksic.astype(str)
SIDO = r'^(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충청|충북|충남|전라|전북|전남|경상|경북|경남|제주)'
CORP = r'주식회사|\(주\)|㈜|유한|법인|재단|조합|대학|연구원|연구소|공사|공단|센터|협회|병원|기술원|진흥원|INC|LTD|CORP|CO\.|GMBH|LLC|AG\b|S\.A|N\.V|B\.V|PLC|KABUSHIKI|KAISHA|컴퍼니|코포레이션|인코포레이티드|리미티드|게엠베하|가부시키'
nmU = KP.nm.str.upper()
dom = KP.addr.str.match(SIDO)
KP['atype'] = np.where(~dom, '외국', np.where(nmU.str.contains(CORP, regex=True), '법인', '개인'))
KP['sec'] = KP.mipc.str[:1].where(KP.mipc.str.match(r'^[A-H]'), '결측')
miss = pd.read_parquet(C/'s1_missing.parquet')            # RLin − RF (등록번호 기준), appl_no 포함
RLu = pd.read_parquet(C/'s1_rlin_unique.parquet')         # 권리목록 범위안(claim_count 원천)
cc = dict(zip(RLu.appl_no, pd.to_numeric(RLu.cc, errors='coerce')))
M = miss[miss.ym.isin(G)][['appl_no', 'ym']].assign(grp='M')
kpr = KP[KP.rgst_no.str.match(r'^10\d{11}$') & KP.rd.str.match(r'^\d{8}$')].assign(ym=lambda d: d.rd.str[:6])
Hh = kpr[kpr.ym.isin(G)][['appl_no', 'ym']].assign(grp='H')
H2 = kpr[(kpr.ym >= '202301') & (kpr.ym <= '202310')][['appl_no', 'ym']].assign(grp="H'")
A = pd.concat([M, Hh, H2]).merge(KP[['appl_no', 'ay', 'atype', 'sec', 'ksic']], on='appl_no', how='left')
A['cc'] = A.appl_no.map(cc); A['ay'] = pd.to_numeric(A.ay, errors='coerce')
A['period'] = np.where(A.ym <= '202412', 'G2024', 'G2025')
o = {'n': A.groupby('grp').size().to_dict(), 'cc_nonnull': A.groupby('grp').cc.apply(lambda s: float(s.notna().mean())).to_dict(),
     'unmatched_kp': int(A.atype.isna().sum())}
def tvd(a, b):
    pa = a.value_counts(normalize=True); pb = b.value_counts(normalize=True)
    return 0.5 * pa.subtract(pb, fill_value=0).abs().sum()
def smd(a, b):
    a = a.dropna(); b = b.dropna(); return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
rng = np.random.default_rng(20261017)
rows = []
for scope in ['all', 'G2024', 'G2025']:
    Ms = A[(A.grp == 'M') & ((A.period == scope) | (scope == 'all'))]
    Hs = A[(A.grp == 'H') & ((A.period == scope) | (scope == 'all'))]
    nM = len(Ms)
    # 귀무: H∪M 합동 풀에서 |M|개 무작위 → 나머지와 비교 (치환 검정). 과제 명세의 '보유군에서 동수 추출'도 함께 기록
    pool = pd.concat([Ms, Hs], ignore_index=True)
    for var, kind in [('ksic', 'c'), ('atype', 'c'), ('sec', 'c'), ('ay', 'c'), ('ay', 'n'), ('cc', 'n')]:
        f = tvd if kind == 'c' else smd
        obs = f(Ms[var], Hs[var])
        nul = []; nul2 = []
        vals = pool[var].values; Hv = Hs[var].values
        for b in range(1000):
            ii = rng.permutation(len(pool)); a_ = pd.Series(vals[ii[:nM]]); b_ = pd.Series(vals[ii[nM:]])
            nul.append(f(a_, b_))
            if b < 300:   # 보유군 내부 동수 추출(보유군 크기가 |M|보다 작으면 비복원 절반 분할)
                k = min(nM, len(Hv) // 2); jj = rng.permutation(len(Hv)); nul2.append(f(pd.Series(Hv[jj[:k]]), pd.Series(Hv[jj[k:]])))
        nul = np.abs(nul); q99 = float(np.quantile(nul, .99))
        big = (abs(obs) >= 0.10)
        verdict = '치우침' if (abs(obs) > q99 and big) else ('통계적 차이(작음)' if abs(obs) > q99 else '치우침 없음')
        rows.append({'scope': scope, 'var': var, 'stat': 'TVD' if kind == 'c' else 'SMD', 'nM': nM, 'nH': len(Hs), 'obs': obs,
                     'null_perm_q99': q99, 'null_withinH_q99': float(np.quantile(np.abs(nul2), .99)), 'verdict': verdict,
                     'M_mean_or_top': (Ms[var].mean() if kind == 'n' else Ms[var].value_counts(normalize=True).head(3).round(3).to_dict()),
                     'H_mean_or_top': (Hs[var].mean() if kind == 'n' else Hs[var].value_counts(normalize=True).head(3).round(3).to_dict())})
    # 보조: 공백 달 보유 H 대 2023 정상 달 H'
    H2s = A[A.grp == "H'"]
    for var, kind in [('atype', 'c'), ('sec', 'c'), ('ksic', 'c'), ('cc', 'n')]:
        f = tvd if kind == 'c' else smd
        rows.append({'scope': scope, 'var': var + "_H_vs_H'", 'stat': 'TVD' if kind == 'c' else 'SMD', 'nM': len(Hs), 'nH': len(H2s),
                     'obs': f(Hs[var], H2s[var]), 'verdict': '참고(귀무 없음)'})
R = pd.DataFrame(rows)
# 범주별 누락률(공백 달 전체): 유형·섹션·KSIC 상위
mr = []
for var in ['atype', 'sec', 'ksic']:
    s = A[A.grp.isin(['M', 'H'])].groupby(var).grp.agg(n='size', miss=lambda g: (g == 'M').sum())
    s['miss_rate'] = s.miss / s.n; s = s[s.n >= 200].sort_values('miss_rate')
    base = (A.grp == 'M').sum() / A.grp.isin(['M', 'H']).sum()
    s['rr'] = s.miss_rate / base; s['var'] = var
    mr.append(s.reset_index().rename(columns={var: 'level'}))
MR = pd.concat(mr); MR.to_csv(C/'s3_miss_rate_by_level.csv', index=False, encoding='utf-8-sig')
o['base_miss_rate_G'] = float((A.grp == 'M').sum() / A.grp.isin(['M', 'H']).sum())
o['ksic_miss_rate_range_n200'] = [float(MR[MR['var'] == 'ksic'].miss_rate.min()), float(MR[MR['var'] == 'ksic'].miss_rate.max())]
o['atype_rates'] = MR[MR['var'] == 'atype'][['level', 'n', 'miss_rate', 'rr']].to_dict('records')
o['sec_rates'] = MR[MR['var'] == 'sec'][['level', 'n', 'miss_rate', 'rr']].to_dict('records')
R.to_csv(H/'missing_bias.csv', index=False, encoding='utf-8-sig', float_format='%.5g')
A.to_parquet(C/'s3_groups.parquet', index=False)
o['secs'] = round(time.time() - t0)
json.dump(o, open(C/'s3_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print('done', o['secs'])
