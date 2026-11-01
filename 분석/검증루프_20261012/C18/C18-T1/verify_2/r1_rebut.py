# -*- coding: utf-8 -*-
"""C18-T1 verify_2 반박: 누락군 M은 '사업자 권리목록'(법인만 조회)으로만 검출된다.
보유군 H를 같은 검출 조건(권리목록에 있음, H_RL)으로 맞추면 편향이 남는가?
원천에서 직접 재계산: business_rights.sqlite(mode=ro), registry_full.sqlite(mode=ro), kr_patents.parquet.
쓰기: verify_2 안만."""
import sys, pathlib, json, sqlite3, time, re
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
t0 = time.time()
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
OUT = B/'분석/검증루프_20261012/C18/C18-T1/verify_2'
ro = lambda p: sqlite3.connect(pathlib.Path(p).as_uri() + '?mode=ro', uri=True)
G = [f'2024{m:02d}' for m in range(1, 9)] + [f'2025{m:02d}' for m in range(2, 9)]
END = '20250822'
o = {}
KP = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '등록번호', '등록일자', '출원연도', '출원인명', '출원인주소', '메인IPC', '세세분류코드'])
KP.columns = ['appl_no', 'rgst_no', 'rd', 'ay', 'nm', 'addr', 'mipc', 'ksic']
for c in ['appl_no', 'rgst_no', 'rd', 'nm', 'addr', 'mipc']: KP[c] = KP[c].astype(str).str.strip()
KP['ksic'] = KP.ksic.astype(str)
SIDO = r'^(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충청|충북|충남|전라|전북|전남|경상|경북|경남|제주)'
CORP = r'주식회사|\(주\)|㈜|유한|법인|재단|조합|대학|연구원|연구소|공사|공단|센터|협회|병원|기술원|진흥원|INC|LTD|CORP|CO\.|GMBH|LLC|AG\b|S\.A|N\.V|B\.V|PLC|KABUSHIKI|KAISHA|컴퍼니|코포레이션|인코포레이티드|리미티드|게엠베하|가부시키'
dom = KP.addr.str.match(SIDO)
KP['atype'] = np.where(~dom, '외국', np.where(KP.nm.str.upper().str.contains(CORP, regex=True), '법인', '개인'))
KP['sec'] = KP.mipc.str[:1].where(KP.mipc.str.match(r'^[A-H]'), '결측')
KP['ay'] = pd.to_numeric(KP.ay, errors='coerce')
P = set(KP.appl_no)
# registry_full 특허 등록번호
rc = ro(B/'수집파이프라인/registry_full.sqlite')
rf = set(r[0] for r in rc.execute("select rgst_no from patent where substr(rgst_no,1,2)='10'"))
rc.close()
# 권리목록: 공백 달 등록, 특허, 범위안
bc = ro(B/'수집파이프라인/business_rights.sqlite')
RL = pd.read_sql("select rgst_no, rgst_date rd, appl_no, claim_count cc, applicant_cd, raw from right_list "
                 "where substr(rgst_no,1,2)='10' and rgst_date>='20240101' and rgst_date<=?", bc, params=(END,))
# 조회된 출원인 코드 앞자리 분포(어떤 출원인이 조회 대상인가)
aq = pd.read_sql("select applicant_cd, status, truncated, total_count, fetched from applicant_query", bc)
bc.close()
o['aq_status_by_prefix'] = aq.assign(p=aq.applicant_cd.str[:1]).groupby(['p', 'status']).size().astype(int).reset_index().values.tolist()
o['aq_truncated_ok'] = int(((aq.status == 'OK') & (aq.truncated == 1)).sum())
RL = RL[RL.appl_no.isin(P) & RL.rd.str[:6].isin(G)]
RL['indv'] = RL.raw.str.extract(r'"indvCorpTpcd": "([^"]*)"')[0]
RL['cdp'] = RL.applicant_cd.str[:1]
o['RL_rows_inG'] = len(RL)
o['RL_indvCorp'] = RL.indv.value_counts().to_dict()
o['RL_cd_prefix'] = RL.cdp.value_counts().to_dict()
RLu = RL.sort_values('rgst_no').drop_duplicates('rgst_no')
cc = dict(zip(RLu.appl_no, pd.to_numeric(RLu.cc, errors='coerce')))
rl_appl = set(RLu.appl_no)
# 군 정의
M = RLu[~RLu.rgst_no.isin(rf)][['appl_no', 'rd']].assign(grp='M')
kpr = KP[KP.rgst_no.str.match(r'^10\d{11}$') & KP.rd.str.match(r'^\d{8}$')]
Hall = kpr[kpr.rd.str[:6].isin(G)][['appl_no', 'rd']].assign(grp='H')
o['n_M'] = len(M); o['n_H'] = len(Hall)
o['M_in_KP_blank'] = float(KP.set_index('appl_no').loc[M.appl_no, 'rgst_no'].str.match(r'^\d{13}$').eq(False).mean())
Hall['inRL'] = Hall.appl_no.isin(rl_appl)
A = pd.concat([M.assign(inRL=True), Hall]).merge(KP[['appl_no', 'ay', 'atype', 'sec', 'ksic']], on='appl_no', how='left')
A['cc'] = A.appl_no.map(cc)
A['period'] = np.where(A.rd <= '20241231', 'G2024', 'G2025')
# 1) 보유군의 권리목록 커버리지(검출 가능성) — 유형별
cov = A[A.grp == 'H'].groupby('atype').inRL.agg(['size', 'mean'])
o['H_RLcoverage_by_atype'] = cov.round(4).reset_index().values.tolist()
o['H_RLcoverage_all'] = float(A[A.grp == 'H'].inRL.mean())
o['H_RLcoverage_by_sec'] = A[A.grp == 'H'].groupby('sec').inRL.mean().round(3).to_dict()
# 2) 누락률: 원판식(M/(M+H)) 대 검출조건판(M/(M+H_RL))
def rates(df, var):
    s = df.groupby(var).grp.agg(n='size', m=lambda g: (g == 'M').sum()); s['rate'] = s.m / s.n; return s
A_all = A; A_rl = A[A.inRL]
o['rate_atype_orig'] = rates(A_all, 'atype').round(4).reset_index().values.tolist()
o['rate_atype_RLmatched'] = rates(A_rl, 'atype').round(4).reset_index().values.tolist()
o['base_rate_orig'] = float((A_all.grp == 'M').mean()); o['base_rate_RL'] = float((A_rl.grp == 'M').mean())
def tvd(a, b):
    pa = a.value_counts(normalize=True); pb = b.value_counts(normalize=True); return 0.5 * pa.subtract(pb, fill_value=0).abs().sum()
def smd(a, b):
    a = a.dropna(); b = b.dropna(); return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
rng = np.random.default_rng(20261018)
rows = []
for comp, D in [('orig_H_all', A_all), ('H_RL_matched', A_rl), ('H_RL_corp_only', A_rl[A_rl.atype == '법인']), ('H_RL_dom_corp_foreign', A_rl[A_rl.atype != '개인'])]:
    for scope in ['all', 'G2024', 'G2025']:
        Ds = D if scope == 'all' else D[D.period == scope]
        Ms = Ds[Ds.grp == 'M']; Hs = Ds[Ds.grp == 'H']
        pool = pd.concat([Ms, Hs], ignore_index=True); nM = len(Ms)
        for var, kind in [('ksic', 'c'), ('atype', 'c'), ('sec', 'c'), ('ay', 'n'), ('cc', 'n')]:
            f = tvd if kind == 'c' else smd
            obs = f(Ms[var], Hs[var]); vals = pool[var].values; nul = []
            for b in range(300):
                ii = rng.permutation(len(pool)); nul.append(abs(f(pd.Series(vals[ii[:nM]]), pd.Series(vals[ii[nM:]]))))
            q99 = float(np.quantile(nul, .99))
            v = '치우침' if (abs(obs) > q99 and abs(obs) >= 0.10) else ('통계적 차이(작음)' if abs(obs) > q99 else '치우침 없음')
            rows.append(dict(comp=comp, scope=scope, var=var, nM=nM, nH=len(Hs), obs=round(float(obs), 5), q99=round(q99, 5), verdict=v,
                             M_mean=(round(float(Ms[var].mean()), 3) if kind == 'n' else None), H_mean=(round(float(Hs[var].mean()), 3) if kind == 'n' else None)))
R = pd.DataFrame(rows); R.to_csv(OUT/'r1_bias_matched.csv', index=False, encoding='utf-8-sig')
# KSIC 누락률 범위(검출조건판, n>=200)
ks = rates(A_rl, 'ksic'); ks = ks[ks.n >= 200].sort_values('rate')
o['ksic_rate_RL_n200_minmax'] = [float(ks.rate.min()), float(ks.rate.max())]
o['ksic_rate_RL_top5'] = ks.tail(5).round(3).reset_index().values.tolist()
o['ksic_rate_RL_bottom5'] = ks.head(5).round(3).reset_index().values.tolist()
o['secs'] = round(time.time() - t0)
json.dump(o, open(OUT/'r1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(R.to_string()); print(json.dumps(o, ensure_ascii=False, default=str)[:4000])
