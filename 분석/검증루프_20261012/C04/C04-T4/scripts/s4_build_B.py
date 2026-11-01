# s4: C04-B 개발 단위(무특허 NPS 법인 × S0) + 결과(공개 (기준일, +6개월]) + 잠금 S0 위험집합 크기만.
# 공용 점검: CORP_APPLICANT 상호 연결이 문서의 '나중 등록 여부'에 의존하는지(C03 교훈), CA 생존 편향(마지막 활동 연도별 연결률).
import sys, json, time
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C04\C04-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libA import *
T0 = time.time()
ca = load_ca(); nm2b6, nst = name_map(ca)
docs = load_docs(); docs['b6'] = docs.nname.map(nm2b6)
raw = pd.read_parquet(B + r'\공모전데이터\작업데이터\kr_patents.parquet', columns=['출원번호', '출원인명'])
docs['kor_corp'] = raw['출원인명'].fillna('').str.contains(r'주식회사|\(주\)|㈜|유한회사').values & docs.dom_prio.values
log = {}
# (1) 등록 의존성: 국내 법인 표지 문서(공개일 있는 것) 중 name_ok 연결률, 공개연도별 × 나중 등록 여부
k = docs[docs.kor_corp & docs.pub.notna()].copy()
k['py'] = k.pub.str[:4].astype(int); k['reg'] = k.rg.notna()
t = k[k.py.between(2012, 2024)].groupby(['py', 'reg']).b6.apply(lambda s: round(float(s.notna().mean()), 4)).unstack()
log['name_ok_rate_by_pubyear_x_registered'] = t.to_dict()
print('name_ok rate by pub year x registered\n', t, flush=True)
# (2) CA 생존 편향: 국내 법인 표지 출원인(상호 단위)의 마지막 보이는 연도별 CA 존재율·name_ok율
nm = k.groupby('nname').agg(last=('py', 'max'), first=('py', 'min'), n=('appl_no', 'size'))
nm['in_ca'] = nm.index.isin(set(ca.nname)); nm['name_ok'] = nm.index.isin(set(nm2b6))
nm['last_bin'] = pd.cut(nm['last'], [1999, 2005, 2010, 2015, 2020, 2025], labels=['~2005', '2006-10', '2011-15', '2016-20', '2021-25'])
sv = nm.groupby('last_bin', observed=True).agg(names=('n', 'size'), in_ca=('in_ca', 'mean'), name_ok=('name_ok', 'mean')).round(4)
log['ca_survivorship_by_last_year'] = sv.reset_index().astype(str).to_dict('records')
print(sv, flush=True)
# NTIS 수행 기업키(연도별)
nt = pd.read_parquet(EXT + r'\ntis_project_raw.parquet', columns=['year', 'corp_rgst_no', 'research_agency'])
nt['bz'] = nt.corp_rgst_no.fillna('').str.replace(r'\D', '', regex=True)
nt = nt[(nt.bz.str.len() == 10) & nt.bz.str[3:5].isin(CORP_TC)].copy()
nt['key'] = nt.bz.str[:6] + '|' + nt.research_agency.map(norm)
U = []; P = []; recs = []
first_vis = docs[docs.b6.notna()].groupby('nname').vis.min()
for s0 in DEV_B + ['202412']:
    ref = REF[s0]; t0 = int(s0[:4]); end = add_months(ref, 6)
    d0 = load_nps(s0, cols=('name', 'biz6', 'stat', 'ldong', 'indcd', 'adpt_dt', 'n_emp', 'amt'))
    f0 = firm_table(d0); f0 = f0[f0.E >= 5]
    vis_names = set(first_vis[first_vis <= ref].index)
    vis_keys = {nm2b6[n] + '|' + n for n in vis_names}
    ntk = set(nt[nt.year.between(t0 - 5, t0 - 3)].key)
    risk = f0[~f0.index.isin(vis_keys)]
    n_risk_pre = len(risk)
    risk = risk[~risk.index.isin(ntk)]
    rec = dict(s0=s0, ref=ref, end=end, nps_E0ge5=len(f0), excluded_has_vis=int(f0.index.isin(vis_keys).sum()),
               excluded_ntis=int(n_risk_pre - len(risk)), risk=len(risk))
    if s0 in DEV_B:
        assert end <= '20240630'
        w = docs[docs.pub.notna() & (docs.pub > ref) & (docs.pub <= end)]
        assert w.pub.max() <= '20240630'
        wl = w[w.b6.notna()].copy()
        wl = wl[wl.nname.map(first_vis) > ref]  # 그 상호의 첫 보이는 문서가 기준일 이후
        wl['key'] = wl.b6 + '|' + wl.nname
        hit = wl[wl.key.isin(risk.index)]
        pos = hit.groupby('key').agg(n_docs=('appl_no', 'size'), ksics=('ksic', lambda s: '|'.join(sorted(set(s))))).reset_index()
        pos['s0'] = s0
        P.append(pos)
        # 창 문서 중 국내 법인 표지의 연결 경로별 비율
        wk = w[w.kor_corp]
        rec.update(window_docs=len(w), window_kor_corp_docs=len(wk), wk_name_ok=round(float(wk.b6.notna().mean()), 4),
                   wk_first_time_name=int(((wk.b6.notna()) & (wk.nname.map(first_vis) > ref)).sum()),
                   pos_keys=len(pos), pos_rate=round(len(pos) / max(len(risk), 1), 5),
                   first_time_names_not_in_nps=int(wl[~wl.key.isin(f0.index) & ~wl.key.isin(set(d0.key))].nname.nunique()))
        rr = risk[['E', 'ind2', 'sido']].copy(); rr['s0'] = s0; rr['y'] = rr.index.isin(set(pos.key)).astype(np.int8)
        U.append(rr.reset_index())
    recs.append(rec); print(rec, round(time.time() - T0), flush=True)
pos = pd.concat(P, ignore_index=True)
pos = pos.sort_values('s0')
uniq = pos.drop_duplicates('key')
log['records'] = recs
log['P1'] = dict(origin_sum=int(len(pos)), unique_first_events=int(len(uniq)), per_origin=pos.groupby('s0').size().to_dict())
print(log['P1'], flush=True)
pos.to_parquet(C + r'\B_dev_pos.parquet', index=False)
pd.concat(U, ignore_index=True).to_parquet(C + r'\B_dev_units.parquet', index=False)
json.dump(log, open(C + r'\s4_B_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print('done', round(time.time() - T0))
