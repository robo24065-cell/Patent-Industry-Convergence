# verify_0/v2: 반박 검증(독립). 매칭 규칙에 의존하지 않는 상한으로 P1 기각이 견고한지 본다.
# A: 명단 전체 창별 신규 수, kr_patents 출원인 상호(어떤 상호든, CA 제한 없음)와 일치하는 신규 수 상한.
# B: NPS 전체 법인 기업키(E0>=5) 기준 '같은 키 새 사업장 + 사업장 수 증가' 상한(특허 연결 없음).
import sys, re, io, json
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
W = B + r'\분석\검증루프_20261012\C06\C06-T4'
EXT = B + r'\분석\모델경쟁_20260927\_shared\ext'
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def nz(s):
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", SUF.sub("", str(s or "")).upper())
out = {}
v = pd.read_csv(io.StringIO(open(W + r'\download\venture_20260521\venture.bin', 'rb').read().decode('utf-8-sig')), dtype=str)
out['venture_rows'] = len(v); out['nr_counts'] = v['신규_재확인'].value_counts().to_dict()
v['s'] = v['벤처유효시작일'].str.replace('-', ''); v['n'] = v['업체명'].map(nz)
k = pd.read_parquet(B + r'\공모전데이터\작업데이터\kr_patents.parquet', columns=['출원인명'])
pn = set(k['출원인명'].map(nz)); pn.discard('')
O = [('20230430', '20231031'), ('20231031', '20240430'), ('20240430', '20241031'), ('20241031', '20250430'), ('20250430', '20251031')]
A = []
new = v[v['신규_재확인'] == '신규']
for a, b in O:
    w = new[(new.s > a) & (new.s <= b)]
    A.append(dict(t0=a, all_new=len(w), new_name_in_any_kr_patent_applicant=int(w.n.isin(pn).sum())))
devw = new[(new.s > '20230430') & (new.s <= '20251031')]
out['A_by_window'] = A
out['A_dev_unique_names_matching_any_applicant'] = int(devw[devw.n.isin(pn)].n.nunique())
out['A_dev_all_new_rows'] = len(devw)
print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
# B 상한
CORP = {str(x) for x in range(81, 89)}
def ld(s):
    d = pd.read_parquet(EXT + rf'\firm_nps_wkpl_{s}.parquet', columns=['name', 'biz6', 'stat', 'adpt_dt', 'n_emp'])
    d = d[(d.stat.astype(str) == '1') & d.biz6.astype(str).str[3:5].isin(CORP)].copy()
    d['key'] = d.biz6.astype(str) + '|' + d.name.map(nz)
    d['ad'] = d.adpt_dt.fillna('').astype(str).str.replace('-', '').str[:8]
    return d
Bres = []
for s0, s1, r0, r1 in [('202112', '202212', '20211231', '20221231'), ('202212', '202312', '20221231', '20231231')]:
    d0, d1 = ld(s0), ld(s1)
    f0 = d0.groupby('key').agg(E0=('n_emp', 'sum'), nw0=('ad', 'size'))
    f0 = f0[f0.E0 >= 5]
    f1 = d1[d1.key.isin(f0.index)].groupby('key').agg(nw1=('ad', 'size'))
    newk = set(d1[d1.key.isin(f0.index) & (d1.ad > r0) & (d1.ad <= r1)].key)
    nw1 = f0.index.map(lambda x: f1.nw1.get(x, 0))
    y = pd.Series(f0.index.isin(newk), index=f0.index) & (pd.Series(nw1, index=f0.index) > f0.nw0)
    Bres.append(dict(s0=s0, all_corp_keys_E0ge5=len(f0), multi_wkpl_share=round(float((f0.nw0 > 1).mean()), 4), pos_all_keys=int(y.sum())))
    print(Bres[-1], flush=True)
out['B_all_keys_upper'] = Bres
json.dump(out, open(W + r'\verify_0\v2_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('done')
