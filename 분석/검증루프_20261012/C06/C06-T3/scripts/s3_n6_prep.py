# -*- coding: utf-8 -*-
"""C06-T3 s3: N6 준비 — (1) 공식 위험집합·레이블 자기 코드 재현 (2) 후보 (기업, 공란 문서) 쌍 생성 (3) 주소 판별 재료.
공식 규칙(C05-T3 확인): NTIS 2019–21 수행 중소·중견·대기업(원 perform_grp) 10자리 사업자번호 → r3nts 사업자번호 코드, 없으면
NTIS 최빈 상호(2010+) 정규화가 r3nts 안에서 유일하면 그 코드. 코드 → app_firm 출원. 위험집합 = known ≤ 2021-12-31 없음, y = 출원 2022–23.
"""
import duckdb, pandas as pd, numpy as np, os, sys, io, re, json, sqlite3, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
t0 = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = ROOT + r"\분석\모델경쟁_20260927"; SH = MC + r"\_shared\ext"
OUT = ROOT + r"\분석\검증루프_20261012\C06\C06-T3"; CA = OUT + r"\cache"
P = (SH + r"\ntis_project_raw.parquet").replace("\\", "/")
o = {}
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def nrm(s):
    s = SUF.sub("", str(s or "")).upper(); return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
# ---- (1) 공식 재현
nt = duckdb.sql(f"""select year::int y, regexp_replace(corp_rgst_no,'[^0-9]','','g') bz, research_agency nm, perform_grp g, region_cd
   from read_parquet('{P}') where year::int between 2008 and 2021 and perform_grp in ('중소기업','중견기업','대기업')""").df()
nt = nt[nt.bz.str.len() == 10]; pop = set(nt[nt.y.between(2019, 2021)].bz)
ns = pd.read_parquet(SH + r"\r3nts_applicant_status.parquet", columns=["applicantCd", "bizno", "aname"])
b2a = ns.groupby("bizno").applicantCd.apply(list).to_dict()
ns["nn"] = ns.aname.map(nrm); g = ns.groupby("nn").applicantCd.agg(["first", "size"]); uq = g[g["size"] == 1]["first"].to_dict()
mvc = nt[nt.y >= 2010].groupby("bz").nm.agg(lambda s: s.value_counts().index[0]).to_dict()
rows = []
for bz in pop:
    cds = b2a.get(bz)
    if cds: rows += [(bz, x, 'bizno') for x in cds]
    else:
        x = uq.get(nrm(mvc.get(bz)))
        if x: rows.append((bz, x, 'name'))
fl = pd.DataFrame(rows, columns=['bz', 'cd', 'how'])
af = pd.read_parquet(MC + r"\_ideas_R3\cache\app_firm.parquet", columns=["applicantCd", "appl_no", "appl_date", "known_date", "src"]).rename(columns={'applicantCd': 'cd'})
af = af[af.appl_date <= 20231231]
fa = fl.merge(af, on='cd')
first = fa.groupby('bz').known_date.min()
risk = sorted(pop - set(first[first <= 20211231].index))
pos = set(fa[fa.appl_date >= 20220101].bz)
pr = pd.read_parquet(MC + r"\N6_정부RnD기업_첫특허전환\final_preds_r5.parquet"); pr['bz'] = pr.bz.astype(str).str.zfill(10)
o['official_rebuild'] = dict(n=len(risk), pos=len(pos & set(risk)), units_equal=bool(set(pr.bz) == set(risk)),
                             y_equal=bool((pr.bz.isin(pos).astype(int) == pr.y).all()), off_n=len(pr), off_pos=int(pr.y.sum()))
fl.to_parquet(CA + r"\n6_fl.parquet", index=False)
fa[['bz', 'cd', 'appl_no', 'appl_date', 'known_date', 'src']].to_parquet(CA + r"\n6_fa.parquet", index=False)
# 기업 속성(규모·주체·지역·상호)
g2 = nt[nt.y.between(2019, 2021)].groupby('bz').agg(grp=('g', lambda s: s.value_counts().index[0]), region=('region_cd', lambda s: s.dropna().value_counts().index[0] if s.notna().any() else None))
firm = pd.DataFrame({'bz': risk}).merge(g2, left_on='bz', right_index=True, how='left')
firm['ntis_name'] = firm.bz.map(mvc)
firm['y'] = firm.bz.isin(pos).astype(int)
# ---- (2) CORP_APPLICANT 상호와 동명 수
rows = []
with open(ROOT + r"\공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt", encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("¶")
        if len(p) >= 5: rows.append((p[0].strip(), p[1].strip(), p[3].strip()))
ca = pd.DataFrame(rows, columns=['cd', 'aname', 'crn']); ca['n'] = ca.aname.map(nrm)
cnt = ca[ca.n.str.len() >= 2].groupby('n').cd.nunique()
ca['n_same'] = ca.n.map(cnt).fillna(0).astype(int)
ca.to_parquet(CA + r"\ca.parquet", index=False)
# ---- (3) 제공 KR 출원(kr_patents): 첫 출원인명·주소·등록번호 공란
import pyarrow.parquet as pq
_KP = ROOT + r"\공모전데이터\작업데이터\kr_patents.parquet"
_nm = pq.ParquetFile(_KP).schema_arrow.names
k = pq.read_table(_KP, columns=[_nm[i] for i in (0, 1, 3, 4, 6, 7, 8, 9, 17)]).to_pandas()
k.columns = ['appl_no', 'ad', 'rg', 'rd', 'pd', 'title', 'names', 'addr', 'ksic']
k['appl_no'] = k.appl_no.astype(str)
k['blank'] = k.rg.fillna('').astype(str).str.strip() == ''
for c in ('rd', 'pd'):
    k[c] = pd.to_numeric(k[c].astype(str).str.replace(r'\D', '', regex=True).str[:8], errors='coerce')
k['known'] = k[['pd', 'rd']].min(axis=1)
k['n'] = k.names.map(nrm)
o['kr'] = dict(rows=len(k), dup_appl=int(k.appl_no.duplicated().sum()), blank=int(k.blank.sum()),
               window=int(k.ad.between(20220101, 20231231).sum()), window_blank=int((k.ad.between(20220101, 20231231) & k.blank).sum()))
# 후보: 위험집합 기업 코드의 상호 == 문서 첫 출원인 상호 (공란 문서만), 결과창 또는 known ≤2021
fr_ = fl[fl.bz.isin(risk)].merge(ca[['cd', 'aname', 'n', 'n_same', 'crn']], on='cd', how='left')
fr_ = fr_[fr_.n.fillna('').str.len() >= 2]
kk = k[k.blank & (k.ad.between(20220101, 20231231) | (k.known <= 20211231))]
cand = fr_.merge(kk[['appl_no', 'ad', 'pd', 'rd', 'known', 'names', 'addr', 'ksic', 'title', 'n']], on='n')
# 이미 app_firm으로 잡힌 쌍 제외
have = set(zip(fa.bz, fa.appl_no))
cand['already'] = [(a, b) in have for a, b in zip(cand.bz, cand.appl_no)]
o['cand_pairs_all'] = len(cand); o['cand_already'] = int(cand.already.sum())
cand = cand[~cand.already].drop(columns='already')
# ---- 권리목록(특허 10-): 출원번호 → 보유 코드
con = sqlite3.connect('file:' + ROOT + r"\수집파이프라인\business_rights.sqlite" + '?mode=ro', uri=True)
rl = pd.read_sql("select applicant_cd cd, rgst_no, rgst_date, appl_no from right_list where substr(rgst_no,1,2)='10'", con)
aq = pd.read_sql("select applicant_cd cd, status from applicant_query", con)
con.close()
rl['appl_no'] = rl.appl_no.astype(str).str.replace(r'\D', '', regex=True)
rl = rl[rl.appl_no.isin(set(k[k.blank].appl_no))].drop_duplicates(['cd', 'appl_no'])
rl['rgst_date'] = pd.to_numeric(rl.rgst_date, errors='coerce')
rl.to_parquet(CA + r"\rl_blank.parquet", index=False)
o['rl_blank_docs'] = int(rl.appl_no.nunique())
holders = rl.groupby('appl_no').cd.apply(set).to_dict()
cand['rl_holders'] = cand.appl_no.map(lambda a: holders.get(a, set()))
cand['rl_self'] = [c in h for c, h in zip(cand.cd, cand.rl_holders)]
cand['rl_other_only'] = [(len(h) > 0) and (c not in h) for c, h in zip(cand.cd, cand.rl_holders)]
cand['rl_holders'] = cand.rl_holders.map(lambda h: '|'.join(sorted(h)))
aq_ok = set(aq[aq.status == 'OK'].cd)
cand['cd_queried'] = cand.cd.isin(aq_ok)
# P1 쌍(권리목록 정확 코드, 이름 무관): 위험집합 기업 코드가 보유한 공란 문서 중 아직 안 잡힌 것
p1 = fr_[['bz', 'cd', 'how', 'aname', 'n', 'n_same']].merge(rl[['cd', 'appl_no', 'rgst_date']], on='cd').merge(
    k[['appl_no', 'ad', 'pd', 'rd', 'known', 'names', 'addr', 'ksic', 'title', 'n']].rename(columns={'n': 'doc_n'}), on='appl_no')
p1 = p1[[(a, b) not in have for a, b in zip(p1.bz, p1.appl_no)]]
p1 = p1[p1.ad.between(20220101, 20231231) | (p1.known <= 20211231)]
p1['name_eq'] = p1.n == p1.doc_n
p1.to_parquet(CA + r"\n6_p1_pairs.parquet", index=False)
o['p1_pairs'] = len(p1); o['p1_name_eq'] = int(p1.name_eq.sum())
# ---- 주소: 문서 첫 출원인 주소 → 시도·시군구 토큰
SIDO = [('서울', '서울'), ('부산', '부산'), ('대구', '대구'), ('인천', '인천'), ('광주', '광주'), ('대전', '대전'), ('울산', '울산'), ('세종', '세종'),
        ('경기', '경기'), ('강원', '강원'), ('충청북', '충북'), ('충북', '충북'), ('충청남', '충남'), ('충남', '충남'), ('전라북', '전북'), ('전북', '전북'),
        ('전라남', '전남'), ('전남', '전남'), ('경상북', '경북'), ('경북', '경북'), ('경상남', '경남'), ('경남', '경남'), ('제주', '제주')]
def sido_txt(t):
    t = str(t or '')
    for pfx, v in SIDO:
        if t.startswith(pfx): return v
    return None
def addr_tok(a):
    toks = str(a or '').split()
    if not toks: return (None, None)
    s = sido_txt(toks[0]); sg = toks[1] if len(toks) > 1 else None
    if s == '세종': sg = None
    return (s, sg)
SIDO_CD = {'11': '서울', '26': '부산', '27': '대구', '28': '인천', '29': '광주', '30': '대전', '31': '울산', '36': '세종', '41': '경기', '42': '강원', '51': '강원',
           '43': '충북', '44': '충남', '45': '전북', '52': '전북', '46': '전남', '47': '경북', '48': '경남', '50': '제주'}
# ldong5 → 시군구 텍스트(경험적): 등록원부 출원인 주소 × NPS 연결표 × NPS 2022-12
con = sqlite3.connect('file:' + ROOT + r"\수집파이프라인\registry_full.sqlite?mode=ro", uri=True)
ra = pd.read_sql("select applicantCd cd, applicantAddr addr from applicant where substr(applicantCd,1,1)='1'", con); con.close()
ra = ra.drop_duplicates('cd')
ra[['s', 'sg']] = pd.DataFrame(ra.addr.map(addr_tok).tolist(), index=ra.index)
lk = pd.read_parquet(SH + r"\firm_nps_applicant_link.parquet"); lk = lk[lk.nps_hit]
nps = {}
for ym in ('202112', '202212', '202312'):
    w = pd.read_parquet(SH + rf"\firm_nps_wkpl_{ym}.parquet", columns=['name', 'biz6', 'ldong', 'n_emp', 'indcd'])
    w['nn'] = w.name.map(nrm); w['biz6'] = w.biz6.astype(str); w['ldong'] = w.ldong.astype(str)
    nps[ym] = w
w22 = nps['202212']
mp = lk[['applicantCd', 'biz6', 'nname']].rename(columns={'applicantCd': 'cd'}).merge(w22[['biz6', 'nn', 'ldong']], left_on=['biz6', 'nname'], right_on=['biz6', 'nn'])
mp = mp.merge(ra[['cd', 's', 'sg']], on='cd').dropna(subset=['sg'])
mp['l5'] = mp.ldong.str[:5]
mm = mp.groupby('l5').sg.agg(lambda s: (s.value_counts().index[0], s.value_counts().iloc[0] / len(s), len(s)))
l5map = {k_: v[0] for k_, v in mm.items() if v[2] >= 3}
o['ldong5_map'] = dict(n_codes=len(mm), n_used=len(l5map), mean_mode_share=float(np.mean([v[1] for v in mm.values if v[2] >= 3])))
# 위험집합 기업 → NPS 주소: biz6 + (NTIS 최빈 상호 또는 코드 상호) 정규화, 2022-12 우선, 없으면 2021-12, 2023-12
fn = fl[fl.bz.isin(risk)].merge(ca[['cd', 'n']], on='cd', how='left').groupby('bz').n.apply(lambda s: set(s.dropna()))
firm['names_n'] = [({nrm(nm)} if nm else set()) | fn.get(b, set()) for b, nm in zip(firm.bz, firm.ntis_name)]
firm['biz6'] = firm.bz.str[:6]
for ym in ('202212', '202112', '202312'):
    w = nps[ym]; ix = w.groupby(['biz6', 'nn']).agg(ldong=('ldong', 'first'), n_emp=('n_emp', 'sum'), indcd=('indcd', 'first')).to_dict('index')
    col_l, col_e, col_i = [], [], []
    for b6, nmset in zip(firm.biz6, firm.names_n):
        hit = None
        for nm in nmset:
            hit = ix.get((b6, nm))
            if hit: break
        col_l.append(hit['ldong'] if hit else None); col_e.append(hit['n_emp'] if hit else None); col_i.append(hit['indcd'] if hit else None)
    firm[f'ldong_{ym}'] = col_l; firm[f'emp_{ym}'] = col_e; firm[f'ind_{ym}'] = col_i
firm['ldong'] = firm.ldong_202212.fillna(firm.ldong_202112).fillna(firm.ldong_202312)
firm['n_emp'] = firm.emp_202212
firm['f_sido'] = firm.ldong.map(lambda x: SIDO_CD.get(str(x)[:2]) if x else None)
firm['f_sg'] = firm.ldong.map(lambda x: l5map.get(str(x)[:5]) if x else None)
o['firm_nps_addr_cover'] = float(firm.ldong.notna().mean()); o['firm_nps_2212_cover'] = float(firm.ldong_202212.notna().mean())
firm.drop(columns=['names_n']).to_parquet(CA + r"\n6_firm.parquet", index=False)
cand[['d_sido', 'd_sg']] = pd.DataFrame(cand.addr.map(addr_tok).tolist(), index=cand.index)
p1 = pd.read_parquet(CA + r"\n6_p1_pairs.parquet")
# 같은 상호 다른 코드의 등록원부 주소(경쟁 후보)
ca_n = ca[['cd', 'n']]
same = cand[['cd', 'n']].drop_duplicates().merge(ca_n.rename(columns={'cd': 'cd2'}), on='n')
same = same[same.cd2 != same.cd].merge(ra[['cd', 's', 'sg']].rename(columns={'cd': 'cd2'}), on='cd2', how='left')
comp = same.dropna(subset=['sg']).groupby('cd').apply(lambda d: set(zip(d.s, d.sg))).to_dict()
cand['comp_addrs'] = cand.cd.map(lambda c: comp.get(c, set()))
cand = cand.merge(firm[['bz', 'f_sido', 'f_sg', 'grp', 'n_emp']], on='bz', how='left')
cand['addr_sido_eq'] = (cand.d_sido.notna()) & (cand.d_sido == cand.f_sido)
cand['addr_sg_eq'] = cand.addr_sido_eq & ((cand.d_sido == '세종') | ((cand.d_sg.notna()) & (cand.d_sg == cand.f_sg)))
cand['comp_same_sg'] = [((ds, dg) in cs) for ds, dg, cs in zip(cand.d_sido, cand.d_sg, cand.comp_addrs)]
cand['comp_addrs'] = cand.comp_addrs.map(lambda s: ';'.join(f'{a} {b}' for a, b in sorted(s, key=str)))
cand['window'] = cand.ad.between(20220101, 20231231); cand['known_le2021'] = cand.known <= 20211231
cand.to_parquet(CA + r"\n6_cand.parquet", index=False)
o['cand'] = dict(pairs=len(cand), firms=int(cand.bz.nunique()), window_pairs=int(cand.window.sum()), known_le2021_pairs=int(cand.known_le2021.sum()),
                 by_nsame_eq1=int((cand.n_same == 1).sum()), rl_self=int(cand.rl_self.sum()), rl_other_only=int(cand.rl_other_only.sum()),
                 addr_sido_eq=int(cand.addr_sido_eq.sum()), addr_sg_eq=int(cand.addr_sg_eq.sum()), comp_same_sg=int(cand.comp_same_sg.sum()),
                 firm_addr_missing=int(cand.f_sido.isna().sum()), doc_addr_kr=int(cand.d_sido.notna().sum()))
o['sec'] = round(time.time() - t0)
json.dump(o, open(CA + r"\s3_out.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))
