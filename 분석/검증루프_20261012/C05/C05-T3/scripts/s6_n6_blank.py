# -*- coding: utf-8 -*-
"""C05-T3 s6: N6 — 등록번호 공란(2023-11 이후 등록분) 문서가 레이블·위험집합에 주는 영향 + KR 2024 결손 무관 확인.
공란 문서 = 제공 meta KR에서 rgst_no 공란이지만 법인 권리목록(business_rights.right_list, 특허 10-)에서 같은 출원번호가 등록된 문서.
공식 레이블(공식 규칙 변형, s5b와 같은 규칙 = 공식 preds와 100% 일치)을 기준으로,
공란 문서를 권리목록 출원인코드로 기업에 붙이면 (i) y 0->1 되는 위험집합 기업, (ii) known<=2021로 위험집합에서 빠져야 할 기업 수, (iii) 재채점 Δskill.
"""
import duckdb, pandas as pd, numpy as np, os, sys, io, re, json, sqlite3
from sklearn.metrics import roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
B = os.path.join(ROOT, "분석"); SH = os.path.join(B, r"모델경쟁_20260927\_shared")
N6 = os.path.join(B, r"모델경쟁_20260927\N6_정부RnD기업_첫특허전환"); R3I = os.path.join(B, r"모델경쟁_20260927\_ideas_R3")
OUT = os.path.join(B, r"검증루프_20261012\C05\C05-T3"); CA = os.path.join(OUT, "cache")
M = os.path.join(SH, "meta", "*.parquet").replace("\\", "/")
P = os.path.join(SH, "ext", "ntis_project_raw.parquet").replace("\\", "/")
o = {}
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def nrm(s):
    s = SUF.sub("", str(s or "")).upper(); return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
# 공식 규칙 기업 연결 (s5b와 동일)
nt = duckdb.sql(f"""select year::int y, regexp_replace(corp_rgst_no,'[^0-9]','','g') bz, research_agency nm
   from read_parquet('{P}') where year::int between 2008 and 2021 and perform_grp in ('중소기업','중견기업','대기업')""").df()
nt = nt[nt.bz.str.len() == 10]; pop = set(nt[nt.y.between(2019, 2021)].bz)
ns = pd.read_parquet(os.path.join(SH, "ext", "r3nts_applicant_status.parquet"), columns=["applicantCd", "bizno", "aname"])
b2a = ns.groupby("bizno").applicantCd.apply(list).to_dict()
ns["nn"] = ns.aname.map(nrm); uq = ns.groupby("nn").applicantCd.agg(["first", "size"]); uq = uq[uq["size"] == 1]["first"].to_dict()
mvc = nt[nt.y >= 2010].groupby("bz").nm.agg(lambda s: s.value_counts().index[0]).to_dict()
rows = []
for bz in pop:
    cds = b2a.get(bz)
    if cds: rows += [(bz, x) for x in cds]
    else:
        x = uq.get(nrm(mvc.get(bz)))
        if x: rows.append((bz, x))
fl = pd.DataFrame(rows, columns=['bz', 'cd'])
af = pd.read_parquet(os.path.join(R3I, "cache", "app_firm.parquet"), columns=["applicantCd", "appl_no", "appl_date", "known_date", "src"]).rename(columns={'applicantCd': 'cd'})
af = af[af.appl_date <= 20231231]
fa = fl.merge(af, on='cd')
first = fa.groupby('bz').known_date.min()
risk = sorted(pop - set(first[first <= 20211231].index))
pos = set(fa[fa.appl_date >= 20220101].bz)
lab = pd.DataFrame({'bz': risk}); lab['y'] = lab.bz.isin(pos).astype(int)
pr = pd.read_parquet(os.path.join(N6, "final_preds_r5.parquet")); pr['bz'] = pr.bz.astype(str).str.zfill(10)
chk = pr.merge(lab, on='bz', how='outer', indicator=True)
o['official_rule_match'] = {'merge': chk._merge.value_counts().to_dict(), 'y_mismatch': int((chk.y_x != chk.y_y).sum())}
# 공란 문서
kr = duckdb.sql(f"""select appl_no, any_value(rgst_no) rg, any_value(appl_date) ad, any_value(pub_date) pd_, any_value(rgst_date) rd
   from read_parquet('{M}') where cc='KR' group by 1""").df()
kr_blank = kr[kr.rg.isna()]
o['kr_rgst_blank_total'] = int(len(kr_blank))
con = sqlite3.connect('file:' + os.path.join(ROOT, r"수집파이프라인\business_rights.sqlite") + '?mode=ro', uri=True)
rl = pd.read_sql("select applicant_cd cd, rgst_no, rgst_date, appl_no, appl_date from right_list where substr(rgst_no,1,2)='10'", con)
con.close()
rl = rl.drop_duplicates(['cd', 'appl_no'])
bl = rl[rl.appl_no.isin(set(kr_blank.appl_no))].merge(kr_blank, on='appl_no')
o['blank_docs_in_rightlist'] = int(bl.appl_no.nunique())
bl['rgst_date'] = pd.to_numeric(bl.rgst_date, errors='coerce')
o['blank_docs_rgst_year'] = (bl.drop_duplicates('appl_no').rgst_date // 10000).value_counts().sort_index().astype(int).to_dict()
o['blank_docs_min_rgst_date'] = float(bl.rgst_date.min())
for col in ('ad', 'pd_'):
    bl[col] = pd.to_numeric(bl[col], errors='coerce')
bl['known_pub'] = bl.pd_
o['blank_docs_window_appl'] = int(bl[(bl.ad >= 20220101) & (bl.ad <= 20231231)].appl_no.nunique())
o['blank_docs_pub_le_2021'] = int(bl[bl.known_pub <= 20211231].appl_no.nunique())
# 기업에 붙이기 (공식 기업 연결 코드)
bf = fl.merge(bl, on='cd')
o['blank_docs_linked_to_pop_firms'] = int(bf.appl_no.nunique()); o['pop_firms_with_blank_docs'] = int(bf.bz.nunique())
# 이미 app_firm에서 같은 (기업, 출원) 쌍이 잡혔는가
cap = bf.merge(fa[['bz', 'appl_no']].drop_duplicates(), on=['bz', 'appl_no'], how='left', indicator=True)
o['blank_pairs_already_captured_by_app_firm'] = int((cap._merge == 'both').sum()); o['blank_pairs_missed'] = int((cap._merge == 'left_only').sum())
miss = cap[cap._merge == 'left_only']
rs = set(risk)
newpos = set(miss[(miss.ad >= 20220101) & (miss.ad <= 20231231) & miss.bz.isin(rs)].bz) - pos
drop = set(miss[(miss.known_pub <= 20211231) & miss.bz.isin(rs)].bz)
o['risk_firms_y_0to1'] = len(newpos); o['risk_firms_should_drop_known_le2021'] = len(drop)
# 재채점
d2 = pr.copy()
d2['y2'] = d2.y | d2.bz.isin(newpos).astype(int)
def sk(y, a, b):
    A = roc_auc_score(y, a); Bv = roc_auc_score(y, b); return A, Bv, (A - Bv) / (1 - Bv)
A0 = sk(d2.y, d2.p_champ, d2.p_m0); A1 = sk(d2.y2, d2.p_champ, d2.p_m0)
d3 = d2[~d2.bz.isin(drop)]
A2 = sk(d3.y2, d3.p_champ, d3.p_m0)
o['rescore'] = {'official': A0, 'y_add_blank': A1, 'y_add_blank_and_drop': A2, 'dskill_add': A1[2] - A0[2], 'dskill_add_drop': A2[2] - A0[2],
                'n_after_drop': int(len(d3)), 'pos_after': int(d3.y2.sum())}
# KR 2024 결손: y·X에 2024+ 출원·등록 사용 여부
o['fa_appl_after_2023'] = int((fa.appl_date > 20231231).sum())
o['risk_uses_known_after_2021'] = 'known_date<=20211231 조건만 사용 — 2024 등록 결손은 known(=min(pub,rgst))의 pub가 있으면 무관'
o['fa_known_from_rgst_only_le2021'] = int(((fa.known_date <= 20211231)).sum())
# 2023-12 출원 비중(결과창) vs 개발 원점 마지막 달
w = fa[(fa.appl_date >= 20220101) & (fa.appl_date <= 20231231)]
o['window_last_month_share'] = float((w.appl_date >= 20231201).mean())
w19 = af[(af.appl_date >= 20200101) & (af.appl_date <= 20211231)]
o['dev_window_2020_21_last_month_share_allfirms'] = float((w19.appl_date >= 20211201).mean())
o['final_window_2022_23_last_month_share_allfirms'] = float((af[(af.appl_date >= 20220101)].appl_date >= 20231201).mean())
json.dump(o, open(os.path.join(CA, 's6_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))
