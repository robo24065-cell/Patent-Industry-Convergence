# -*- coding: utf-8 -*-
"""data_check 보조 수치: owner 표 규모, applicant_query 대 등록원부 코드 수, CA·NPS 틀 분포 → cache/dc_extra.json"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
out = {}; rg = ro(RGF); br = ro(BRS)
out['owner_rows'] = int(pd.read_sql("select count(*) n from owner", rg).n[0]); out['owner_patents'] = int(pd.read_sql("select count(distinct rgst_no) n from owner", rg).n[0])
out['owner_date_missing'] = int(pd.read_sql("select count(*) n from owner where ownerRgstCsDate is null or length(ownerRgstCsDate)<8", rg).n[0])
out['registry_patents'] = int(pd.read_sql("select count(*) n from patent", rg).n[0]); out['registry_distinct_applicantCd'] = int(pd.read_sql("select count(distinct applicantCd) n from applicant where applicantCd is not null and applicantCd<>''", rg).n[0])
aq = pd.read_sql("select applicant_cd, status from applicant_query", br); out['aq_rows'] = int(len(aq)); out['aq_status'] = aq.status.value_counts().to_dict()
rcd = set(pd.read_sql("select distinct applicantCd from applicant where applicantCd is not null and applicantCd<>''", rg).applicantCd)
out['registry_codes_not_in_aq'] = len(rcd - set(aq.applicant_cd)); out['aq_not_in_registry'] = len(set(aq.applicant_cd) - rcd)
ca = pd.read_parquet(CA); ca.columns = ['applicant_cd', 'name', 'jr', 'br', 'nm', 'jurir', 'bizr']; out['ca_rows'] = int(len(ca)); out['ca_codes'] = int(ca.applicant_cd.nunique())
out['ca_prefix'] = ca.applicant_cd.astype(str).str[0].value_counts().to_dict(); out['ca_dup_codes'] = int(ca.applicant_cd.duplicated().sum())
P = pd.read_parquet(EXTD/'firm_nps_firm_panel.parquet', columns=['applicantCd', 'snap', 'n_emp', 'indcd']); p19 = P[(P.snap == '2019-12') & (P.n_emp.fillna(0) > 0)]
out['nps_2019_codes'] = int(p19.applicantCd.nunique())
fr_ = pd.read_parquet(CC/'pseudo_x3.parquet'); E = p19.drop_duplicates('applicantCd').set_index('applicantCd').reindex(fr_.applicant_cd)
out['pseudo_x3_emp_bins'] = pd.cut(E.n_emp, [0, 9, 49, 299, 1e9], labels=['<10', '10–49', '50–299', '300+']).value_counts().sort_index().astype(int).to_dict()
out['pseudo_x3_ind2_top'] = E.indcd.astype(str).str[:2].value_counts().head(8).to_dict()
# 공식 신규진입자 법인(비교용) 국민연금 2019-12 연결률·규모
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'newc']); nf = te.firm[te.newc.values].unique(); En = p19.drop_duplicates('applicantCd').set_index('applicantCd').reindex(nf)
out['newc_nps2019_linked'] = float(En.n_emp.notna().mean()); out['newc_emp_bins'] = pd.cut(En.n_emp, [0, 9, 49, 299, 1e9], labels=['<10', '10–49', '50–299', '300+']).value_counts().sort_index().astype(int).to_dict()
out['newc_ind2_top'] = En.indcd.astype(str).str[:2].value_counts().head(8).to_dict()
out['mem_peak'] = G.rss_peak(); jdump(out, CC/'dc_extra.json'); print(json.dumps(out, ensure_ascii=False, indent=1))
