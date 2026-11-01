# 반박검증: N6 최종 원점(t0=2021) 모집단을 공식 final 방식(firm_link 2008-2021, 출원<=2023 known_date<=2021-12-31 제외)으로 재현하고 교정판과 비교. 레이블(결과)은 계산하지 않음.
import os, re, sys, json
import pandas as pd, duckdb
sys.stdout.reconfigure(encoding='utf-8')
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
EXT = os.path.join(MC, r"_shared\ext"); EXTF = EXT.replace("\\", "/"); R3I = os.path.join(MC, "_ideas_R3")
OUT = os.path.dirname(os.path.abspath(__file__))
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
sys.path.insert(0, os.path.join(MC, "N6_정부RnD기업_첫특허전환"))
import n6_lib as L   # 공식 norm/firm_link 사용
ns = None
def load(corrected):
    cond = "perform_agent in ('중소기업','중견기업','대기업')" if corrected else "perform_grp in ('중소기업','중견기업','대기업')"
    nt = duckdb.sql(f"""select pjt_no, year, research_agency nm, regexp_replace(corp_rgst_no,'[^0-9]','','g') bz, perform_agent
      from read_parquet('{EXTF}/ntis_project_raw.parquet') where year between 2008 and 2021 and {cond}""").df()
    return nt[nt.bz.str.len()==10].copy()
af0 = pd.read_parquet(f"{R3I}/cache/app_firm.parquet", columns=["applicantCd","appl_date","known_date"])
af0 = af0[af0.appl_date <= 20231231]
res = {}
for corrected in (False, True):
    nt = load(corrected)
    firm = L.firm_link(nt)
    ex = firm.explode("codes").dropna(subset=["codes"])
    af = af0[af0.applicantCd.isin(set(ex.codes))].merge(ex[["bz","codes"]], left_on="applicantCd", right_on="codes")
    fk = af.groupby("bz").known_date.min()
    w = nt[nt.year.between(2019, 2021)]
    idx = pd.Index(w.bz.unique()); idx = idx[~(idx.map(fk) <= 20211231)]
    aw = w[w.bz.isin(idx)].groupby('bz').perform_agent.agg(set)
    res['corrected' if corrected else 'official'] = dict(n=len(idx), only_govdept=int(aw.map(lambda s: s=={'정부부처'}).sum()),
        only_midsize=int(aw.map(lambda s: s=={'중견기업'}).sum()), any_govdept=int(aw.map(lambda s: '정부부처' in s).sum()))
    if not corrected: off_idx = set(idx)
    else: res['set_diff'] = dict(added=len(set(idx)-off_idx), removed=len(off_idx-set(idx)))
# 공식 final preds 크기
fp = pd.read_parquet(os.path.join(MC, r"N6_정부RnD기업_첫특허전환\final_preds_r5.parquet"))
res['final_preds_rows'] = len(fp); res['final_preds_cols'] = list(fp.columns)[:10]
if 'bz' in fp.columns: res['official_recon_vs_preds_overlap'] = len(set(fp.bz.astype(str)) & off_idx)
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open(os.path.join(OUT, 'v01_final_pop.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
