# (b)+(e) perform_grp 오표기(06=정부부처가 '중견기업', 08=중견기업이 '기타')와 보안과제 마스킹이 N6 모집단·N9 하위집단에 주는 영향
# 개발 원점(t0 2010-2019, 결과 출원 < 2022-01-01)만 레이블 계산. 최종 원점 t0=2021은 모집단 크기만(결과 미계산).
import os, re, json, sys
import numpy as np, pandas as pd, duckdb
sys.stdout.reconfigure(encoding='utf-8')
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
EXT = os.path.join(MC, r"_shared\ext"); EXTF = EXT.replace("\\", "/")
R3I = os.path.join(MC, "_ideas_R3")
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T1"
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def norm(s):
    s = SUF.sub("", str(s or "")).upper(); return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
res = {}
# 1) 보안과제 행 마스킹 범위
raw = pd.read_parquet(os.path.join(EXT, 'ntis_project_raw.parquet'), columns=['year', 'secret_flag', 'corp_rgst_no', 'ministry', 'perform_agent', 'region_cd', 'title_ko', 'total_funds', 'gov_funds', 'dev_phase'])
s = raw[raw.secret_flag == True]
res['secret_rows'] = int(len(s))
res['secret_masked_share'] = {c: float(((s[c].isna()) | (s[c].astype(str).str.upper().str.contains('SECRET'))).mean()) for c in ['corp_rgst_no', 'ministry', 'perform_agent', 'region_cd', 'title_ko', 'gov_funds', 'dev_phase']}
res['secret_total_funds_present'] = float(s.total_funds.notna().mean())
del raw, s
# 2) N6 모집단 재현: 공식(perform_grp 3종) vs 교정(perform_agent 대기업·중견기업·중소기업)
def load_nt(year_hi, corrected):
    cond = "perform_agent in ('중소기업','중견기업','대기업')" if corrected else "perform_grp in ('중소기업','중견기업','대기업')"
    nt = duckdb.sql(f"""select pjt_no, year, research_agency nm, regexp_replace(corp_rgst_no, '[^0-9]', '', 'g') bz, perform_agent, perform_grp
      from read_parquet('{EXTF}/ntis_project_raw.parquet') where year between 2008 and {year_hi} and {cond}""").df()
    return nt[nt.bz.str.len() == 10].copy()
ns = pd.read_parquet(f"{EXT}/r3nts_applicant_status.parquet", columns=["applicantCd", "bizno", "aname"])
b2a = ns.groupby("bizno").applicantCd.apply(list).to_dict()
ns["nn"] = ns.aname.map(norm); uq = ns.groupby("nn").applicantCd.agg(["first", "size"]); uq = uq[uq["size"] == 1]["first"].to_dict()
def firm_link(nt):
    src = nt[nt.year >= 2010]
    firm = src.groupby("bz").nm.agg(lambda s: s.value_counts().index[0]).rename("nm").reset_index()
    rest = nt[~nt.bz.isin(set(firm.bz))]
    if len(rest):
        f2 = rest.groupby("bz").nm.agg(lambda s: s.value_counts().index[0]).rename("nm").reset_index(); firm = pd.concat([firm, f2], ignore_index=True)
    firm["codes"] = firm.bz.map(b2a)
    nm_code = firm.nm.map(norm).map(uq)
    firm["codes"] = [c if isinstance(c, list) else ([n] if isinstance(n, str) else []) for c, n in zip(firm.codes, nm_code)]
    firm["linked"] = firm.codes.map(len) > 0
    return firm
af0 = pd.read_parquet(f"{R3I}/cache/app_firm.parquet", columns=["applicantCd", "appl_no", "appl_date", "known_date", "ksic"])
af0 = af0[af0.appl_date < 20220101]   # 개발 봉인과 동일
rows = []
for corrected in (False, True):
    nt = load_nt(2021, corrected)
    firm = firm_link(nt[nt.year <= 2019])   # 공식 dev와 같은 상호 범위(2008-2019)
    ex = firm.explode("codes").dropna(subset=["codes"])
    af = af0[af0.applicantCd.isin(set(ex.codes))].merge(ex[["bz", "codes"]], left_on="applicantCd", right_on="codes")
    first_known = af.groupby("bz").known_date.min()
    for t0 in list(range(2010, 2020)) + [2021]:
        end = t0 * 10000 + 1231; lo = (t0 + 1) * 10000 + 101; hi = (t0 + 2) * 10000 + 1231
        w = nt[nt.year.between(t0 - 2, t0)]
        idx = pd.Index(w.bz.unique())
        idx = idx[~(idx.map(first_known) <= end)]
        r = dict(version='corrected' if corrected else 'official', t0=t0, n=len(idx))
        aw = w[w.bz.isin(idx)].groupby('bz').perform_agent.agg(lambda s: set(s))
        r['n_only_govdept(06)'] = int(aw.map(lambda s: s == {'정부부처'}).sum())
        r['n_only_midsize(08)'] = int(aw.map(lambda s: s == {'중견기업'}).sum())
        if t0 <= 2019:
            o = af[(af.appl_date >= lo) & (af.appl_date <= hi)]
            yv = idx.isin(set(o.bz)); r['pos'] = int(yv.sum()); r['rate'] = float(yv.mean())
            mid = aw.map(lambda s: s == {'중견기업'}); mids = set(mid[mid].index)
            if mids:
                ym = pd.Index(sorted(mids)).isin(set(o.bz)); r['midonly_rate'] = float(ym.mean())
        rows.append(r)
T = pd.DataFrame(rows)
print(T.to_string())
T.to_csv(os.path.join(OUT, '_tab_e_n6_population.csv'), index=False, encoding='utf-8-sig')
# 공식 패널과 대조(재현 검증)
P = pd.read_parquet(os.path.join(MC, r"N6_정부RnD기업_첫특허전환\cache\panel_dev.parquet"), columns=['bz', 't0', 'y'])
chk = P.groupby('t0').agg(n=('bz', 'size'), rate=('y', 'mean')).reset_index()
m = chk.merge(T[T.version == 'official'][['t0', 'n', 'rate']], on='t0', suffixes=('_panel', '_recon'))
print(m.to_string()); res['n6_recon_check'] = m.to_dict('records')
# 3) N9: firm 하위집단에서 빠진 실제 중견기업(08) 과제 비중 (과제연도 2022, 비보안)
d = pd.read_parquet(os.path.join(EXT, 'ntis_project_raw.parquet'), columns=['year', 'perform_agent', 'perform_grp', 'secret_flag'])
d22 = d[(d.year == 2022) & (~d.secret_flag)]
firm_official = d22.perform_grp.isin(['중소기업', '대기업', '중견기업']); firm_true = d22.perform_agent.isin(['중소기업', '대기업', '중견기업'])
res['n9_2022'] = dict(n=int(len(d22)), firm_official=int(firm_official.sum()), firm_true=int(firm_true.sum()),
                      govdept_in_official_firm=int((firm_official & ~firm_true).sum()), midsize_missing_from_firm=int((firm_true & ~firm_official).sum()))
json.dump(res, open(os.path.join(OUT, '_tab_e_mislabel.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, default=float)[:3000])
