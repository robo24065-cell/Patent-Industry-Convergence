# -*- coding: utf-8 -*-
"""C05-T3 s4: N6 최종 원점 t0=2021 '첫 제공범위 KR 출원' 레이블 독립 재구성 (트랙 코드 미사용).
정의(DATA_NOTES·final_spec 문장만):
 모집단: 과제연도 2019–2021 NTIS 과제 수행, perform_grp ∈ {중소기업, 중견기업, 대기업}(공유 필드 그대로), corp_rgst_no 10자리
 연결: bizno -> 출원인코드 (1차 CORP_APPLICANT 사업자번호, 보조: NTIS 최빈 상호 정규화명이 CORP_APPLICANT에서 유일)
       출원인코드 -> 제공 KR 출원: 등록특허는 registry_full applicant(모든 seq) / 미등록(또는 등록원부 없음)은 출원인명 정규화 매칭(CORP_APPLICANT 유일명)
 위험집합: 연결 출원 중 known=min(pub,rgst) <= 2021-12-31 없음
 y: 연결 출원 중 출원일 2022-01-01..2023-12-31 >=1
출력: cache/n6_lab.parquet, n6_label_rebuild.csv, cache/s4_out.json
"""
import duckdb, pandas as pd, numpy as np, json, os, sys, io, re, sqlite3, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
t00 = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
B = os.path.join(ROOT, "분석")
SH = os.path.join(B, r"모델경쟁_20260927\_shared")
N6 = os.path.join(B, r"모델경쟁_20260927\N6_정부RnD기업_첫특허전환")
OUT = os.path.join(B, r"검증루프_20261012\C05\C05-T3"); CA = os.path.join(OUT, "cache")
c = duckdb.connect(); c.execute("SET memory_limit='3GB'; SET threads=2")
P = os.path.join(SH, "ext", "ntis_project_raw.parquet").replace("\\", "/")
M = os.path.join(SH, "meta", "*.parquet").replace("\\", "/")
o = {}


def norm(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s)
    s = re.sub(r"\(주\)|㈜|주식회사|\(유\)|유한회사|\(사\)|사단법인|\(재\)|재단법인|\(합\)|합자회사|유한책임회사|농업회사법인|영농조합법인", "", s)
    s = re.sub(r"[\s\.\,\-\(\)\[\]&·'\"]", "", s).lower()
    return s or None


# 1) population
pj = c.sql(f"""select pjt_no, year::int y, perform_grp, regexp_replace(corp_rgst_no,'[^0-9]','','g') bz, research_agency nm
   from read_parquet('{P}') where year::int between 2019 and 2021 and perform_grp in ('중소기업','중견기업','대기업')""").df()
pj = pj[pj.bz.str.len() == 10]
firms = pj.bz.unique()
o['pop_firms_all'] = int(len(firms))
# 최빈 상호
mode_nm = pj.groupby(['bz', 'nm']).size().reset_index(name='k').sort_values(['bz', 'k', 'nm'], ascending=[True, False, True]).drop_duplicates('bz')
mode_nm['nn'] = mode_nm.nm.map(norm)
# 2) CORP_APPLICANT
_rows = []
with open(os.path.join(ROOT, r"공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt"), encoding="utf-8", errors="replace") as fh:
    next(fh)
    for line in fh:
        f = line.rstrip(chr(13)+chr(10)).split("¶")
        if len(f) >= 5:
            _rows.append(f[:5])
        else:
            o.setdefault('ca_bad_lines', 0); o['ca_bad_lines'] += 1
ca = pd.DataFrame(_rows, columns=['cd', 'name', 'ename', 'crn', 'bizno'])
ca = ca.apply(lambda s: s.str.strip())
ca['bz'] = ca.bizno.fillna('').str.replace(r'[^0-9]', '', regex=True)
ca['nn'] = ca.name.map(norm)
o['ca_rows'] = int(len(ca)); o['ca_with_bz10'] = int((ca.bz.str.len() == 10).sum())
o['ca_dup_cd'] = int(ca.cd.duplicated().sum())
ca10 = ca[ca.bz.str.len() == 10]
link1 = ca10[ca10.bz.isin(firms)][['bz', 'cd']].drop_duplicates()
nn_cnt = ca.groupby('nn').cd.nunique()
uniq_nn = nn_cnt[nn_cnt == 1].index
nn2cd = ca[ca.nn.isin(uniq_nn)][['nn', 'cd']].drop_duplicates('nn')
no1 = mode_nm[~mode_nm.bz.isin(link1.bz)]
link2 = no1.merge(nn2cd, on='nn')[['bz', 'cd']]
fl = pd.concat([link1.assign(route='bizno'), link2.assign(route='name')], ignore_index=True)
o['firm_link_bizno'] = int(link1.bz.nunique()); o['firm_link_name'] = int(link2.bz.nunique())
# 3) applications: provided KR
kr = c.sql(f"""select appl_no, any_value(rgst_no) rgst_no, any_value(appl_date) ad, any_value(pub_date) pd_, any_value(rgst_date) rd,
   any_value(applicant) applicant from read_parquet('{M}') where cc='KR' group by 1""").df()
for col in ('ad', 'pd_', 'rd'):
    kr[col] = pd.to_numeric(kr[col], errors='coerce')
kr['known'] = kr[['pd_', 'rd']].min(axis=1)
con = sqlite3.connect('file:' + os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite") + '?mode=ro', uri=True)
ra = pd.read_sql("select rgst_no, applicantCd cd from applicant", con)
con.close()
ra = ra.drop_duplicates()
o['reg_applicant_rows'] = int(len(ra))
reg_set = set(ra.rgst_no)
a1 = kr[kr.rgst_no.isin(reg_set)][['appl_no', 'rgst_no']].merge(ra, on='rgst_no')[['appl_no', 'cd']].assign(asrc='registry')
rest = kr[~kr.rgst_no.isin(reg_set)].copy()
rest['nn'] = rest.applicant.map(norm)
a2 = rest.merge(nn2cd, on='nn')[['appl_no', 'cd']].assign(asrc='name')
apps = pd.concat([a1, a2], ignore_index=True).drop_duplicates(['appl_no', 'cd'])
o['app_code_rows'] = int(len(apps)); o['app_code_registry'] = int(len(a1)); o['app_code_name'] = int(len(a2))
o['kr_blank_rgst_with_rgstno_missing'] = int(kr.rgst_no.isna().sum())
# 4) firm x app
fa = fl.merge(apps, on='cd').merge(kr[['appl_no', 'ad', 'known', 'rd', 'rgst_no']], on='appl_no')
known_by = fa[fa.known <= 20211231].groupby('bz').size()
risk = pd.Index(firms).difference(known_by.index)
pos_apps = fa[(fa.ad >= 20220101) & (fa.ad <= 20231231)]
ypos = set(pos_apps.bz)
lab = pd.DataFrame({'bz': risk})
lab['y_rb'] = lab.bz.isin(ypos).astype(int)
lab['linked'] = lab.bz.isin(fl.bz).astype(int)
lab['n_pos_apps'] = lab.bz.map(pos_apps.groupby('bz').appl_no.nunique()).fillna(0).astype(int)
# variant: 공란 등록(rgst_no 없음, 2023-11 이후 등록)이 포함된 양성 — appl in window but via name route only
pos_name_only = pos_apps.groupby('bz').asrc.apply(lambda s: set(s) == {'name'})
lab['pos_name_only'] = lab.bz.map(pos_name_only).fillna(False).astype(int)
# 5) compare official preds
pr = pd.read_parquet(os.path.join(N6, "final_preds_r5.parquet"))
pr['bz'] = pr.bz.astype(str).str.zfill(10)
m = pr.merge(lab, on='bz', how='outer', indicator=True)
o['unit_cmp'] = m._merge.value_counts().to_dict()
both = m[m._merge == 'both']
o['n_official'] = int(len(pr)); o['pos_official'] = int(pr.y.sum())
o['n_rebuild'] = int(len(lab)); o['pos_rebuild'] = int(lab.y_rb.sum())
o['unit_jaccard'] = float(len(both) / len(m))
o['y_agree_common'] = float((both.y == both.y_rb).mean())
o['y_off1_rb0'] = int(((both.y == 1) & (both.y_rb == 0)).sum()); o['y_off0_rb1'] = int(((both.y == 0) & (both.y_rb == 1)).sum())
m.drop(columns=['_merge']).assign(src=m._merge.astype(str)).to_csv(os.path.join(OUT, 'n6_label_rebuild.csv'), index=False, encoding='utf-8-sig')
m.assign(src=m._merge.astype(str)).drop(columns=['_merge']).to_parquet(os.path.join(CA, 'n6_lab.parquet'), index=False)
fa.to_parquet(os.path.join(CA, 'n6_firm_apps.parquet'), index=False)
fl.to_parquet(os.path.join(CA, 'n6_firm_link.parquet'), index=False)
o['sec'] = round(time.time() - t00, 1)
json.dump(o, open(os.path.join(CA, 's4_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))
