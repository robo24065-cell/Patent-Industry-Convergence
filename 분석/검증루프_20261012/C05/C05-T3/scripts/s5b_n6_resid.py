# -*- coding: utf-8 -*-
"""C05-T3 s5b: 공식 규칙 변형(OFF)과 공식 preds의 잔여 불일치(단위 3, 양성 2) 사례 확인 — 상호 최빈값 동점 처리 점검"""
import duckdb, pandas as pd, numpy as np, os, sys, io, re, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"; SH = os.path.join(B, r"모델경쟁_20260927\_shared")
N6 = os.path.join(B, r"모델경쟁_20260927\N6_정부RnD기업_첫특허전환"); R3I = os.path.join(B, r"모델경쟁_20260927\_ideas_R3")
OUT = os.path.join(B, r"검증루프_20261012\C05\C05-T3"); CA = os.path.join(OUT, "cache")
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def nrm(s):
    s = SUF.sub("", str(s or "")).upper(); return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
P = os.path.join(SH, "ext", "ntis_project_raw.parquet").replace("\\", "/")
nt = duckdb.sql(f"""select pjt_no, year::int y, regexp_replace(corp_rgst_no,'[^0-9]','','g') bz, research_agency nm
   from read_parquet('{P}') where year::int between 2008 and 2021 and perform_grp in ('중소기업','중견기업','대기업')""").df()
nt = nt[nt.bz.str.len() == 10]
pop = set(nt[nt.y.between(2019, 2021)].bz)
ns = pd.read_parquet(os.path.join(SH, "ext", "r3nts_applicant_status.parquet"), columns=["applicantCd", "bizno", "aname"])
b2a = ns.groupby("bizno").applicantCd.apply(list).to_dict()
ns["nn"] = ns.aname.map(nrm); uq = ns.groupby("nn").applicantCd.agg(["first", "size"]); uq = uq[uq["size"] == 1]["first"].to_dict()
# value_counts 방식(트랙과 같은 pandas 동작) 최빈 상호
src = nt[nt.y >= 2010]
mvc = src.groupby("bz").nm.agg(lambda s: s.value_counts().index[0])
af = pd.read_parquet(os.path.join(R3I, "cache", "app_firm.parquet"), columns=["applicantCd", "appl_no", "appl_date", "known_date"])
af = af[af.appl_date <= 20231231]
def build(names):
    rows = []
    for bz in pop:
        cds = b2a.get(bz)
        if cds: rows += [(bz, x) for x in cds]
        else:
            x = uq.get(nrm(names.get(bz)))
            if x: rows.append((bz, x))
    fl = pd.DataFrame(rows, columns=['bz', 'applicantCd'])
    fa = fl.merge(af, on='applicantCd')
    first = fa.groupby('bz').known_date.min()
    risk = sorted(pop - set(first[first <= 20211231].index))
    pos = set(fa[(fa.appl_date >= 20220101)].bz)
    return pd.DataFrame({'bz': risk, 'y': [int(b in pos) for b in risk]}), fa
d, fa = build(mvc.to_dict())
pr = pd.read_parquet(os.path.join(N6, "final_preds_r5.parquet")); pr['bz'] = pr.bz.astype(str).str.zfill(10)
m = pr.merge(d, on='bz', how='outer', suffixes=('_off', '_rb'), indicator=True)
o = {'n': len(d), 'pos': int(d.y.sum()), 'merge': m._merge.value_counts().to_dict(),
     'y_mismatch': int(((m._merge == 'both') & (m.y_off != m.y_rb)).sum())}
bad = m[(m._merge != 'both') | (m.y_off != m.y_rb)]
cases = []
for bz in bad.bz:
    cases.append({'bz_tail': bz[-4:], 'merge': str(bad[bad.bz == bz]._merge.iloc[0]), 'y_off': bad[bad.bz == bz].y_off.iloc[0], 'y_rb': bad[bad.bz == bz].y_rb.iloc[0],
                  'has_bizno_code': bz in b2a, 'n_names': int(src[src.bz == bz].nm.nunique()),
                  'known_dates': sorted(fa[fa.bz == bz].known_date.dropna().astype(int).tolist())[:5],
                  'appl_dates_window': sorted(fa[(fa.bz == bz) & (fa.appl_date >= 20220101)].appl_date.astype(int).tolist())[:5]})
o['cases'] = cases
json.dump(o, open(os.path.join(CA, 's5b_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))
