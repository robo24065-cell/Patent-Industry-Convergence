# v1: 양성 표본 40건을 원천(sqlite mode=ro, T1 app_pub)에서 직접 재확인. 잠금 창 행은 읽지 않는다.
import re, sys, json, sqlite3
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
W = B + r'\분석\검증루프_20261012\C02\C02-T4'
MC = B + r'\분석\모델경쟁_20260927'
con = sqlite3.connect('file:' + B + r'\수집파이프라인\registry_full.sqlite?mode=ro', uri=True)
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda s: STRIP.sub('', str(s)).lower()
P = pd.read_parquet(W + r'\cache\dev_pos.parquet').sample(40, random_state=7)
ap = pd.read_parquet(MC + r'\T1_산업대기실\cache\app_pub.parquet', columns=['applicant_cd', 'ksic', 'pub_date', 'reg', 'appl_no'])
ap = ap[~ap.reg]
res = []
for r in P.itertuples():
    T = f'{r.t0}1231'
    o = con.execute("select ownerName, ownerRgstCsName, ownerRgstCsDate from owner where rgst_no=? and ownerRgstCsDate=? and ownerRgstCsDate<'20250101'", (r.rgst_no, r.d)).fetchall()
    ok_reason = any(x[1] in ('양도', '일부양도', '지분양도', '신탁계약에 의한 양도') for x in o)
    names = {norm(x[0]) for x in o if x[1] in ('양도', '일부양도', '지분양도', '신탁계약에 의한 양도')}
    k = con.execute("select ksic from patent where rgst_no=?", (r.rgst_no,)).fetchone()[0]
    # 별칭: 정규화 상호 -> 코드 유일(가시일 <= T)
    al = con.execute("select a.applicantCd, a.applicantName, min(coalesce(p.opnDate,'99999999'), coalesce(p.pubDate,'99999999'), coalesce(p.rgstDate,'99999999')) v "
                     "from applicant a join patent p on a.rgst_no=p.rgst_no where a.applicantCd=?", (r.cd,)).fetchall()
    cd_names = {norm(x[1]) for x in al if x[2] <= T}
    name_hit = bool(names & cd_names)
    # 같은 상호를 쓰는 다른 코드가 T 전에 있는지(모호성) — 상호 일치 후보만 점검
    amb = False
    for nm in names & cd_names:
        rows = con.execute("select distinct a.applicantCd, a.applicantName from applicant a join patent p on a.rgst_no=p.rgst_no "
                           "where min(coalesce(p.opnDate,'99999999'), coalesce(p.pubDate,'99999999'), coalesce(p.rgstDate,'99999999')) <= ? and a.applicantName like ?", (T, '%' + nm[:2] + '%')).fetchall()
        amb |= any(norm(x[1]) == nm and x[0] != r.cd for x in rows)
    # 자기 문서에 k 라벨이 t0 말까지 없는가
    own_k = con.execute("select count(*) from applicant a join patent p on a.rgst_no=p.rgst_no where a.applicantCd=? and p.ksic=? and "
                        "min(coalesce(p.opnDate,'99999999'), coalesce(p.pubDate,'99999999'), coalesce(p.rgstDate,'99999999')) <= ?", (r.cd, k, T)).fetchone()[0]
    own_k2 = int(((ap.applicant_cd == r.cd) & (ap.ksic.astype(str) == k) & (ap.pub_date.astype(str) <= T)).sum())
    ndoc = con.execute("select count(distinct p.applNo) from applicant a join patent p on a.rgst_no=p.rgst_no where a.applicantCd=? and "
                       "min(coalesce(p.opnDate,'99999999'), coalesce(p.pubDate,'99999999'), coalesce(p.rgstDate,'99999999')) <= ?", (r.cd, T)).fetchone()[0]
    ndoc2 = int(((ap.applicant_cd == r.cd) & (ap.pub_date.astype(str) <= T)).sum())
    inwin = f'{r.t0 + 1}0101' <= r.d <= f'{r.t0 + 2}0731'
    res.append(dict(t0=r.t0, rgst_no=r.rgst_no, ok_reason=ok_reason, k_match=(k == r.k), name_hit=name_hit, ambiguous=amb, own_k_reg=own_k, own_k_unreg=own_k2,
                    docs_ge3=(ndoc + ndoc2) >= 3, in_window=inwin, prefix1=r.cd[0] == '1'))
D = pd.DataFrame(res)
D['all_ok'] = D.ok_reason & D.k_match & D.name_hit & ~D.ambiguous & (D.own_k_reg == 0) & (D.own_k_unreg == 0) & D.docs_ge3 & D.in_window & D.prefix1
D.to_csv(W + r'\cache\v1_spot.csv', index=False, encoding='utf-8-sig')
print(D.drop(columns=['rgst_no']).to_string()); print('all_ok', D.all_ok.sum(), '/', len(D))
