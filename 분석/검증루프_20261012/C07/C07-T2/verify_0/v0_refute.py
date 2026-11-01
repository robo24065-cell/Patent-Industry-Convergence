# C07-T2 verify_0: 반박 검증자 — '정의 불일치'·'부분'·'반박' 판정을 다른 코드 경로로 다시 확인.
# s1–s4 스크립트 import 없음. 다른 라이브러리/추출 방식 사용(duckdb, XML 직접 파싱, 다른 정규화·부트스트랩).
import os, sys, json, re, zipfile, sqlite3
import numpy as np, pandas as pd, duckdb
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = ROOT + r"\분석\모델경쟁_20260927"; SH = MC + r"\_shared\ext"; REP = ROOT + r"\분석\보고서_20261012"
C6 = ROOT + r"\분석\검증루프_20261012\C06"
V0 = ROOT + r"\분석\검증루프_20261012\C07\C07-T2\verify_0"
R = {}
def auc_mw(y, s):  # 다른 구현: 정렬 누적 방식
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    o = np.argsort(s, kind='mergesort'); s2 = s[o]; y2 = y[o]
    # 동점 평균 순위
    _, inv, cnt = np.unique(s2, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt; avg = (start + csum + 1) / 2.0
    ranks = avg[inv]; n1 = y2.sum(); n0 = len(y2) - n1
    return (ranks[y2 == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
sk = lambda a, b: (a - b) / (1 - b)

# V1. 267행: Δskill(M) 구간이 0을 포함하는가(다른 부트스트랩: 층화 부트스트랩, 시드 99, 1000회) + 267행 문구에 그 단서·'등록번호'가 있는가
d = pd.read_csv(C6 + r"\C06-T3\n6_blank_full.csv")
rng = np.random.default_rng(99)
m_in = d.in_risk_M.values == 1
ds = []
idx1 = np.where(d.y.values == 1)[0]; idx0 = np.where(d.y.values == 0)[0]
for _ in range(1000):
    b = np.concatenate([rng.choice(idx1, len(idx1)), rng.choice(idx0, len(idx0))])
    bb = d.iloc[b]
    s0 = sk(auc_mw(bb.y, bb.p_champ), auc_mw(bb.y, bb.p_m0))
    mm = bb[bb.in_risk_M == 1]
    s1 = sk(auc_mw(mm.y_M, mm.p_champ), auc_mw(mm.y_M, mm.p_m0))
    ds.append(s1 - s0)
md = open(REP + r"\manuscript_v3.md", encoding='utf-8').read().split('\n')
l267 = md[266]
R['V1_267'] = {'dskill_M_boot_2.5_97.5': [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))], 'includes_0': bool(np.percentile(ds, 2.5) < 0 < np.percentile(ds, 97.5)),
               'line_has_등록번호': '등록번호' in l267, 'line_has_구간or판정불변': ('구간' in l267.split('공란')[-1]) or ('판정 불변' in l267), 'line_tail': l267[-60:]}

# V2. 410행 N6 괄호: N6 복원 문서 중 '등록 공란'(권리목록상 등록 확인)인 문서는 얼마인가 — C06-T3 후보쌍(데이터)과 권리목록(sqlite ro)으로 독립 대조
cand = pd.read_parquet(C6 + r"\C06-T3\cache\n6_cand.parquet", columns=['bz', 'appl_no', 'window', 'addr_sg_eq', 'comp_same_sg', 'rl_other_only'])
con = sqlite3.connect('file:' + ROOT + r"\수집파이프라인\business_rights.sqlite?mode=ro", uri=True)
reg = set(pd.read_sql("select distinct appl_no from right_list where substr(rgst_no,1,2)='10'", con).appl_no.astype(str).str.replace(r'\D', '', regex=True)); con.close()
cand['appl_no'] = cand.appl_no.astype(str).str.replace(r'\D', '', regex=True)
mcand = cand[(cand.addr_sg_eq == True) & (cand.comp_same_sg != True) & (cand.rl_other_only != True)]
R['V2_410'] = {'cand_pairs': len(cand), 'M_like_pairs': len(mcand), 'M_like_window_pairs': int(mcand.window.sum()),
               'M_like_registered_in_rightlist_share': round(float(mcand.appl_no.isin(reg).mean()), 4),
               'all_cand_registered_share': round(float(cand.appl_no.isin(reg).mean()), 4),
               'line410_has_N6_in_blank_parenthesis': ('N6 +0.006–0.010' in md[409]) and ('19% 적다' in md[409])}

# V3. 13행 '대기업 구간' vs 322·407행 정의 — R12 res_sub 구간 라벨 직접 확인
sub = pd.read_csv(MC + r"\_shared_report\label_origin_R10\confirm_R12\res_sub.csv", encoding='utf-8-sig')
r = sub[(sub['기준'] == 'NPS5') & (sub['구간'].astype(str).str.contains('300'))]
R['V3_13'] = {'r12_label': r[['구간종류', '구간', '기업', 'PROV-BEST', 'lo', 'hi']].values.tolist(), 'line13_has_대기업구간': '대기업 구간' in md[12],
              'line407_has_300명': '가입자 300명 이상' in md[406], 'line322_has_300명': '가입자 300명 이상 기업(81사)' in md[321]}

# V4. E-C06T3-03 보안 비중(duckdb, 과제고유번호 고유 수 기준으로 다시)
q = duckdb.sql(f"""select year, count(distinct pjt_no) n, count(distinct case when perform_grp='보안과제' then pjt_no end) s
                  from read_parquet('{(SH + chr(92) + 'ntis_project_raw.parquet').replace(chr(92), '/')}') where year between 2015 and 2024 group by year order by year""").df()
q['share'] = q.s / q.n
def w(a, b):
    t = q[(q.year >= a) & (q.year <= b)]; return round(float(t.s.sum() / t.n.sum()), 5)
R['V4_secret'] = {'by_year': q.round(5).values.tolist(), '2023_24': w(2023, 2024), 'N7_dev_outcome_2018_22': w(2018, 2022), '2015_21': w(2015, 2021)}

# V5. E-C05T2-04 N4 업태: 다른 스냅샷 규칙(결과창 직전 2023-12만, 없으면 미상)으로 제조 비중
pr = pd.read_parquet(MC + r"\N4_특허행태_기업폐업조기경보\final_test_pred_r4.parquet", columns=['applicantCd', 'y'])
lk = pd.read_parquet(SH + r"\firm_nps_applicant_link.parquet"); lk = lk[lk.nps_hit][['applicantCd', 'biz6', 'nname']]
wk = pd.read_parquet(SH + r"\firm_nps_wkpl_202312.parquet", columns=['name', 'biz6', 'indcd'])
SUF = re.compile(r"주식회사|유한회사|\(주\)|㈜|\(유\)", re.I)
wk['nname'] = wk.name.map(lambda s: re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", SUF.sub("", str(s or "")).upper()))
m = lk.merge(wk, on=['biz6', 'nname']).drop_duplicates('applicantCd')
code = dict(zip(m.applicantCd, m.indcd.astype(str).str[:2]))
pos = pr[pr.y == 1].copy(); pos['c2'] = pos.applicantCd.map(code)
known = pos[pos.c2.notna() & pos.c2.str.isdigit() & (pos.c2 != '99')]
R['V5_n4'] = {'pos': len(pos), 'known_2023_12_only': len(known), 'unknown_share': round(1 - len(known) / len(pos), 4),
              'manuf_share_known(15-37)': round(float(known.c2.astype(int).between(15, 37).mean()), 4)}

# V6. E-C05T2-05: 다른 정규화(괄호·법인격만 제거)로 '상호 탐지'를 다시 하고, 놓침 중 동명 법인 비율
pr6 = pd.read_parquet(MC + r"\N6_정부RnD기업_첫특허전환\final_preds_r5.parquet"); pr6['bz'] = pr6.bz.astype(str).str.zfill(10)
n = duckdb.sql(f"""select regexp_replace(corp_rgst_no,'[^0-9]','','g') bz, research_agency nm, count(*) c
                  from read_parquet('{(SH + chr(92) + 'ntis_project_raw.parquet').replace(chr(92), '/')}') where year<=2021 group by 1,2""").df()
n = n[n.bz.str.len() == 10]
nm2 = lambda s: re.sub(r"\s|주식회사|\(주\)|㈜|\(유\)|유한회사", "", str(s or ""))
n['k'] = n.nm.map(nm2); n = n[n.k.str.len() >= 3]
n = n.sort_values('c', ascending=False).drop_duplicates('bz')
uniq = n.groupby('k').bz.nunique(); n = n[n.k.map(uniq) == 1]
F = pr6.merge(n[['bz', 'k']], on='bz', how='left')
cols = __import__('pyarrow.parquet', fromlist=['x']).ParquetFile(ROOT + r"\공모전데이터\작업데이터\kr_patents.parquet").schema_arrow.names
kp = duckdb.sql(f"""select "{cols[8]}" a from read_parquet('{(ROOT + chr(92) + '공모전데이터' + chr(92) + '작업데이터' + chr(92) + 'kr_patents.parquet').replace(chr(92), '/')}')
                   where try_cast("{cols[1]}" as bigint) between 20220101 and 20231231""").df()
hit = set(kp.a.map(nm2))
F['det'] = F.k.isin(hit)
rows = []
with open(ROOT + r"\공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt", encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.split('¶')
        if len(p) >= 2: rows.append((p[0].strip(), nm2(p[1])))
ca = pd.DataFrame(rows, columns=['cd', 'k']); nc = ca.groupby('k').cd.nunique()
F['nc'] = F.k.map(nc).fillna(0)
miss = F[F.det & (F.y == 0)]; capt = F[F.det & (F.y == 1)]
ya = np.maximum(F.y.values, F.det.values.astype(int))
yu = np.maximum(F.y.values, (F.det & (F.nc < 2)).values.astype(int))
R['V6_n6name'] = {'detected': int(F.det.sum()), 'capture': round(len(capt) / max(F.det.sum(), 1), 4), 'missed': len(miss),
                  'missed_samename_share': round(float((miss.nc >= 2).mean()), 4), 'captured_samename_share': round(float((capt.nc >= 2).mean()), 4),
                  'skill_add_all': sk(auc_mw(ya, F.p_champ), auc_mw(ya, F.p_m0)), 'skill_add_unique_only': sk(auc_mw(yu, F.p_champ), auc_mw(yu, F.p_m0))}

# V7. 제출본: docx XML 직접 파싱으로 v8 텍스트에 변경 문장·낡은 문장 존재 확인(python-docx 미사용)
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
import xml.etree.ElementTree as ET
def xml_text(p):
    root = ET.fromstring(zipfile.ZipFile(p).read('word/document.xml'))
    return re.sub(r'\s+', '', ''.join(t.text or '' for t in root.iter(W + 't')))
t8 = xml_text(REP + r"\보고서_초안_v8.docx"); t7 = xml_text(REP + r"\보고서_초안_v7.docx")
keys = {'n6_267': '동명법인의공란출원을주소로되살려도skill은0.156–0.159다', 'p300_407': 'NPS특허가중·가입자300명이상기업·그룹안5자리',
        'stale_대기업그룹': 'NPS특허가중·대기업·그룹안5자리', 'summary_대기업구간': 'NPS특허가중과대기업구간', 'n8_286': '권리자청구추정약9%포함',
        'old_3_6_5_한계': '모든대체시험은50개산업특허풀위에서라벨만바꿨다', 'old_324': '원점2019기준NPS9,054사의26.8%'}
R['V7_docx_xml'] = {k: {'v8': v in t8, 'v7': v in t7} for k, v in keys.items()}
json.dump(R, open(V0 + r"\v0_refute_out.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(R, ensure_ascii=False, indent=1, default=float))
