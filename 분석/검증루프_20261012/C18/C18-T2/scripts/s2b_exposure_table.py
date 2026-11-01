# -*- coding: utf-8 -*-
r"""C18-T2 s2b: 노출 지도 exposure.csv — 공식 9트랙·원고 수치 생성 코드에서 해외(US·EP·JP) 메타 행/필드 사용처를 grep으로 찾아 줄 번호를 붙인다.
 판단(사용 유형·동결 스펙 포함 여부·노출)은 cache/s2_grep_hits.csv 와 각 동결 스펙을 읽고 적은 것이며, 줄 번호와 근거 문자열은 스크립트가 다시 찾는다.
 결함: D1 = US 7/8자리 이중 수록, D1E = EP 짧은/긴 형식 이중 수록, D2 = US 등록번호·등록일 2021-04 이후만, D2J = JP 등록 필드 2021-01 이후만(같은 유형), D2E = EP 등록일 2021H1·2022-04~09 공백,
       FT = family 토큰 번호 형식 이중 표기(US7324782|US07324782)."""
import os, sys, re, csv
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
ROWS = [
 # track, file(MC 상대), regex, 사용 유형, 국가, 결함, 동결 스펙 포함, 노출, 근거·조치
 ('T3', 'T3_기술셀부상예측/R3/build_base_pub.py', r"least\(nullif\(pub_date", '카운트(셀×국가×공개연도 출원 수) + 공개연도=least(pub,rgst)', 'US·EP·JP·CN', 'D1,D1E,D2(정의)', '예(US_*·EP_*·JP_*·CN_* 모멘텀·수준, fx·tot·kr_share)', '있음', 'V1·V1E·V2·V12 재빌드·재채점(sens_results.csv)'),
 ('T3', 'T3_기술셀부상예측/build_panel.py', r"count\(distinct appl_no\)|count\(\*\)", '카운트(appl_no 고유 → 같은 문서 두 번호면 2건)', 'US·EP', 'D1,D1E', '예', '있음', '같은 경로(build_panel 수정 없이 변형 base_rows로 재실행)'),
 ('T3', '_shared/ext/r3ros_01_family.py', r"'own' from rec", '등록번호 연결(US 자국 rgst_no → US 특허번호, RoS 논문 인용)', 'US', 'D2', '예(ros_sci·ros_ppp: xgbc·nb 성분)', '있음(t0=2021 원점만 own-only 0.30%)', 'V2r: own-only 연결 제거 후 ros_* 재계산·재채점'),
 ('T3', 'T3_기술셀부상예측/R3/build_ros_pubrule_r3d.py', r"group by 1,2,3", 'RoS 연결 (KSIC,mg,US특허) 단위 dedup', 'US', 'D1', '예', '없음(US특허 번호 단위 고유 집계)', 'D1 쌍은 같은 US 특허로 묶여 한 번만 셈'),
 ('T1', 'T1_산업대기실/final_run.py', r"where cc='EP' and appl_year", '카운트(법인 영문명 매칭 EP 출원 수: ep_k·ep_n·ep_share·ep_k_rec·ep_krprio)', 'EP', 'D1E', '예(R3E29 cols에 ep_* 5개)', '있음(ep_firm 2,026행·548개 법인)', 'V1E: ep_* 재계산 후 동결 학습·채점(P0 비트 재현)'),
 ('T1', 'T1_산업대기실/R2/r2ext.py', r"for nm, ccs in \(\('us'", '카운트(gl_us_g 등 해외 출원 모멘텀)', 'US·EP', 'D1,D1E', '아니오(ext 목록에 glob 없음)', '없음', 'R3 final_spec_r3.json ext=[io,flow,ipcsim,ipcfit,design,um,price,ipcx,ipcm,entfit,peer]'),
 ('T1', 'T1_산업대기실/R3/r3ext2.py', r"r3ros_family_us_pairs", 'RoS(KR 기록 family US 토큰)', 'KR', '-', '예(ipcx 등과 별개, ros는 KR 기록만)', '없음', "filters cc='KR' → own(US rgst_no) 경로 아님"),
 ('T4', 'T4_특허포기예측/s2/extract_family_dated.py', r"for tok in set\(f.split", '패밀리 토큰 개수(fam_n, KR 기록 family 필드)', 'KR(토큰은 US·EP·JP 번호)', 'FT', '예(fam_n)', '있음(FT): 시험 단위 39.9% 부풀림, 순위상관 0.9985', '정량만(s3f_t4_famn.json). 재학습은 범위 밖 → 후속 과제'),
 ('T4', 'T4_특허포기예측/R2/build_meta_cells.py', r"from read_parquet\('\{META\}'\) where appl_year", '카운트(tm_* 셀 모멘텀)', '5개국', 'D1,D1E', '아니오(final_spec_r2에 tm_ 없음)', '없음', 'spec grep tm_ 0건'),
 ('T4', 'T4_특허포기예측/extract_family.py', r"toks=set\(s.split", '패밀리 토큰 국가 수·이진(fam_cc·fam_us 등)', 'KR', 'FT', '예', '없음(국가 수·이진)', '-'),
 ('N4', 'N4_특허행태_기업폐업조기경보/n4lib.py', r"where cc in \('US','EP','JP','CN'\) and prio_cc", '해외 문서 수(fkey=cc+appl_no, fam_n_3y·fam_prio_2y)', 'US·EP·JP·CN', 'D1,D1E', '아니오(final_spec_r4 features에 fam_* 없음)', '없음', 'C14_foreign_family 묶음 미채택'),
 ('N5', 'N5_표준특허선언레이더/n5lib.py', r"where cc in \('US','EP','JP'\) and ksic=", '해외 문서 수(fam_n = fkey 고유 수)·국가 이진(fam_US/EP/JP)', 'US·EP·JP (KSIC 26429)', 'D1,D1E', '예(M1 묶음)', '있음', 'P0 재현 후 V1·V1E 재채점(sens_results.csv)'),
 ('N5', 'N5_표준특허선언레이더/n5lib.py', r"r3ros_family_us_pairs", 'RoS(KR 기록 family US 토큰)', 'KR', '-', '예(X_ros)', '없음', "KR 기록만 → own 경로 아님"),
 ('N6', 'N6_정부RnD기업_첫특허전환/n6_features.py', r"ntis_patent_link", 'KR 특허·NTIS', 'KR', '-', '-', '없음', '해외 메타 행 사용 없음'),
 ('N7', 'N7_특허기업_정부RnD첫수행/n7feat.py', r"right_list", 'KR 등록원부·NTIS', 'KR', '-', '-', '없음', '해외 메타 행 사용 없음'),
 ('N8', 'N8_특허분쟁표적레이더/n8lib.py', r"where cc='KR' group by appl_no", '패밀리 토큰 국가 수·이진(fam_offices·fam_us)', 'KR', 'FT', '예', '없음(nunique·이진)', 'final_spec_r6 fam_* = fam_offices, fam_us'),
 ('N9', 'N9_정부RnD과제_특허성과예측/n9_build.py', r"where cc='KR' and appl_year", 'KR 출원', 'KR', '-', '-', '없음', '해외 메타 행 사용 없음'),
 ('원고3.1 지형도·RTA·1위 수·ρ', '_shared_report/fusion_advantage/map5/m1_build_counts.py', r"select ksic, cc, appl_year, count\(\*\) n_all", '카운트(KSIC×국가×출원연도)', 'US·EP', 'D1,D1E', '-', '있음(출원 2000–09 창만)', 'map5·split34 사본을 V1·V1E 메타로 재실행(P0 원본 대조 차 0)'),
 ('원고3.1.1 동조', '_shared_report/fusion_advantage/map5/m3c_contemporaneous.py', r"FOR = \[", '카운트 연차 변화 상관', 'US·EP', 'D1,D1E', '-', '있음', 'V1E: EP 우선권 제거 0.0744→0.076'),
 ('원고3.1.2 RCA', '_shared_report/fusion_advantage/map5/m4_industry_quadrant.py', r"CCS = \[", 'RTA(P1·P2 창)', 'US·EP', 'D1,D1E', '-', '있음', 'V1E: 변화 ρ CI 하한 −0.265→−0.264'),
 ('원고3.1.3 분산 분해', '_shared_report/fusion_advantage/split34', None, '카운트 성장(2005–07 vs 2015–17)', 'EP', 'D1E', '-', '있음', 'V1E: 한국기원점유 변화 72.8→72.7%'),
 ('표2·표3 같은 발명 일치·재연결', '_shared_report/label_origin_R10/lo_fam.py', r"FROM tok JOIN kr USING", '출원번호 연결 쌍 수·일치율(해외 문서 행 단위)', 'US·EP·JP', 'D1,D1E', '-', '있음', 'V1: US 392,048→391,899쌍(87.3% 불변), V1E: EP 150,230→147,239쌍, 90.5→90.6%'),
 ('표2 행 수', '_shared_report/fusion_advantage/c5_R10/c1_assign.py', r"select cc, count\(\*\) from d", '행 수', 'US·EP', 'D1,D1E', '-', '사실(행 수)·고유 문서 수와 다름', 'warning: 주석 제안(US 고유 2,380,849, EP 799,548)'),
 ('표2·표3 KR 등록번호 연결', '_shared_report/label_origin_R10/lo_fam.py', r"WHERE cc='KR' AND rgst_no IS NOT NULL", '등록번호 연결(KR 등록번호)', 'KR', 'D2 해당 없음', '-', '없음(US rgst 미사용)', 'KR rgst_no만 사용'),
]
out_rows = []
for tr, f, rg, use, cc, df, spec, ex, note in ROWS:
    p = os.path.join(MC, f)
    lines = []
    if rg and os.path.isfile(p):
        for i, line in enumerate(open(p, encoding='utf-8', errors='replace'), 1):
            if re.search(rg, line): lines.append(str(i))
    elif os.path.isdir(p):
        lines = ['(폴더)']
    out_rows.append([tr, f, ';'.join(lines[:6]) if lines else ('MISSING' if not os.path.exists(p) else '-'), use, cc, df, spec, ex, note])
with open(os.path.join(W, 'exposure.csv'), 'w', encoding='utf-8-sig', newline='') as fh:
    w = csv.writer(fh); w.writerow(['track_or_section', 'file', 'lines', 'use_type', 'rows_country', 'defect', 'in_frozen_spec', 'exposure', 'evidence_action']); w.writerows(out_rows)
for r in out_rows: out(r[0], r[2], r[7])
