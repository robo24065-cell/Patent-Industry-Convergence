# C10-T4 s10: sources_audit.csv, refs_audit.csv 작성(조회 결과를 표로 고정)
import csv, os, json, sys
sys.stdout.reconfigure(encoding="utf-8")
W = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C10\C10-T4"

S = [
 # row, source, table13_text, official_name_provider_id, terms_public, checked, used_in_results, verdict, fix
 ("제공 데이터", "특허-산업 융합데이터", "특허-산업 융합데이터(KSIC 50개, KR·US·JP·CN·EP), 지식재산처·한국특허전략개발원(KISTA). 15,141,159행. 공유 금지, 집계값만 수록",
  "공고문: '지식재산처 및 한국특허전략개발원의 데이터', KSIC 세세분류 50개 코드 특허 리스트", "공고문 p4 '데이터는 공유를 금지', 유의사항 8항, 서약서 6항(팀원 외 공유 금지·종료 후 폐기)",
  "공고문 PDF(cache\\notice_text.txt)", "전 트랙", "ok", ""),
 ("특허 권리", "특허 등록원부", "특허 등록원부(공공데이터포털, KIPRIS Plus)",
  "공공데이터포털 KIPRISPlus API 제공기관 '지식재산처'(관리 한국특허정보원)", "등록원부 API 페이지의 이용허락범위 값은 공개 검색으로 확인하지 못함(KIPRISPlus 계열은 월 1,000회 무료)",
  "WebSearch data.go.kr KIPRISPlus", "T4·N8·N4 등 결과·피처", "ok(조건 확인 불가)", "선택: 'KIPRIS Plus(지식재산처)'로 기관 병기"),
 ("특허 권리", "KIPRIS Plus 디자인·실용신안", "KIPRIS Plus 디자인·실용신안", "지식재산처 KIPRIS Plus", "위와 같음", "-", "T1 피처(99행, 그림 6 캡션)", "ok", ""),
 ("특허 권리", "사업자(출원인) 권리목록 business_rights", "(별도 행 없음, 'KIPRIS Plus'에 포함된 것으로 읽힘)", "KIPRIS Plus 출원인별 권리 목록(2026 스냅샷, 법인 53,967개)", "-",
  "_shared\\README.md §3", "4.4-4(410행) 시점 정합 한계, N4·N7·T1 연결", "ok(경미)", "선택: 특허 권리 행에 '출원인별 권리목록' 추가"),
 ("정부 R&D", "NTIS Open API", "NTIS Open API 활용(국가과학기술지식정보서비스 과제·성과검색, 2026-09-29 수집; 비상업, 제3자 제공 금지)",
  "NTIS(과학기술정보통신부·KISTI) Open API 과제검색·성과검색(rpatent)", "공개 안내 페이지(rndopen/api/mng/apiMain.do)에는 상업·제3자 조항이 없음(신청 승인형). 프로젝트 기록 ntis_SOURCES_B.md: '비상업·내부 분석, 결과물에 \"NTIS Open API\" 활용 명시, 제3자 제공 금지'",
  "WebFetch ntis.go.kr; _shared\\ext\\ntis_SOURCES_B.md", "N9·N6·N7 결과·그림 7", "ok(공개 페이지로 조건 원문 확인 불가, 승인 조건 기록과 일치)", ""),
 ("기업", "국민연금 가입 사업장 내역", "국민연금공단 가입 사업장 내역(공공데이터포털)",
  "data.go.kr 15083277 '국민연금공단_국민연금 가입 사업장 내역_20260923', 제공 국민연금공단", "이용허락범위 제한 없음", "WebFetch data.go.kr/data/15083277", "3.6.2 표 9·그림 8, N4·N7", "ok", ""),
 ("기업", "국세청 사업자 상태조회", "국세청 사업자등록 상태조회 API(공공데이터포털)",
  "data.go.kr 15081808 '국세청_사업자등록정보 진위확인 및 상태조회 서비스', 제공 국세청", "이용허락범위 제한 없음", "WebFetch data.go.kr/data/15081808", "N4 결과(99·289행)", "ok(약칭)", ""),
 ("기업", "OpenDART", "OpenDART(금융감독원)", "금융감독원 OpenDART", "약관 제16조 저작권 금감원, 제23조 공시정보 정확성 보장 안 함; 상업·재배포 조항 없음(공공데이터법 준거)",
  "WebFetch opendart.fss.or.kr/intro/terms.do", "표 9·그림 8 DART 업종", "ok", ""),
 ("기업", "KIND", "KIND(한국거래소)", "한국거래소 KIND 상장·상장폐지 현황", "웹 조회 미실시(로그인 없는 공개 조회 기록: R7_SOURCES)", "_shared\\ext\\R7_SOURCES.md", "323·342·359행 KIND 3자리", "ok", ""),
 ("산업·무역", "KOSIS 광업·제조업조사·광공업생산지수", "KOSIS(통계청 광업·제조업조사(10인 이상, 명목), 광공업생산지수)",
  "KOSIS DT_1FS1001(10차 조사기반 2008–2020)·DT_1FS1101(11차 등록기반 2020–2024); 작성기관 통계청 → 2024년 기준부터 국가데이터처 명의", "출처 표시(공공누리 1유형 예시: 서울 열린데이터 광업·제조업 페이지)",
  "WebSearch(CiNii 서지: 통계청 –2023년 기준, 국가데이터처 2024년 기준–)", "3.1.3·그림 5, T3 피처", "warning(낮음·명칭)", "'통계청(현 국가데이터처)'로 표기(글자 +7)"),
 ("산업·무역", "ECOS 산업연관표·물가", "ECOS(한국은행 산업연관표·물가)", "한국은행 ECOS", "웹 조회 미실시(공고문 붙임5 지정 출처)", "공고문 붙임5", "T1 피처, 그림 6", "ok", ""),
 ("산업·무역", "IO-KSIC-HS 대조표", "한국은행 「IO코드와 KSIC 및 HS코드 대조표」(2020년 기준)",
  "로컬 파일 시트 'IO코드와 KSIC 비교표'(제목 '2020 IO상품_KSIC 비교표')·'IO코드와 HS코드 비교표'", "공개 페이지에서 원 파일·발행기관을 찾지 못함(기존 편집 메모 D-3 '내려받은 URL 기록 없음'과 같음)",
  "WebSearch; scripts\\s6_iomap.py", "표 3·표 5·T1", "확인 불가(기존 미처리 항목)", "출처 URL 확보 시 그대로 유지"),
 ("산업·무역", "UN Comtrade", "UN Comtrade(HS6 수출입)", "UN Comtrade Database(UNSD), public preview API", "UN 저작권; 사전 서면 허가 없는 재배포·출판·상업 이용 금지(이용약관 5절). 원고는 집계 지표(RCA·순위상관)만 실음",
  "WebFetch comtrade.un.org/licenseagreement.html", "3.1.2·표 5, T3·T1 피처", "ok(경미: 조건 미기재)", "선택: '원자료 재배포 불가, 집계만' 병기"),
 ("표준·과학·가치", "ETSI IPR 데이터베이스", "ETSI IPR 데이터베이스", "ETSI IPR Online Database 공개 추출본(ISLD-export.zip, ETSI SR 000 314)", "면책: 선언된 IPR의 정확성·유효성·필수성에 대해 ETSI는 보증하지 않음, 서지는 선언자 제공",
  "WebSearch ipr.etsi.org·SR 000 314", "N5 결과(선언)", "ok", "원고는 '선언'으로만 써서 필수성 주장 없음"),
 ("표준·과학·가치", "Reliance on Science", "Reliance on Science(Marx & Fuegi, CC BY-NC 4.0, 비상업)",
  "Zenodo 10.5281/zenodo.22967630 'Reliance on Science'(2026-09-25, 작성 Marx, Matt), 파일 pcs_oa_uspto.csv", "라이선스 cc-by-nc-4.0 확인. 인용 요청: 인용 데이터를 쓰면 두 논문(Marx & Fuegi 2020 SMJ 41(9):1572–1594; Marx & Fuegi 2022 JEMS 31(2):369–392, doi 10.1111/jems.12455)을 모두 인용",
  "Zenodo API records/22967630; Crossref 10.1111/jems.12455", "T3 최종 피처 ros_sci·ros_ppp(본문 인용 bodyonly 포함 confscore≥3), N5 피처", "warning(조건 일부 누락)", "참고문헌에 Marx & Fuegi (2022) JEMS 추가 또는 표 13 RoS 괄호에 '(Marx & Fuegi 2020·2022)'"),
 ("표준·과학·가치", "OpenAlex", "OpenAlex", "OpenAlex(OurResearch)", "CC0(인용 의무 없음)", "WebFetch help.openalex.org", "피처 후보(최종 모형 제외, 99행)", "ok", ""),
 ("표준·과학·가치", "KPSS", "KPSS 특허 가치 자료(Kogan·Papanikolaou·Seru·Stoffman)", "GitHub KPSS2017/...Extended-Data, KPSS_2025(2026-09-09판)", "라이선스 표기 없음; README가 Kogan et al. (2017) QJE 132(2):665–712 인용을 안내 → 참고문헌에 있음",
  "GitHub API·README", "N8 참고 로지스틱 비교 모형, 최종 모형 제외(99행)", "ok", ""),
 ("분류·보정", "KSIC 해설서", "통계분류포털 KSIC 해설서", "통계분류포털(kssc) 한국표준산업분류 해설서(10차)", "공고문 붙임5 지정 출처", "공고문 붙임5", "TXT·TXT-FREE 대조군", "ok", ""),
 ("분류·보정", "IPC-KSIC 연계표(2013)", "IPC-KSIC 연계표(2013, 대조군)", "20131002_IPC-KSIC.xlsx(KSIC 9차 기준). 공공데이터포털 15044348 '지식재산처_KIPRISPlus_KSIC_IPC 연계표'(R7_SOURCES 기록)", "작성기관 미표기",
  "_shared\\ext\\R7_SOURCES.md", "대조군 전반(1.1·2.4·3.6)", "warning(낮음·기관 누락)", "'IPC-KSIC 연계표(특허청, 2013)' 등 작성기관 병기"),
 ("분류·보정", "WIPO IP 통계", "WIPO IP 통계(공개건수 보정)", "WIPO IP Statistics Data Center 지표(wipo-data-patent-indicators), WIPO IPC–기술분야 대응표(ipc_technology.xlsx)", "-",
  "_shared\\ext\\quality_SOURCES.md", "2.5 보정(143행), 168행 'WIPO 35개 분야로 옮기면 … 0.45'", "warning(낮음·용도 누락)", "'WIPO IP 통계·IPC 기술분야 대응표(공개건수 보정, 35개 분야 비교)'"),
 ("(누락)", "KSIC 9↔10↔11 산업직업분류 연계표", "없음",
  "data.go.kr 15159305 '국가데이터처 국가데이터연구원_국가데이터처_산업직업분류 연계표_20260430', 제공 국가데이터처 국가데이터연구원(수정 2026-05-06)", "공공저작물: 출처표시, 상업적 이용금지, 변경금지(제4유형)",
  "WebFetch data.go.kr/data/15159305; 사용 코드: T3 ESSENTIAL_R7 conc_lib.py(10→9차), confirm_R12 r12_reg.py·foreign_public_R11 fp_tests.py·T1 TXT_R9 s1_firm.py(KIND 업종명→KSIC)",
  "136·300행(T3 연계표 대체 0.788→0.589), 323행(KIND 3자리 +1.9%p), 342·359행(해외 KIND −0.33%p), 부록 R7-T3-CONC·R12-3D·R11-M-KIND", "warning(정의 불일치: 출처표시 의무 자료 누락)",
  "표 13 분류·보정 행에 'KSIC 신구 연계표(국가데이터처, 공공데이터포털 15159305, 공공누리 제4유형)' 추가"),
 ("(누락)", "출원인-법인 연결표 CORP_APPLICANT", "없음(표 3 91행에는 '출원인-법인 연결표 385,256행'으로 등장)",
  "공모전데이터\\외부데이터\\Corporate_20260720\\CORP_APPLICANT.txt(출원인코드·출원인명·법인번호·사업자번호). 프로젝트 문서에 원천·이용 조건 기록 없음('KIPRIS 공개정보' 언급 1회)", "확인 불가",
  "_shared\\README.md §2, 부록 DATA-CORPAPP", "표 3, 1.4·4.4-4, N4·N6·N7·N8·N9·T1·R12 기업 연결의 입구", "warning(출처 미기재·원천 미확인)", "원천 확인 후 표 13 기업 행에 '출원인-법인 연결표(<제공기관>, 2026-07)' 추가. 원천 확인은 메인 세션"),
 ("(사용 안 함 확인)", "KIPRIS 상표·공휴일·벤처기업명단·OECD·NBER·ISO·TTBD", "없음", "-", "-",
  "트랙 DATA_NOTES 9개, 원고 grep", "원고 결과에 쓰이지 않음(N6 '벤처기업 목록 미사용', 상표는 N4 한계 문장·NTIS 성과 IP 유형 안, 공휴일은 C09-T4 루프 점검만)", "ok(누락 아님)", ""),
]
with open(os.path.join(W, "sources_audit.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["table13_row", "source", "table13_text(manuscript 425–431행)", "official_name_provider_id", "terms_public_or_record", "checked_via", "used_in_results", "verdict", "proposed_fix"])
    w.writerows(S)

R = [
 ("Balassa 1965", "Balassa, B. (1965). Trade liberalisation and \"revealed\" comparative advantage. The Manchester School, 33(2), 99–123.", "10.1111/j.1467-9957.1965.tb00050.x",
  "Balassa, Bela; Trade Liberalisation and “Revealed” Comparative Advantage; The Manchester School 33(2) 99-123; print 1965-05", "일치", "본문에 저자-연도 인용 없음(RCA 172·174·183·185·342·357행)", "ok(본문 인용 표지 없음은 경고)"),
 ("Benjamini & Hochberg 1995", "Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. Journal of the Royal Statistical Society: Series B, 57(1), 289–300.", "10.1111/j.2517-6161.1995.tb02031.x",
  "Benjamini, Yoav; Hochberg, Yosef; Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing; JRSS Series B 57(1) 289-300; 1995", "일치(부제 생략·관행)", "본문 인용 없음('다중검정' 163·372행)", "ok(경미)"),
 ("Chen & Guestrin 2016", "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD, 785–794.", "10.1145/2939672.2939785",
  "Chen, Tianqi; Guestrin, Carlos; XGBoost (A Scalable Tree Boosting System); Proc. 22nd ACM SIGKDD 785-794; 2016-08-13", "일치(학회명 약칭)", "본문 인용 없음('XGB' 216행)", "ok(경미)"),
 ("Granger 1969", "Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. Econometrica, 37(3), 424–438.", "10.2307/1912791",
  "Granger, C. W. J.; Investigating Causal Relations by Econometric Models and Cross-spectral Methods; Econometrica 37(3) 시작쪽 424(Crossref는 시작쪽만); 1969-08", "일치(끝쪽 438은 Crossref에 없음, JSTOR 서지 관행과 같음)", "본문 인용 없음('그랜저형' 163행)", "ok"),
 ("Kogan et al. 2017", "Kogan, L., Papanikolaou, D., Seru, A., & Stoffman, N. (2017). Technological innovation, resource allocation, and growth. Quarterly Journal of Economics, 132(2), 665–712.", "10.1093/qje/qjw040",
  "Kogan; Papanikolaou; Seru; Stoffman; Technological Innovation, Resource Allocation, and Growth; QJE 132(2) 665-712; print 2017-05", "일치", "표 13(430행)에서 저자명으로 대응", "ok"),
 ("Marx & Fuegi 2020", "Marx, M., & Fuegi, A. (2020). Reliance on science: Worldwide front-page patent citations to scientific articles. Strategic Management Journal, 41(9), 1572–1594.", "10.1002/smj.3145",
  "Marx, Matt; Fuegi, Aaron; Reliance on science: Worldwide front‐page patent citations to scientific articles; SMJ 41(9) 1572-1594; print 2020-09(online 2020-04)", "일치", "표 13(430행)에서 대응", "ok"),
 ("Priem et al. 2022", "Priem, J., Piwowar, H., & Orr, R. (2022). OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. arXiv:2205.01833.", "arXiv:2205.01833",
  "arXiv API: Jason Priem, Heather Piwowar, Richard Orr; 같은 제목; 제출 2022-05-04", "일치", "본문 인용 없음(표 13 'OpenAlex')", "ok(경미)"),
 ("(데이터 제공자 요청 인용, 목록에 없음) Marx & Fuegi 2022", "없음(편집 메모 E: 'Marx & Fuegi 2022 뺌')", "10.1111/jems.12455",
  "Marx, Matt; Fuegi, Aaron; Reliance on science by inventors: Hybrid extraction of in‐text patent‐to‐article citations; JEMS 31(2) 369-392; 2022(online 2021-09-23)", "-", "Zenodo 22967630이 인용 데이터 사용 시 두 논문 인용을 요청. T3 피처가 본문 인용(bodyonly) 포함", "warning(조건 누락)"),
]
with open(os.path.join(W, "refs_audit.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ref", "manuscript_text(434–441행)", "doi_or_id", "public_metadata(Crossref/arXiv)", "biblio_match", "in_text_correspondence", "verdict"])
    w.writerows(R)
print("ok", len(S), len(R))
