"""C09-T3 공용 경로·정규식 (읽기 전용 입력, 쓰기는 이 폴더만)."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
WD = os.path.dirname(HERE)                       # C09-T3
CACHE = os.path.join(WD, 'cache'); os.makedirs(CACHE, exist_ok=True)
ROOT = os.path.abspath(os.path.join(WD, '..', '..', '..'))   # 분석
T3 = os.path.join(ROOT, '모델경쟁_20260927', 'T3_기술셀부상예측')
R3 = os.path.join(T3, 'R3')
CP_DEV = os.path.join(R3, 'cache_pub')           # 공개연도 ≤2021
CP_FIN = os.path.join(R3, 'cache_pub_final')     # 공개연도 ≤2024 (최종 원점 피처는 ≤2021만 씀)
MANU = os.path.join(ROOT, '보고서_20261012', 'manuscript_v3.md')

def ddb(path):
    return path.replace('\\', '/')

# 법인형 이름 정규식(국가별, 데이터 확인 전용 — 판정에 쓰지 않음). duckdb regexp(RE2) 문법, 대소문자 무시는 (?i)
CORP_RE = {
    'US': r"(?i:\b(inc|corp|corporation|co|company|ltd|limited|llc|l\.l\.c|lp|llp|plc|gmbh|ag|sa|s\.a|nv|n\.v|bv|b\.v|kk|k\.k|ab|oy|spa|s\.p\.a|srl|sas|se|kg|pte|pty|holdings?|group|technolog(y|ies)|electronics|systems|industries|laborator(y|ies)|university|universit[aäé]t|institute|foundation|trust|association|government|council|centre|center|kabushiki|kaisha)\b)",
    'EP': r"(?i:\b(inc|corp|corporation|co|company|ltd|limited|llc|lp|llp|plc|gmbh|ag|sa|s\.a|nv|n\.v|bv|b\.v|kk|ab|oy|spa|s\.p\.a|srl|sas|se|kg|pte|pty|holdings?|group|technolog(y|ies)|electronics|systems|industries|laborator(y|ies)|university|universit[aäé]t|institute|foundation|association|centre|center|kabushiki|kaisha|aktiengesellschaft|gesellschaft|societe|société)\b)",
    'KR': r"(주식회사|\(주\)|㈜|유한회사|대학|연구원|연구소|재단|공사|공단|협회|조합|센터|\b(inc|corp|corporation|co|company|ltd|limited|llc|gmbh|ag|kk)\b|인크|코포레이션|컴퍼니|리미티드|게엠베하|가부시키|가부시끼|가이샤|카이샤|엘엘씨|에스에이|엔브이|비브이|아게|홀딩스|테크놀로지|일렉트로닉스|정부|대한민국|시청|도청|특허청|농촌진흥청)",
    'JP': r"(株式会社|有限会社|合同会社|会社|大学|研究所|研究機構|機構|法人|協会|組合|センター|コーポレーション|カンパニー|インコーポレイテッド|リミテッド|ゲーエムベーハー|エルエルシー|(?i:\b(inc|corp|corporation|co|company|ltd|limited|llc|gmbh|ag)\b))",
    'CN': r"(公司|大学|学院|研究院|研究所|研究中心|集团|厂|医院|中心|协会|局|部|委员会|株式会社|(?i:\b(inc|corp|corporation|co|company|ltd|limited|llc|gmbh|ag)\b))",
}
