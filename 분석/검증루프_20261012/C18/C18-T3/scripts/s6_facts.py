# C18-T3 s6(C16-T4 s3_facts.py에서 생성): facts 단일 패치(C14 facts_sync + C15 facts_patch + C12–C15 ERRATA facts 항목 + C15 부록 반영분 동기)
# 원본 facts는 읽기만. 작업 폴더 facts_work\ 사본에 적용 후 현재 원고(v19 b316df8e)와 숫자 토큰 재대조.
import csv, io, os, re, sys, json, shutil, hashlib
sys.stdout.reconfigure(encoding='utf-8')
AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
R = AN + r"\보고서_20261012"; FD = R + r"\facts"; L = AN + r"\검증루프_20261012"
OUT = L + r"\C18\C18-T3"; WK = OUT + r"\facts_work"
os.makedirs(WK, exist_ok=True)
HDR = ['id', '주제', '수치', '단위', '정확한 문장 조각', '출처 파일 경로', '조건·경고', 'note']
SRC_SHA = {}
def load(name):
    p = FD + '\\' + name
    raw = open(p, 'rb').read(); SRC_SHA[name] = hashlib.sha1(raw).hexdigest()[:8]
    txt = raw.decode('utf-8-sig').replace('\r\n', '\n')
    rows = list(csv.reader(io.StringIO(txt)))
    hdr, body = rows[0], rows[1:]
    out = []
    for i, r in enumerate(body, start=2):
        if len(r) != len(hdr):
            out.append(dict(_line=i, _bad=r))
        else:
            d = dict(zip(hdr, r)); d['_line'] = i; out.append(d)
    return hdr, out
H1, FA = load('facts.csv'); H2, SU = load('facts_supplement.csv'); H3, R12 = load('facts_supplement_R12.csv')
orig_snapshot = json.dumps([FA, SU, R12], ensure_ascii=False)
ms = open(R + r"\manuscript_v3.md", encoding='utf-8').read().replace('\r\n', '\n').split('\n')
st = next(i for i, s in enumerate(ms) if s.startswith('# [부록')); en = next(i for i, s in enumerate(ms) if s.startswith('# [편집 메모'))
APP = {}
for i in range(st, en):
    s = ms[i]
    if s.startswith('| ') and not s.startswith('| ---'):
        c = [x.strip() for x in s.strip().strip('|').split(' | ')]
        if len(c) >= 4 and c[0] != 'id':
            key = re.sub(r'\s*\((보충|보충 R12)\)$', '', c[0])
            APP[key] = dict(line=i + 1, sec=c[1], val=' | '.join(c[2:-1]) if len(c) > 4 else c[2], src=c[-1])
def app(i): return APP[i]['val']
def appl(i): return APP[i]['line']

TABLE = {'facts.csv': FA, 'facts_supplement.csv': SU, 'facts_supplement_R12.csv': R12}
def row(f, i):
    for r in TABLE[f]:
        if r.get('id') == i: return r
    return None
log = []   # consolidated rows
def rec(f, i, col, before, after, why, rnd, kind):
    log.append(dict(파일=f, id=i, 열=col, 현재값=before, 새값=after, 근거=why, 출처바퀴=rnd, 종류=kind))
def op_set(f, i, col, new, why, rnd, expect_old=None):
    r = row(f, i); assert r is not None, (f, i)
    old = r[col]
    if expect_old is not None and old != expect_old:
        why += f' [주의: 현재값이 예상({expect_old})과 다름]'
    if old == new:
        rec(f, i, col, old, new, why + ' [이미 같음]', rnd, '변경 없음'); return
    r[col] = new; rec(f, i, col, old, new, why, rnd, '설정')
def op_rep(f, i, col, a, b, why, rnd):
    r = row(f, i); assert r is not None, (f, i)
    old = r[col]
    if a not in old:
        rec(f, i, col, old, old, why + f' [실패: 원문에 "{a}" 없음]', rnd, '실패'); return
    r[col] = old.replace(a, b); rec(f, i, col, old, r[col], why, rnd, '치환')
def op_app(f, i, col, add, why, rnd):
    r = row(f, i); assert r is not None, (f, i)
    old = r[col]
    if add in old:
        rec(f, i, col, old, old, why + ' [이미 있음]', rnd, '변경 없음'); return
    r[col] = (old + ' / ' if old else '') + add; rec(f, i, col, old, r[col], why, rnd, '덧붙임')
def op_add(f, d, why, rnd):
    if row(f, d['id']) is not None:
        rec(f, d['id'], '(행)', '있음', '있음', why + ' [이미 있음]', rnd, '변경 없음'); return
    nd = {h: d.get(h, '') for h in HDR}; nd['_line'] = None
    TABLE[f].append(nd); rec(f, d['id'], '(행 추가)', '', ' | '.join(nd[h] for h in HDR if nd[h]), why, rnd, '행 추가')
FS = 'facts.csv'; SP = 'facts_supplement.csv'
C14 = 'C14(facts_sync)'; C14E = 'C14(ERRATA)'; C15 = 'C15(facts_patch/ERRATA)'; C16 = 'C16(부록 반영분 동기)'; C18 = 'C18(C16·C17 facts·v18–v19 부록 동기)'
SY = r'검증루프_20261012\C14\C14-T2\facts_sync.csv'
# ---- C14 facts_sync 1–4 (조건 열) ----
op_rep(FS, 'MAP-KR-RTA', '조건·경고', 'ρ 0.20–0.36', 'ρ 0.19–0.37(동률 규칙에 따라 최대 0.48; C11-T2 대안 A·verify_5)', SY + ' 2행; E-C12T2-11; 원고 168행', C14)
for i in ['B-P50-CW27', 'B-P50-TXT50', 'B-P50-FREE50']:
    op_rep(FS, i, '조건·경고', 'AUC 0.002~0.004', 'AUC 0.002~0.005(공개 대안 다섯, C10-T2)', SY + '; E-C12T2-11; 원고 13·407행', C14)
op_app(FS, 'B-P50-FREE50', '조건·경고', '이원 군집: 법인×대상 업종은 모든 추정량에서 0 포함(CGM [−0.0002, +0.0049]); 법인×서브클래스는 곱 가중 [−0.0006, +0.0057]만 0 포함(CGM [+0.0003, +0.0045]) — C14-T1',
       SY + " 5행 정정안('법인×서브클래스 CI 0 포함')은 C14-T1(나중 검증 통과)로 대체; 원고 334행 '법인×대상 업종'", 'C14(T1 우선, 충돌 해소)')
# ---- 5–34 옛값 ----
op_set(FS, 'LINK-IO', '정확한 문장 조각', app('LINK-IO'), SY + ' 6행; 원고 부록 %d행' % appl('LINK-IO'), C14)
op_set(FS, 'T3NC-MG', '정확한 문장 조각', app('T3NC-MG'), SY + ' 7행; 부록 %d행' % appl('T3NC-MG'), C14)
op_set(FS, 'T3NC-prov_vs_txt', '정확한 문장 조각', '제공 − TXT +0.0017 [−0.0004, +0.0038], 6/9 (같다(경계))', SY + ' 8행; 부록 %d행' % appl('T3NC-prov_vs_txt'), C14)
op_set(FS, 'T3NC-prov_vs_txt', '수치', '+0.0017', SY + ' 8행', C14, expect_old='+0.0020')
op_app(FS, 'T3NC-prov_vs_txt', '조건·경고', '연결 정의에 따라 +0.0014~+0.0026, 상속 없음이면 CI 하한>0', SY + ' 8행', C14)
op_rep(FS, 'T3NC-txt_vs_none', '정확한 문장 조각', '−0.0023 [−0.0041, −0.0005], 2/9', '−0.0020 [−0.0039, −0.0003], 2/9', SY + ' 9행(facts 전용 행)', C14)
op_rep(FS, 'T3EXT-prov-txt-사업체수', '정확한 문장 조각', '[+0.053, +0.332], 7/9, BH q 0.150', '[+0.053, +0.333], 7/9, BH q 0.250(seed에 따라 0.05–0.25)', SY + ' 10행(facts 전용 행)', C14)
op_set(FS, 'T3EXT-prov-txt-종사자수', '정확한 문장 조각', '제공 − TXT: ρ 0.099 대 −0.0065, 차이 +0.106 [−0.029, +0.234], 7/9, BH q 0.865', SY + ' 11행(facts 전용 행)', C14)
op_rep(FS, 'T3EXT-34', '정확한 문장 조각', 'TXT 0.040, +0.115 [+0.0004, +0.220]', 'TXT 0.044, +0.111 [+0.0043, +0.214]', SY + ' 12행(facts 전용 행)', C14)
op_set(FS, 'C-LAND-free_F', '정확한 문장 조각', app('C-LAND-free_F'), SY + ' 13행 + E-C15T4-A1(부록 482행 v17)', 'C14+C15')
op_app(FS, 'C-LAND-free_F', '조건·경고', '최빈 동률 규칙 의존(표 동률만 반대 0.473, 배정까지 반대 0.476, 균등 분할 0.469; C11-T2 verify_5 v5_out.json·v5_frac.json)', SY + ' 13행(0.469) + E-C15T4-A1(0.476)', 'C14+C15')
op_set(FS, 'C-LAND-txt_F', '정확한 문장 조각', app('C-LAND-txt_F'), SY + ' 14행; 부록 %d행' % appl('C-LAND-txt_F'), C14)
op_set(FS, 'C-LAND-free_H', '정확한 문장 조각', 'KR제공+해외FREE H: 한국 RTA>1 21개, 한국 1위 10개, 제공 1위 7개 중 유지 4, 한국 RTA ρ(제공 대비) 0.311', SY + ' 15행(facts 전용 행)', C14)
op_set(FS, 'C-LAND-free_H', '수치', '10', SY + ' 15행', C14, expect_old='9')
op_app(FS, 'C-LAND-free_H', '조건·경고', '동률 반대 0.369, 균등 분할 0.355(C11-T2 verify_5)', SY + ' 15행', C14)
op_rep(FS, 'C-LAND-txt_H', '정확한 문장 조각', '한국 RTA>1 28개, 한국 1위 20개, 제공 1위 7개 중 유지 5, 한국 RTA ρ(제공 대비) 0.225', '한국 RTA>1 27개, 한국 1위 20개, 제공 1위 7개 중 유지 5, 한국 RTA ρ(제공 대비) 0.228', SY + ' 16행(facts 전용 행)', C14)
for i, oldn, newn in [('C-AGREE-US', '51.0', '50.5'), ('C-AGREE-JP', '48.2', '48.0'), ('C-AGREE-EP', None, None)]:
    op_set(FS, i, '정확한 문장 조각', app(i), SY + ' %s행 + E-C14T2-03(부록 %d행, PKR 대안 A)' % ({'C-AGREE-US': 17, 'C-AGREE-JP': 18, 'C-AGREE-EP': 20}[i], appl(i)), C14)
    if oldn: op_set(FS, i, '수치', newn, SY, C14, expect_old=oldn)
    op_app(FS, i, '출처 파일 경로', r'분석\검증루프_20261012\C10\C10-T3\res\s2_agree.csv; 분석\검증루프_20261012\C14\C14-T2\pkr_ipc_altA.csv', 'E-C14T2-03 출처 칸 보충', C14)
op_app(FS, 'C-AGREE-CN', '출처 파일 경로', r'분석\검증루프_20261012\C14\C14-T2\pkr_ipc_altA.csv', SY + ' 19행(값 불변, 출처만)', C14)
op_set(FS, 'C-AGREE-RANGE', '정확한 문장 조각', app('C-AGREE-RANGE'), SY + ' 21행; 부록 %d행' % appl('C-AGREE-RANGE'), C14)
op_set(FS, 'C-AGREE-RANGE', '수치', '48–53', SY + " 21행('48–54'는 이중 반올림)", C14, expect_old='48–54')
# C-AGREE-UB → C-AGREE-PURE (행 대체)
r = row(FS, 'C-AGREE-UB')
if r is not None:
    before = ' | '.join(r[h] for h in HDR if r[h])
    r['id'] = 'C-AGREE-PURE'; r['수치'] = '52.3/49.4/55.0'; r['정확한 문장 조각'] = app('C-AGREE-PURE')
    r['출처 파일 경로'] = r'분석\검증루프_20261012\C10\C10-T3\res\s2_agree.csv(free5_pure)'
    r['조건·경고'] = (r['조건·경고'] + ' / ' if r['조건·경고'] else '') + "결함 상한 보정값 C-AGREE-UB(52.3/49.3/55.2, 결과 파일 없음)를 대체(E-C10T3-04)"
    rec(FS, 'C-AGREE-UB→C-AGREE-PURE', '(행 대체)', before, ' | '.join(r[h] for h in HDR if r[h]), SY + ' 22행; E-C10T3-04; 부록 %d행' % appl('C-AGREE-PURE'), C14, '행 대체')
op_set(FS, 'C-RCA-TXT', '정확한 문장 조각', app('C-RCA-TXT'), SY + ' 23행; 부록 %d행' % appl('C-RCA-TXT'), C14)
op_set(FS, 'C-RCA-FREE-IPC', '정확한 문장 조각', app('C-RCA-FREE-IPC'), SY + ' 24행; 부록 %d행' % appl('C-RCA-FREE-IPC'), C14)
op_set(FS, 'C-RCA-FREE-IPC', '수치', '0.608', SY + ' 24행', C14, expect_old='0.571')
op_rep(FS, 'C-RCA-SUB', '정확한 문장 조각', 'FREE −0.015', 'FREE −0.024', SY + ' 25행(facts 전용 행; CW-eq +0.091은 duckdb 동점 순서에 따라 +0.087)', C14)
op_rep(FS, 'C-ABROAD-N', '조건·경고', 'FREE-IPC 팔은 이 중 8.9%(5,002건)가 결함 연결로 배정(R10 §6-1 문서 계산, 결과 파일 없음)', 'FREE-IPC 팔은 올바른 등록번호 연결 상속 31.1%(공개번호 53.9%); 결함 연결 8.9%(R10 §6-1)는 폐기', SY + ' 26행', C14)
# C15-T3 facts_patch (C14 27·30·33·34·35 대체)
FP = r'검증루프_20261012\C15\C15-T3\facts_patch.csv'
op_set(FS, 'C-T3-AUCS', '정확한 문장 조각', app('C-T3-AUCS'), FP + ' 1행(E-C15T3-F) = 부록 519행 v17; C14 facts_sync 27행 표시안 대체', C15)
op_app(FS, 'C-T3-AUCS', '조건·경고', 'hyb_pkr 0.7755는 대안 A 등록번호 연결(상속 없음 0.7739, R8 결함 연결 0.7735); 재빌드 잡음 ±0.0002(C15-T3)', FP, C15)
op_app(FS, 'C-T3-AUCS', '출처 파일 경로', r'분석\검증루프_20261012\C15\C15-T3\res\s5_t3_summary.json; 분석\검증루프_20261012\C10\C10-T3\res\s6_t3_summary.json', FP + '; E-C12T2-08', C15)
op_set(FS, 'C-T3-RET', '정확한 문장 조각', app('C-T3-RET'), FP + ' 2행 = 부록 638행 v17; C14 33행 대체', C15)
op_set(FS, 'C-T3-RET', '수치', '29/23', FP + ' 2행', C15, expect_old='29/25')
op_app(FS, 'C-T3-RET', '조건·경고', 'hyb_pkr 67%는 대안 A(상속 없음 62%, R8 결함 연결 61%)', FP, C15)
op_set(FS, 'C-T3-drop_ros-hyb_pkr', '정확한 문장 조각', '제공 − hyb_pkr(진단) +0.0113 [+0.0061, +0.0164], 8/9', FP + ' 3행; C14 30행 표시안 대체', C15)
op_set(FS, 'C-T3-drop_ros-hyb_pkr', '수치', '+0.0113', FP + ' 3행', C15, expect_old='+0.0133')
op_app(FS, 'C-T3-drop_ros-hyb_pkr', '조건·경고', '대안 A 등록번호 연결(상속 없음 +0.0129 [+0.0075, +0.0181] 9/9, R8 결함 연결 +0.0133 [+0.0079, +0.0185] 9/9); 2011 원점 차 −0.00088은 빌드 잡음 크기, 승수 9/9 가능', FP + ' 3행 + ERRATA E-C15T3-F 권고', C15)
op_set(FS, 'C-T3-drop_ros-hyb_txt', '정확한 문장 조각', app('C-T3-drop_ros-hyb_txt'), SY + ' 28행; 부록 %d행' % appl('C-T3-drop_ros-hyb_txt'), C14)
op_set(FS, 'C-T3-drop_ros-hyb_txt', '수치', '+0.0242', SY + ' 28행', C14, expect_old='+0.0239')
op_set(FS, 'C-T3-drop_ros-hyb_free', '정확한 문장 조각', app('C-T3-drop_ros-hyb_free'), SY + ' 29행; 부록 %d행' % appl('C-T3-drop_ros-hyb_free'), C14)
op_set(FS, 'C-T3-drop_ros-hyb_free', '수치', '+0.0262', SY + ' 29행', C14, expect_old='+0.0255')
op_rep(FS, 'C-T3-hyb_txt-kr_rebuild', '정확한 문장 조각', '+0.0100 [+0.0072, +0.0131], 9/9', '+0.0097 [+0.0069, +0.0127], 9/9', SY + ' 31행(facts 전용 행)', C14)
op_set(FS, 'C-T3-hyb_txt-kr_rebuild', '수치', '+0.0097', SY + ' 31행', C14, expect_old='+0.0100')
op_rep(FS, 'C-T3-hyb_free-kr_rebuild', '정확한 문장 조각', '+0.0084 [+0.0038, +0.0129], 8/9', '+0.0077 [+0.0030, +0.0124], 8/9', SY + ' 32행(facts 전용 행)', C14)
op_set(FS, 'C-T3-hyb_free-kr_rebuild', '수치', '+0.0077', SY + ' 32행', C14, expect_old='+0.0084')
op_set(FS, 'C-RCA-PKR', '정확한 문장 조각', "PKR-IPC(진단) ρ 0.537 [0.280, 0.747], 차이(제공−PKR) +0.007 [−0.027, +0.048] '같다'", FP + ' 4행; C14 34행 표시안 대체', C15)
op_set(FS, 'C-RCA-PKR', '수치', '0.537', FP + ' 4행', C15, expect_old='0.548')
op_app(FS, 'C-RCA-PKR', '조건·경고', '대안 A 등록번호 연결(상속 없음 0.542·+0.002, R8 결함 연결 0.548·−0.003), 판정 같다·정합 유지', FP, C15)
op_rep(FS, 'C-LAND-pkr_H', '정확한 문장 조각', 'KR제공+해외PKR H(진단):', 'KR제공+해외PKR H(진단, 대안 A 등록번호 연결):', FP + ' 5행; C14 35행 대체', C15)
op_rep(FS, 'C-LAND-pkr_H', '정확한 문장 조각', 'ρ(제공 대비) 0.869', 'ρ(제공 대비) 0.874', FP + ' 5행', C15)
op_app(FS, 'C-LAND-pkr_H', '조건·경고', '상속 없음 0.866, R8 결함 연결 0.869(C15-T3)', FP, C15)
op_rep(FS, 'C-T3-COVD', '정확한 문장 조각', 'hyb_txt 0.11, hyb_free 0.36, hyb_pkr 1.02', 'hyb_txt 0.12, hyb_free 0.41, hyb_pkr 1.02', FP + ' 6행(facts 전용; C15-T3 verify_4 확인)', C15)
op_set(FS, 'C-T3-COVD', '수치', '0.12', FP + ' 6행', C15, expect_old='0.11')
op_app(FS, 'C-T3-COVD', '조건·경고', '대안 A 등록번호 연결 캐시(R8 결함 연결 0.11/0.36/1.02, hyb_pkr 상속 없음 1.03)', FP, C15)
# 단서 미동기(C14 36–52) → 조건 열에 v17 부록 표기
for i, n in [('OFF-T3-TOP10', 36), ('OFF-T3-TOP100', 37), ('OFF-T3-NEWIPC', 38), ('OFF-N5-SKILL', 39), ('OFF-N7-SB', 40), ('LBL-CW-GROUP', 41), ('A-COVER', 42),
             ('MAP-SYNC', 44), ('MAP-rho_patRTA_expRCA_P2', 45), ('MAP-rho_patRTAchange_expRCAchange_0817', 46), ('DATA-CORPAPP', 49), ('LINK-HS6-EXP', 50)]:
    op_app(FS, i, '조건·경고', '원고 부록 표기(v19 %d행): %s' % (appl(i), app(i)), SY + ' %d행(단서 미동기)' % n, C14)
op_app(SP, 'X-T3-FUT', '조건·경고', '원고 부록 표기(v19 %d행): %s' % (appl('X-T3-FUT'), app('X-T3-FUT')), SY + ' 52행(단서 미동기)', C14)
op_set(FS, 'DATA-REG-EVENTS', '정확한 문장 조각', app('DATA-REG-EVENTS'), SY + ' 48행; 부록 %d행' % appl('DATA-REG-EVENTS'), C14)
op_set(FS, 'DATA-REG-EVENTS', '수치', '88,323', SY + ' 48행(권리이전 정의 정정)', C14, expect_old='122,974')
for i, n in [('MAP-KR-TOP', 43), ('DATA-REGISTRY', 47), ('LINK-T1-R3', 51)]:
    rec(FS, i, '-', '', '', SY + ' %d행: 부록 셀 축약(…), facts가 더 완전 → 변경 없음' % n, C14, '변경 없음')
# 보충 행(C14 53–59 + C14·C15 ERRATA)
def S(i, sec, val, src, cond='', subj='검증루프 보충', num='', unit=''):
    return {'id': i, '주제': subj, '수치': num, '단위': unit, '정확한 문장 조각': val, '출처 파일 경로': src, '조건·경고': cond, 'note': 'C16-T4 통합 패치(%s)' % sec}
op_add(SP, S('C11-ROS-LINK', '3.2, 4.4', 'RoS·KPSS 연결 번호 정밀도(같은 family) 99.0%(순열 0.9%), 같은 발명 69%(수동 100쌍); 초대형 패밀리 제거(남은 연결 같은 발명 97%) 최종 +0.0024 [−0.0005, +0.0052], 개발 +0.0016; 같은 우선일 연결(남은 쌍 같은 발명 71%) 최종 +0.0015',
             r'분석\검증루프_20261012\C11\C11-T3\sens_results.csv, link_audit.csv'), 'E-C11T3-03(ERRATA 771행); ' + SY + ' 53행', C14)
op_add(SP, S('C12-T3-SAB', '3.2.2', '단일 출원인 부상(증가의 절반 이상) 41%: 이름 205/504(40.7%)·코드 205·공동출원 전원 209(41.5%), 빼도 skill 0.4378/0.4378/0.4445, 미등록 대치 41.6% [41.5, 41.9]·0.4447; 과반 기준 19.6–20.4%, 증가 5건 이상 셀 22%',
             r'분석\검증루프_20261012\C12\C12-T3\a_concentration.csv'), 'E-C12T3-F1; ' + SY + ' 54행', C14)
op_add(SP, S('C12-T3-IPI-BREAK', '2.5, 3.2.1, 표 13', '광공업생산지수 9→10차(2015-01) 이음매 성장률 차 평균 −0.0023(|·| 0.0046); T3 최종 seed 0–4 짝 Δ: 걸침 NaN +0.0009, 9차 구계열 접속 +0.0017, 생산지수 묶음 제거 +0.0082(RoS 제거 +0.0041)',
             r'분석\검증루프_20261012\C12\C12-T3\b_sens.csv, verify_1\v1_summary.csv'), 'E-C12T3-F2; ' + SY + ' 55행', C14)
op_add(SP, S('C12-N5-SEED', '2.2, 4.4', 'N5 동결 스펙 시드 3개 묶음 21개 skill 0.126–0.169, M0 대비 하한>0 5/21; 키 완전판 S1 0.153 [−0.024, 0.366]; S2 상위5사 제외 0.073 [−0.118, 0.223]; S4 2011 이관 제외 0.115 [−0.052, 0.293]; 강한 기준선 대비 모두 0 포함',
             r'분석\검증루프_20261012\C12\C12-T1\sens_results.csv, verify_0\v1_agg_off_*.json'), 'E-C12T1-F1; ' + SY + ' 56행', C14)
op_add(SP, S('C13-SEED6', '2.2, 2.3, 4.1, 4.4', app('C13-SEED6'),
             r'분석\검증루프_20261012\C13\C13-T2\seed_bundles.csv, variance_decomp.csv; 분석\검증루프_20261012\C15\C15-T2\seed_bundles.csv, variance_decomp.csv',
             cond='고정한 것: 부트스트랩 seed·T4 학습 프레임 표본·N9 텍스트 분류기 시드(학습 시드 몫만); 하한 이동은 N8·N5 최대 약 0.05, 나머지 0.01 이하; N9만 공식이 9묶음 중 최고(+0.004), T3는 공식이 최솟값; T4 상위 1% 적중은 시드 민감(40.7–48.4%, 1/8 강화 기준선 역전); SE = CI폭/3.92(재표본 SD 기준이면 T1 0.092)'),
       'E-C13T2-F1(' + SY + ' 57행)의 값은 C15 확장(E-C15T2-01, 부록 534행 v17)으로 대체 — 나중 바퀴 검증 통과값 우선', 'C13→C15')
op_add(SP, S('C13-N8-CAPTURE', '3.5, 4.2', 'N8 결과 원천 공식 대비 포착률(존속 몫 보정) 2008–22 0.479·2023–24 0.367(비 0.766 [0.650, 0.897], 2023 범위 밖 급증 보정 0.94), 무효/권리범위 1.54, 원부 내부 포착 99–100%, 예고등록 직권 의무(등록령 제6조①4·제19조②); 레이블 민감도 S1 0.413(Δ+0.004)·S3 무효 단독 0.481',
             r'분석\검증루프_20261012\C13\C13-T1\capture_rate.csv, sens_results.csv'), 'E-C13T1-F1; ' + SY + ' 58행', C14)
op_add(SP, S('C14-PKR-ALTA', '부록 C-AGREE', '진단용 PKR-IPC 대안 A(등록번호 연결 상속): US 71.9%, JP 73.8%, CN 68.0%, EP 72.8%(R8 결함 연결 68.5/71.1/68.0/68.8%)',
             r'분석\검증루프_20261012\C14\C14-T2\pkr_ipc_altA.csv', cond='상속 없는 IPC 표만이면 70.2/72.6/68.0/71.2%'), SY + ' 60행', C14)
op_add(SP, S('B-C14T1-2WAY', '3.6.3', '제공−TXT-FREE ΔAUC +0.0024 이원 CI: 법인×sc CGM [+0.0003, +0.0045] / 곱 가중 [−0.0006, +0.0057]; 법인×대상 업종 CGM [−0.0002, +0.0049] / 곱 가중 [−0.0009, +0.0057]; 법인×업종 하한은 통합 가중에 민감(역분산 +0.0001)',
             r'분석\검증루프_20261012\C14\C14-T1\se_compare.csv'), 'E-C14T1-F1', C14E)
op_add(SP, S('C14-T1-SURV', '2.5, 4.4', 'T1 범위밖 권리목록 생존 편향: 범위안 아날로그 누락 15.3%(넘긴 55.5%·보유 1.2%, 진입 7.3%·비진입 11.6%); 최종 Δskill S1 −0.0008·S2 −0.0009·S3 −0.0069·S4 −0.0012·AN_B−A +0.0001(문턱 0.0087, 측정 변형 기준·상한 아님)',
             r'분석\검증루프_20261012\C14\C14-T3\analog_dropout.csv, sens_results.csv'), 'E-C14T3-F1', C14E)
# C15 ERRATA facts
op_set(FS, 'DATA-KR-REG', '정확한 문장 조각', app('DATA-KR-REG'), 'E-C15T4-F; 부록 503행 v17; registry_full.sqlite(mode=ro) 10: 615,042·20: 41,318', C15)
op_add(SP, S('C15-T4-CAPTURE', '2.5, 3.5', 'T4 결과 원천 공식 대조(연보 존속·등록 재고-유량): 결과창 포착률 1.054(불납 하한 0.928), 학습창 코호트 0.988, 원부 내부 누락 0.096%, 실용신안 혼입 0',
             r'분석\검증루프_20261012\C15\C15-T1\capture_rate_T4.csv', cond='연보에 사유별 소멸 표 없음 → 전 사유 이탈 비교, 불납은 하한'), 'E-C15T1-F1', C15)
op_add(SP, S('C15-NTIS-HORIZON', '4.2', '정부 R&D 기원 대 비기원 포기: 5년 14.2/21.0%, 7년 37.7/35.2%, 10년(등록 2007–15) 55.5/52.2%; 6년차 기한 소멸 13.1/7.1%',
             r'분석\검증루프_20261012\C15\C15-T1\cache\s4_horizon.csv', cond='소멸등록일 기준(regdate 열); 실효일 경계 포함이면 5년 격차 0.7%p; 출원인 유형 표준화 뒤 6–10년 기원 +1.7–2.9%p'), 'E-C15T1-F2', C15)
op_add(SP, S('C15-SEED-T1T3', '2.2, 2.3, 4.1, 4.4', 'T1·T3 학습 시드 10묶음: T1 skill 0.472–0.475·M0 하한 0.453–0.456·SBB 0.392–0.399(하한 0.376–0.384)·상위 1000 298–316; T3 skill 0.423–0.432·하한 0.366–0.376·SB3 0.182–0.201(하한 0.124–0.143)·상위 100 49–52; 하한>0 20/20, 강화 우위 20/20',
             r'분석\검증루프_20261012\C15\C15-T2\seed_bundles.csv, variance_decomp.csv', cond='피처 생성·로지스틱·부트스트랩 seed 고정; T1 DEV2FINAL 4묶음 +0.124~+0.126'), 'E-C15T2-F1', C15)
# C16: C15 부록 반영분 중 facts 미동기(조건 열에 v17 부록 표기)
for i in ['OFF-T1-SKILL', 'OFF-T3-SKILL', 'OFF-T1-TOP1000', 'OFF-DEV2FINAL', 'NTIS-T3-SUM']:
    op_app(FS, i, '조건·경고', '원고 부록 표기(v19 %d행): %s' % (appl(i), app(i)), 'C15 ERRATA 반영(E-C15T2-02·03, E-C15T1-02)으로 부록만 바뀜 → facts 조건 열 동기', C16)
# 형식(R12 7·12행)
fixed = 0
for r in R12:
    if '_bad' in r:
        b = r['_bad']; k = len(b) - len(H3) + 1
        # 따옴표 없는 쉼표는 마지막 note 칸('원고는 +7.3 [+5.1, +9.7]' 등)에 있음 → note로 합침
        merged = b[:len(H3) - 1] + [','.join(b[len(H3) - 1:])]
        before = ','.join(b)
        d = dict(zip(H3, merged)); d['_line'] = r['_line']; r.clear(); r.update(d); fixed += 1
        rec('facts_supplement_R12.csv', d['id'], '(형식)', f'필드 {len(b)}개', f'필드 {len(merged)}개(note 칸 따옴표: {merged[-1]})', 'E-C14T2-F2; ' + SY + ' 61행', C14E, '형식')


# ==== C18 추가: v18·v19 부록 변경 동기 + C16·C17 facts 항목(PREREG 6절) ====
ms_sha = hashlib.sha1(open(R + r"\manuscript_v3.md", "rb").read()).hexdigest()
assert ms_sha.startswith("b316df8e"), ms_sha
# (1) E-C16T3-F + v18 부록 변경(LINK-NTIS·C-COUNTRY-SHARE) 동기
op_set(FS, 'DATA-KR-APP', '정확한 문장 조각', '제공 KR 출원 1,087,511건(특허 1,036,864·실용신안 50,647)', 'E-C16T3-F; C16-T3 kr_prefix.csv(C18 재계산 일치)', C18)
op_set(FS, 'LINK-NTIS', '정확한 문장 조각', app('LINK-NTIS'), 'E-C16T3-01 반영(부록 %d행 v19) 동기' % appl('LINK-NTIS'), C18)
op_app(FS, 'LINK-NTIS', 'note', '분모·분자 실용신안 포함(실용신안 연결 83건); 특허만 9.78%·등록 11.67%(C16-T3)', 'E-C16T3-F', C18)
op_app(FS, 'LBL-CW-5D', 'note', '분모 KR 출원 1,087,511은 실용신안 포함(C16-T3)', 'E-C16T3-F(선택)', C18)
op_set(FS, 'C-COUNTRY-SHARE', '정확한 문장 조각', app('C-COUNTRY-SHARE'), 'E-C16T3-02 반영(부록 %d행 v19) 동기' % appl('C-COUNTRY-SHARE'), C18)
# (2) E-C16T1-F1: C16-T1-VIS(부록 535행 값을 문장 조각으로, 4자리 값은 조건 열)
op_add(SP, S('C16-T1-VIS', '3.3.1, 4.4', app('C16-T1-VIS'), r'분석\검증루프_20261012\C16\C16-T1\variants.csv',
             cond='원점 가시 모집단 S1-full·L-a 0.4607 [0.4423, 0.4767](독립 −0.0112, 짝지은 Δ CI [−0.0192, −0.0054]), S1-fixed 0.4671(짝지은 CI 0 포함), S2 0.4713, L-b 0.5453; 의존 쌍 7.30%; M0·SBB 분류 불변. S1만 고친 판이며 원점 정보 전체 결합판은 C17-T1-CMB(본문 410행 0.464–0.471)'),
       'E-C16T1-F1 + E-C17T1-F1 비고 권고; 부록 %d행 v19' % appl('C16-T1-VIS'), C18)
# (3) E-C17T1-F1: C17-T1-NEWC(부록 행 있음)·RC·CMB
op_add(SP, S('C17-T1-NEWC', '3.3.1, 4.4', app('C17-T1-NEWC'), r'분석\검증루프_20261012\C17\C17-T1\variants.csv, verify_0\v0_out.json, verify_1\v1_out.json',
             cond='T1-02 신규진입자 1,304법인 모두 범위안 첫 등록 2020–25(원점 이후); 원점에 날짜 확인되는 범위안 기록 232곳(주 정의 하한; 비모호 owner 매칭 최대 245, 범위밖 등록 포함 1,197); N1(재계산) skill 0.4781 [0.4601, 0.4962]; N2 +0.0064; N3 −0.0006~−0.0053(공여 점수, 범위 보고용); 무작위 동수 제외 +0.0052로 같음(구성 효과)'),
       'E-C17T1-F1(verify 반영 수정: 232곳 정의·무작위 동수); 부록 %d행 v19' % appl('C17-T1-NEWC'), C18)
op_add(SP, S('C17-T1-RC', '3.3.1, 4.4', 'T1-05 RC:001 법인 308곳(위험쌍 14,801·양성 6): R1 재계산 Δ +0.0056 [+0.0037, +0.0081], 고정 +0.0001, 경계 [−0.052, +0.0087]',
             r'분석\검증루프_20261012\C17\C17-T1\variants.csv',
             cond='경계는 기준선 재구축 값 고정 시(기준선도 극단이면 [−0.0553, +0.0356]); Δ는 기준선 약화 몫(RC 행 자체 약 +0.00004); RC 행 무작위 대입 +0.0005~+0.0019'),
       'E-C17T1-F1(verify 반영 수정)', C18)
op_add(SP, S('C17-T1-CMB', '3.3.1, 4.4', '원점 정보 결합(S1-full+N1+R1) skill 0.4711 [0.4518, 0.4909] Δ −0.0016 [−0.0110, +0.0087](고정 0.4696); S1+N1 0.4642(재계산)·0.4689(고정); 원점 정보판 범위 0.464–0.471',
             r'분석\검증루프_20261012\C17\C17-T1\variants.csv', cond='본문 410행 0.464–0.471의 원천(C18 재계산 일치); 모두 영향 없음, M0·SBB 분류 유지'),
       'E-C17T1-F1', C18)
# (4) E-C17T2-F1(E-C16T1-F2 대체) + E-C17T2-F2
op_add(SP, S('C17-T2-ALIAS', '3.3.2, 4.4', app('C17-T2-ALIAS'), r'분석\검증루프_20261012\C17\C17-T2\sens_results.csv, cache\s2_out.json',
             cond='이름 경로 전용 양성 2,990쌍: 코드 판정 1,001(권리목록 979 오매칭 0, 등록원부 22 중 오매칭 20), 주소 플래그 208쌍 중 오매칭 추정 112; 오매칭률 E2 5.0% [4.0, 6.3](V7 4.1%, V8a 4.2%, E1 4.9%, 위약 보정 4.4%); Δskill(제외) +0.0066 [0.0041, 0.0087], (음성 전환) +0.0065, 하한 +0.0002, V9 +0.0087, 유리 결합 +0.0036; R 코드 확정 오매칭 121쌍(같은 계보 의심 20쌍), 되돌림 Δ(iii) −0.0001~+0.0011(양의 값은 공개 전용 양성 16쌍 처리 때문, 그중 5쌍은 진입 문서도 다른 코드 출원), 확률판 −0.0001~+0.0013, 결합 Δ(iv) +0.0067~+0.0077; kappa 3범주 0.987(판정 가능 134쌍 0.853, 오매칭 n=4). C16-T1-ALIAS 표본 320 수치(+0.0112/+0.0014/+0.0125)는 C16 표본 수치로만 남기고 본문 근거로 쓰지 않음(E-C17T2-F2). 상한 6.3% > 5%라 \'정합\'으로 쓰지 않음'),
       'E-C17T2-F1(수정판: R 보조·kappa) + E-C17T2-F2; 부록 %d행 v19' % appl('C17-T2-ALIAS'), C18)
# (5) E-C17T3-F1: C17-T3-NEWIPC(부록 행 있음)·C17-NTIS-COUNT·C17-PIT-MAP-V2(조건부). '혼합' 해석 문장은 verify 반박이라 뺌.
op_add(SP, S('C17-T3-NEWIPC', '3.2.2, 4.4', app('C17-T3-NEWIPC'), r'분석\검증루프_20261012\C17\C17-T3\t3_newipc_sens.csv; cache\t3_sens_out.json',
             cond='IPC 2021.01에 없고 이후 판에만 있는 메인그룹 90셀(하위클래스 신설 72·메인그룹 신설 18; H10*·B64U 48셀 포함): 제외(레이블 유지) 0.406 [0.349, 0.463], 제외+레이블 재선정 0.380 [0.321, 0.438], 구 IPC 되돌림 0.400 [0.341, 0.458]; 개발 원점 평균 Δ −0.006/−0.005/−0.008. 동결 예측 재채점(재학습 없음). 본문 409행 0.38–0.42 하한의 원천(H10만 빼거나 합친 C05-T1 0.391–0.416과 합침). 개발 대비 \'혼합\' 해석은 verify 반박으로 쓰지 않음(E-C17T3-D4)'),
       'E-C17T3-F1(facts_candidates.csv 1행, 반박된 혼합 문장 제거); 부록 %d행 v19' % appl('C17-T3-NEWIPC'), C18)
op_add(SP, S('C17-NTIS-COUNT', '3.4', 'NTIS 2026 스냅샷 과제 수 대 공식: 2018–21 1.00000, 2022 0.99996, 2023 0.99993, 2024 0.99995(원문 확인: KIST 2022·2023 PDF, 2024 보도); 금액 포착률 2010–24 0.985–1.005',
             r'분석\검증루프_20261012\C17\C17-T3\cache\ntis_count_check.csv; cache\ntis_amount_check.csv',
             cond='과제 수 공식값은 2018–24만, 2010–17은 금액으로 대체. 본문 257행 \'과제 수는 2018–24년 공식 통계와 같고\'의 원천(C18 재계산 0.99993–1.00000)'),
       'E-C17T3-F1(facts_candidates.csv 2행)', C18)
op_add(SP, S('C17-PIT-MAP-V2', '2.1, 2.5, 4.4(원고 숫자 기재 금지)', '시점 정합 지도 확정판: 트랙 86요소 A 16·B 11(B1 9, B2 2)·C 48(C* 12)·D 7·E 2·NA 2, 트랙 밖 2(C 1·NA 1); 트랙 B·C 59 중 v18 원고 공개됨 38·부분 21·미공개 0',
             r'분석\검증루프_20261012\C17\C17-T3\pit_map_v2.csv; grade_table.csv; grade_rule.txt',
             cond='조건부(정의 의존): 과제 한 줄 정의를 쓰면 B 33. 코더 kappa 86행 0.602·88행 0.596. 두 코더·재판정자는 같은 에이전트. 원고에 숫자로 쓰지 않음. E-C16T2-F1(C16-PIT-MAP) 대체'),
       'E-C17T3-F1(facts_candidates.csv 3행, 조건부)', C18)
# (6) v19 본문 409행 숫자 원천 보충(H10 몫 45%·범위 0.38–0.42)
op_add(SP, S('C18-T3-H10-RANGE', '4.4', 'H10 셀 45개가 부상 504셀의 4%이나 순증가의 45%(개발 원점 0–21%); H10만 빼거나 합쳐 재채점 0.391–0.416(C05-T1 S1–S3), 원점 이후 신설 90셀 제외+재선정 0.380(C17-T3) → 본문 0.38–0.42',
             r'분석\검증루프_20261012\C05\C05-T1\rescore_scenarios.csv; 분석\검증루프_20261012\C17\C17-T3\t3_newipc_sens.csv',
             cond='공식 0.423 불변; 0.38은 H10 외 신설 셀(G06V·B64U·메인그룹 신설)을 포함한 90셀 재선정판(문장 주어 H10과 대상 다름 — 선택 E-C17T3-01 권장판이 정확)'),
       'C18-T3 원천 재검(recheck_numbers.csv E-C17T3-01)', C18)

# 쓰기(작업 사본)
def write(name, hdr, rows):
    rawo = open(FD + '\\' + name, 'rb').read()
    bom = rawo.startswith(b'\xef\xbb\xbf'); crlf = b'\r\n' in rawo
    with open(WK + '\\' + name, 'w', encoding='utf-8-sig' if bom else 'utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\r\n' if crlf else '\n'); w.writerow(hdr)
        for r in rows: w.writerow([r.get(h, '') for h in hdr])
write('facts.csv', H1, FA); write('facts_supplement.csv', H2, SU); write('facts_supplement_R12.csv', H3, R12)
assert json.dumps([FA, SU, R12], ensure_ascii=False) != orig_snapshot
with open(OUT + r"\facts_patch_v19.csv", 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(log[0].keys())); w.writeheader(); w.writerows(log)
# 원본 불변 확인
for name in SRC_SHA:
    assert hashlib.sha1(open(FD + '\\' + name, 'rb').read()).hexdigest()[:8] == SRC_SHA[name]
from collections import Counter
print('patch rows', len(log), Counter(x['종류'] for x in log), Counter(x['출처바퀴'] for x in log))
for x in log:
    if x['종류'] in ('실패',) or '주의' in x['근거']: print('!!', x)
json.dump(dict(src_sha=SRC_SHA, n=len(log), kinds=Counter(x['종류'] for x in log), r12_fixed=fixed), open(OUT + r"\cache\s6_out.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
