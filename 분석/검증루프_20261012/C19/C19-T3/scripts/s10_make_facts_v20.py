# C19-T3 s10: C18-T3 s6_facts.py(facts_patch_v19 생성기)를 복사해 E-C18T3-F 결함 3가지를 고친 생성기 s10_facts_v20_gen.py를 만든다(복사본만 수정, 원본 읽기만).
#  (1) LINK-IO: 부록 축약 셀 덮어쓰기(op_set app) → 앞부분만 E-C12T2-11 문구로 치환(op_rep), 뒷부분 보존
#  (2) 행 추가: 수치·단위 칸을 채우고, 패치 csv '새값'을 헤더 8칸 그대로의 CSV 한 줄로 기록
#  (3) 순환 18행: s11 원천 대조 결과(facts_work/circular_evidence.csv)를 근거 칸에 붙이고, 근거 미확보 행은 패치에서 뺀다
#  + v20 원고(40406e29) 기준 부록 변경 동기(원천 대조된 값만) + 대안 B의 V1E 조건 주석
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
src = open(C18T3 + r'\scripts\s6_facts.py', encoding='utf-8').read()
s = src
def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a[:80], s.count(a))
    s = s.replace(a, b)
rep('OUT = L + r"\\C18\\C18-T3"; WK = OUT + r"\\facts_work"', 'OUT = L + r"\\C19\\C19-T3"; WK = OUT + r"\\facts_work"')
rep('assert ms_sha.startswith("b316df8e"), ms_sha', 'assert ms_sha.startswith("40406e29"), ms_sha')
# (1) LINK-IO
rep("op_set(FS, 'LINK-IO', '정확한 문장 조각', app('LINK-IO'), SY + ' 6행; 원고 부록 %d행' % appl('LINK-IO'), C14)",
    "op_rep(FS, 'LINK-IO', '정확한 문장 조각', '한국은행 산업연관표 2010(통합소분류 165)', '한국은행 2010–2015–2020 접속 산업연관표의 2010년 표(2020 분류, 통합소분류 165; 파일 수정 2026-06-30)', SY + ' 6행; E-C12T2-11(앞부분만, 뒷부분 \\'KSIC→310), 제품코드 수 비례 가중)\\' 보존 — E-C18T3-F(1)); 원천 C10-T1 data_check.txt [6]·verify_2 v2_io_guide.txt, T1 R2 ext_link_summary.csv', C14)")
# 부록 표기 라벨 v19 → v20(현재 원고 줄 번호)
s = s.replace("'원고 부록 표기(v19 %d행): %s'", "'원고 부록 표기(v20 %d행): %s'")
s = s.replace("(부록 %d행 v19)", "(부록 %d행 v20)").replace("부록 %d행 v19'", "부록 %d행 v20'")
# (2) 행 추가: 수치·단위 채우기 + 새값 8칸 CSV 한 줄
rep("""def op_add(f, d, why, rnd):
    if row(f, d['id']) is not None:
        rec(f, d['id'], '(행)', '있음', '있음', why + ' [이미 있음]', rnd, '변경 없음'); return
    nd = {h: d.get(h, '') for h in HDR}; nd['_line'] = None
    TABLE[f].append(nd); rec(f, d['id'], '(행 추가)', '', ' | '.join(nd[h] for h in HDR if nd[h]), why, rnd, '행 추가')""",
"""def _csvline(vals):
    b = io.StringIO(); csv.writer(b, lineterminator='').writerow(vals); return b.getvalue()
NUMUNIT = {  # C19: 행 추가 22행의 수치·단위(조각 안의 대표 숫자; 아래 assert로 조각 포함 확인)
 'C11-ROS-LINK': ('99.0', '%'), 'C12-T3-SAB': ('41', '%'), 'C12-T3-IPI-BREAK': ('−0.0023', '성장률 차'), 'C12-N5-SEED': ('0.126–0.169', 'skill'),
 'C13-SEED6': ('76/76', '묶음'), 'C13-N8-CAPTURE': ('0.479/0.367', '포착률'), 'C14-PKR-ALTA': ('71.9/73.8/68.0/72.8', '%'), 'B-C14T1-2WAY': ('+0.0024', 'ΔAUC'),
 'C14-T1-SURV': ('15.3', '%'), 'C15-T4-CAPTURE': ('1.054', '포착률'), 'C15-NTIS-HORIZON': ('14.2/21.0', '%'), 'C15-SEED-T1T3': ('0.472–0.475/0.423–0.432', 'skill'),
 'C16-T1-VIS': ('0.461', 'skill'), 'C17-T1-NEWC': ('+0.0055', 'Δskill'), 'C17-T1-RC': ('+0.0056', 'Δskill'), 'C17-T1-CMB': ('0.4711', 'skill'),
 'C17-T2-ALIAS': ('5.0', '%'), 'C17-T3-NEWIPC': ('0.406/0.380/0.400', 'skill'), 'C17-NTIS-COUNT': ('0.99993/1.00000', '비율'), 'C17-PIT-MAP-V2': ('86', '요소'),
 'C18-T3-H10-RANGE': ('0.38–0.42', 'skill'), 'C18-T1-REGGAP': ('−19.5/−47.3', '%'),
}
def _num_in_frag(num, frag):
    f = frag.replace('−', '-')
    return all(p.replace('−', '-').lstrip('+') in f for p in re.split(r'[/–]', num) if p)
def op_add(f, d, why, rnd):
    if row(f, d['id']) is not None:
        rec(f, d['id'], '(행)', '있음', '있음', why + ' [이미 있음]', rnd, '변경 없음'); return
    nd = {h: d.get(h, '') for h in HDR}; nd['_line'] = None
    if d['id'] in NUMUNIT and not nd['수치']:
        nd['수치'], nd['단위'] = NUMUNIT[d['id']]
    assert nd['수치'] and nd['단위'], ('수치·단위 빈칸', d['id'])
    assert _num_in_frag(nd['수치'], nd['정확한 문장 조각']), ('수치가 조각에 없음', d['id'], nd['수치'])
    TABLE[f].append(nd); rec(f, d['id'], '(행 추가)', '', _csvline([nd[h] for h in HDR]), why + ' [새값 = 헤더 8칸 CSV 한 줄: ' + ','.join(HDR) + ']', rnd, '행 추가')""")

# (2b) 행 대체도 새값을 8칸 CSV 한 줄로(이전: 빈칸을 뺀 ' | ' 결합)
rep("    rec(FS, 'C-AGREE-UB→C-AGREE-PURE', '(행 대체)', before, ' | '.join(r[h] for h in HDR if r[h]), SY + ' 22행; E-C10T3-04; 부록 %d행' % appl('C-AGREE-PURE'), C14, '행 대체')",
    "    rec(FS, 'C-AGREE-UB→C-AGREE-PURE', '(행 대체)', _csvline([before_r.get(h, '') for h in HDR]), _csvline([r[h] for h in HDR]), SY + ' 22행; E-C10T3-04; 부록 %d행 [현재값·새값 = 헤더 8칸 CSV 한 줄]' % appl('C-AGREE-PURE'), C14, '행 대체')")
rep("    before = ' | '.join(r[h] for h in HDR if r[h])\n    r['id'] = 'C-AGREE-PURE'", "    before = ' | '.join(r[h] for h in HDR if r[h]); before_r = dict(r)\n    r['id'] = 'C-AGREE-PURE'")
# (3)+v20 동기+V1E 주석: 쓰기 직전에 C19 블록 삽입
C19BLOCK = r'''
# ==== C19 추가: v20(40406e29) 부록 변경 동기 — s11 원천 대조(circular_evidence.csv) 통과 값만 ====
C19 = 'C19(v20 부록 동기·원천 대조)'
EV = r'C19\C19-T3\facts_work\circular_evidence.csv'
op_set(FS, 'LINK-KOSIS5', '정확한 문장 조각', app('LINK-KOSIS5'), 'E-C07T1-08 반영(부록 %d행 v20); 원천 C07-T1 official_compare.csv·T1 R2 ext_link_summary.csv(' % appl('LINK-KOSIS5') + EV + ')', C19)
op_set(FS, 'LINK-KOSIS5', '수치', '50/50', '부록 %d행 v20: 10·11차 결합 50/50(2008년 값만 49/50)' % appl('LINK-KOSIS5'), C19, expect_old='49/50')
op_app(SP, 'R11-PKR', '조건·경고', '상속 없는 IPC 표, 결함 연결 비의존(C14-T2)', 'E-C14T2-05 반영(부록 %d행 v20) 동기; 원천 R11 res_main.csv PROV-PKR 값 불변(' % appl('R11-PKR') + EV + ')', C19)
op_app(FS, 'NTIS-T3-SUM', '조건·경고', '출원인 유형·등록연도·KSIC 표준화 시 격차 1.4–3.5%p(C02-T1 _tab_e_lapse.json standardized)', r'E-C02T1-07 반영(부록 %d행 v20) 동기; 원천 C02\C02-T1\_tab_e_lapse.json(' % appl('NTIS-T3-SUM') + EV + ')', C19)
op_add(SP, S('C18-T1-REGGAP', '4.4', app('C18-T1-REGGAP'), r'분석\검증루프_20261012\C18\C18-T1\monthly_capture.csv, sens_results.csv',
             cond='공백 달 2024-01~08·2025-02~08; 수집 결함 아님(수집 목록 = 제공 등록번호, 원 CSV 공란); 개인 등록 누락은 권리목록(법인 전용)으로 복원 불가 — 보충은 법인·외국 한정 하한(C18-T1 verify_2)'),
       'E-C18T1-B1 반영(부록 %d행 v20) 동기; 원천 C18-T1 monthly_capture.csv·sens_results.csv(' % appl('C18-T1-REGGAP') + EV + ')', C19)
# ==== C19 추가: 대안 B — 제공 메타 US·EP 이중 수록(V1E) 조건 주석(값은 제공 데이터 그대로 유지; V1E 값은 C19-T3 diff_numbers.csv·cache/s9_values.json) ====
import json as _json
_V = _json.load(open(OUT + r'\cache\s9_values.json', encoding='utf-8'))['V1E']
C19B = 'C19(대안 B: V1E 조건 주석)'
_B = '제공 데이터 그대로(번호 형식 이중 수록 포함) 계산'
_D = 'US 7/8자리 2,099쌍·EP 짧은/긴 형식 7,357쌍 이중 수록 제거(V1E, 긴 형식 유지) 시'
_SRC = r'C19\C19-T3\diff_numbers.csv'
op_app(FS, 'CIRC-AGREE-CC', '조건·경고', f"{_B}. {_D} US {_V['US_agree']*100:.1f}%, JP {_V['JP_agree']*100:.1f}%, EP {_V['EP_agree']*100:.1f}%(EP만 끝자리 변화; 범위 표기 87–91%)", _SRC, C19B)
op_app(FS, 'CIRC-LINKS', '조건·경고', f"{_B}. {_D} US {_V['US_pairs']:,}, JP {_V['JP_pairs']:,}, EP {_V['EP_pairs']:,}쌍", _SRC, C19B)
op_app(FS, 'MAP-SYNC', '조건·경고', f"{_B}. {_D} 우선권 제거 후 EP {_V['ep_purged']:.3f}(순열 p ≤ {_V['p_max']:.3f}); 나머지 불변", _SRC, C19B)
op_app(FS, 'MAP-rho_patRTAchange_expRCAchange_0817', '조건·경고', f"{_B}. {_D} CI 하한 {_V['m4_ci_lo']:.3f}(ρ·상한 불변)", _SRC, C19B)
op_app(FS, 'SP-VAR-한국기원점유_변화pp', '조건·경고', f"{_B}. {_D} {_V['krorig']*100:.1f}%(p {_V['krorig_p']:.4f}); KR 출원 성장 56.1%·5개국 29.1% 불변", _SRC, C19B)
op_app(FS, 'SP-FCST', '조건·경고', f"{_B}. {_D} 쌍 순서 {_V['pairacc']:.3f}, RMSE {_V['rmse']:.4f} [{_V['ci_lo']:.4f}, {_V['ci_hi']:.4f}], L=2 KR +{_V['kr_l2']:.4f}(8/8·0/4·0.523 불변)", _SRC, C19B)
op_app(FS, 'MAP-LEADLAG', '조건·경고', f"{_B}. {_D} 원 p<0.05는 방향마다 {_V['n_fwd']}개(BH 통과 0개 불변)", _SRC, C19B)
op_app(FS, 'DATA-META', '조건·경고', '행 수는 제공 데이터 그대로(번호 형식 이중 수록 포함): 고유 문서 기준 US 2,380,849(−2,099)·EP 799,548(−7,357), 합계 15,131,703', r'C19\C19-T3\cache\s0_v1e_check.json', C19B)
op_app(FS, 'C-COUNTRY-SHARE', '조건·경고', '행 수는 이중 수록 포함(고유 문서 US 2,380,849·EP 799,548; 비중 표기 불변)', r'C19\C19-T3\cache\s0_v1e_check.json', C19B)
# ==== C19 추가: 순환 18행 원천 대조 표시(근거 칸) ====
import csv as _csv
_ev = {}
for _r in _csv.DictReader(open(OUT + r'\facts_work\circular_evidence.csv', encoding='utf-8-sig')):
    _ev.setdefault(_r['id'], []).append(_r)
CIRC18 = ['LINK-IO', 'T3NC-MG', 'C-LAND-free_F', 'C-LAND-txt_F', 'C-AGREE-US', 'C-AGREE-JP', 'C-AGREE-EP', 'C-AGREE-RANGE', 'C-RCA-TXT', 'C-RCA-FREE-IPC',
          'C-T3-AUCS', 'C-T3-RET', 'C-T3-drop_ros-hyb_txt', 'C-T3-drop_ros-hyb_free', 'DATA-REG-EVENTS', 'DATA-KR-REG', 'LINK-NTIS', 'C-COUNTRY-SHARE']
DROPPED = []
for _i in CIRC18:
    rows_ = _ev.get(_i, [])
    n_ok = sum(1 for r in rows_ if r['근거'] != '없음'); n = len(rows_)
    for x in log:
        if x['id'] == _i and x['열'] == '정확한 문장 조각':
            if n and n_ok == n:
                x['근거'] += f' [C19 원천 대조: 숫자 토큰 {n_ok}/{n} 원천 확인(구조 토큰 포함), {EV}]'
            else:
                x['근거'] += f' [C19 근거 미확보: {n_ok}/{n}]'; DROPPED.append(_i)
assert not DROPPED, DROPPED
'''
rep("\n# 쓰기(작업 사본)\n", C19BLOCK + "\n# 쓰기(작업 사본)\n")
rep('with open(OUT + r"\\facts_patch_v19.csv", \'w\', encoding=\'utf-8-sig\', newline=\'\') as f:', 'with open(OUT + r"\\facts_patch_v20.csv", \'w\', encoding=\'utf-8-sig\', newline=\'\') as f:')
rep('json.dump(dict(src_sha=SRC_SHA, n=len(log), kinds=Counter(x[\'종류\'] for x in log), r12_fixed=fixed), open(OUT + r"\\cache\\s6_out.json", \'w\', encoding=\'utf-8\'), ensure_ascii=False, indent=1)',
    'json.dump(dict(src_sha=SRC_SHA, n=len(log), kinds=Counter(x[\'종류\'] for x in log), r12_fixed=fixed), open(OUT + r"\\cache\\s10_out.json", \'w\', encoding=\'utf-8\'), ensure_ascii=False, indent=1)')
s = '# [C19-T3 s10 생성본] 원본 C18-T3 scripts/s6_facts.py — 변경: 경로·원고 sha·LINK-IO 앞부분 치환·행 추가 8칸·v20 동기·V1E 주석·순환 근거 표시\n' + s
open(os.path.join(W, 'scripts', 's10_facts_v20_gen.py'), 'w', encoding='utf-8').write(s)
out('written s10_facts_v20_gen.py', len(s))
