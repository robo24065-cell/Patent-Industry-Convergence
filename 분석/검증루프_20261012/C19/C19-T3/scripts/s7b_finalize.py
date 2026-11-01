# C19-T3 s7b: s7 1차 분류에 줄 규칙(본문 전용·다른 절 일치 숫자)과 재실행 결과(diff_numbers.csv)를 붙여 source_map.csv를 만든다.
# 줄 규칙은 원고 문맥을 읽고 정했다(주석의 근거). '연결 방법' 열로 신뢰도를 남긴다.
import os, sys, re, csv, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
rows = json.load(open(C + r'\s7_rows_raw.json', encoding='utf-8'))
R_MAP = 'C19 V1E 재실행(map 사본)'; R_LO = 'C19 V1E 재실행(label_origin 사본)'; NO = '미재실행(모형 재학습 필요)'
K = {  # 분류 이름 → (원천 스크립트, 사용구분, 재실행)
 'LO': ('label_origin_R10 lo_fam(→lo_tests·R11)', '사용', R_LO),
 'MAP': ('map5·split34 계열', '사용', R_MAP),
 'ROWS': ('제공 메타 행 수(_shared README·c1)', '사용(행 수 사실)', '해당 없음(V1E 고유 문서 수는 s0: US 2,380,849·EP 799,548)'),
 'INV': ('창 2015–19/2017–19 라벨 재현·지형도 대안(c1·c2·C10-T3·C11-T2)', '사용(구조상 불변)', '불필요(쌍 행은 US 출원 2000–03, EP 2000–09)'),
 'KR': ('KR 문서·법인·NTIS·외부 원천', '사용 안 함', '해당 없음'),
 'R12': ('R12 확인 시험(KR 법인 문서)', '사용 안 함', '해당 없음'),
 'T3OFF': ('T3 공식 최종시험', '사용(공식 성적, 정정 대상 아님)', '공식 불변(C18-T2 V1E 민감도)'),
 'T1OFF': ('T1 공식 최종시험', '사용(공식 성적, 정정 대상 아님)', '공식 불변(C18-T2 V1E 민감도)'),
 'T3AN': ('T3 계열 비공식 분석(재학습)', '사용', NO),
 'T1AN': ('T1 계열 비공식 분석(재학습, ep_*)', '사용', NO),
 'N5': ('N5 계열', '사용(공식 성적, 정정 대상 아님)', '공식 불변(C18-T2 V1E 민감도)'),
 'NX': ('N4·N6·N7·N8·N9·T4 트랙', '사용 안 함', '해당 없음'),
 'B1': ('T1 TXT_R9/NONCIRC(KR, ep_* 없음)', '사용 안 함', '해당 없음'),
 'DEF': ('정의·설계 숫자(개수·기간·자릿수)', '해당 없음', '해당 없음'),
 'STR': ('구조 숫자(절·표·그림 참조)', '해당 없음', '해당 없음'),
}
# 줄 기본값(본문 전용·다른 절 일치 숫자에 적용) — 원고 문맥 확인
LINE = {9: 'KR', 11: 'ROWS', 13: 'R12', 15: 'MAP', 17: 'T3OFF', 19: 'DEF', 33: 'KR', 46: 'DEF', 49: 'DEF', 50: 'DEF', 51: 'DEF', 53: 'DEF',
        57: 'ROWS', 59: 'KR', 60: 'LO', 67: 'R12', 68: 'INV', 69: 'INV', 70: 'INV', 71: 'INV', 73: 'INV', 75: 'INV',
        88: 'KR', 89: 'KR', 90: 'KR', 91: 'KR', 92: 'KR', 93: 'KR', 94: 'KR', 95: 'KR', 96: 'KR', 97: 'LO', 99: 'KR',
        105: 'STR', 109: 'T3OFF', 112: 'DEF', 116: 'DEF', 124: 'T3OFF', 125: 'T3OFF', 126: 'NX', 127: 'NX', 128: 'STR', 132: 'DEF', 134: 'R12', 135: 'STR', 136: 'T3AN', 138: 'DEF',
        142: 'KR', 143: 'MAP', 150: 'STR', 153: 'DEF', 161: 'MAP', 162: 'MAP', 163: 'MAP', 166: 'DEF', 168: 'INV', 172: 'MAP', 174: 'DEF', 183: 'DEF', 185: 'MAP',
        189: 'DEF', 191: 'MAP', 192: 'MAP', 194: 'KR', 199: 'DEF', 205: 'DEF', 209: 'T3OFF', 215: 'T3OFF', 216: 'T3AN', 217: 'T3AN', 219: 'T3AN', 223: 'T3AN',
        229: 'DEF', 233: 'T1OFF', 239: 'T1OFF', 240: 'T1AN', 242: 'T1AN', 244: 'B1', 248: 'T1AN', 251: 'DEF', 253: 'KR', 257: 'KR', 262: 'KR', 264: 'NX', 265: 'NX', 267: 'NX',
        275: 'NX', 283: 'NX', 284: 'NX', 286: 'NX', 287: 'NX', 288: 'NX', 289: 'NX', 293: 'STR', 297: 'DEF', 299: 'R12', 314: 'STR', 316: 'R12', 317: 'R12', 321: 'R12', 322: 'R12',
        323: 'R12', 324: 'R12', 326: 'LO', 328: 'R12', 336: 'B1', 349: 'STR', 351: 'R12', 352: 'R12', 353: 'R12', 362: 'STR', 370: 'R12', 374: 'KR', 384: 'T3OFF', 385: 'T1OFF',
        387: 'NX', 388: 'NX', 389: 'MAP', 395: 'NX', 401: 'DEF', 402: 'STR', 407: 'B1', 408: 'DEF', 409: 'T3AN', 410: 'T1AN', 415: 'DEF', 427: 'KR'}
TOK = {(13, '87'): 'LO', (13, '90'): 'LO', (13, '0.002'): 'B1', (13, '0.005'): 'B1', (13, '7.1'): 'KR', (13, '29.4'): 'KR', (13, '40.4'): 'KR',
       (60, '87'): 'LO', (60, '90'): 'LO', (67, '89.7'): 'LO', (67, '1087511'): 'ROWS', (67, '7.2'): 'ROWS', (68, '2382948'): 'ROWS', (68, '15.7'): 'ROWS', (68, '87.3'): 'LO',
       (69, '2163018'): 'ROWS', (69, '14.3'): 'ROWS', (69, '87.0'): 'LO', (70, '8700777'): 'ROWS', (70, '57.5'): 'ROWS', (71, '806905'): 'ROWS', (71, '5.3'): 'ROWS', (71, '90.5'): 'LO',
       (73, '48.8'): 'KR', (142, '0.713'): 'MAP', (142, '0.523'): 'MAP', (143, '0.0022'): 'T1AN', (143, '0.007'): 'T1AN', (143, '0.25'): 'KR', (143, '0.26'): 'KR', (143, '0.20'): 'KR', (143, '0.21'): 'KR',
       (168, '0.45'): 'MAP', (168, '9'): 'MAP', (168, '27'): 'MAP', (168, '29'): 'MAP', (168, '70'): 'KR', (168, '48'): 'KR', (172, '0.30'): 'INV', (172, '0.34'): 'INV', (172, '0.04'): 'INV', (172, '0.01'): 'INV',
       (172, '48'): 'DEF', (172, '7'): 'DEF', (172, '14'): 'DEF', (194, '6'): 'KR', (194, '13'): 'KR', (253, '6.4'): 'MAP', (253, '16.5'): 'MAP', (253, '12.6'): 'KR',
       (409, '0.003'): 'NX', (410, '0.006'): 'NX', (410, '0.010'): 'NX', (410, '18'): 'KR', (410, '96'): 'KR', (410, '19'): 'KR', (410, '47'): 'KR', (410, '5.5'): 'R12'}
D = list(csv.DictReader(open(W + r'\diff_numbers.csv', encoding='utf-8-sig')))
changed = {}
NUM = re.compile(r'[−\-+]?\d[\d,]*(?:\.\d+)?')
for d in D:
    for m in NUM.finditer(d['원래문자열']):
        t = m.group(0).replace('−', '').replace('-', '').replace('+', '').replace(',', '')
        changed.setdefault((int(d['원고행']), t), []).append(d['새문자열'])
outrows = []
for r in rows:
    ln = int(r['원고행']); nt = r['정규화'].replace(',', '')
    method = ''
    if r['원천'].startswith('구조 숫자'): method = '구조 숫자(정규식)'
    elif r['구분'] == '부록' and r['부록ID']: method = '부록 행 출처 칸'
    elif r['구분'] == '부록': method = '부록 머리·설명'
    elif '절 불일치' in r['원천'] or r['원천'].startswith('본문 전용'):
        cls = TOK.get((ln, nt)) or LINE.get(ln)
        if cls:
            scr, use, rr = K[cls]
            r = {**r, '원천 스크립트': scr, 'US·EP 행 사용': use, '재실행': rr}
            method = '줄·토큰 규칙(원고 문맥)' if (ln, nt) not in TOK else '토큰 규칙(원고 문맥)'
        else:
            method = '절 기본값'
    else:
        method = '부록 일치(같은 절)'
        if (ln, nt) in TOK:
            scr, use, rr = K[TOK[(ln, nt)]]; r = {**r, '원천 스크립트': scr, 'US·EP 행 사용': use, '재실행': rr}; method = '토큰 규칙(원고 문맥)'
    res = ''
    if r['재실행'].startswith('C19 V1E 재실행') or r['재실행'].startswith('C19 cache'):
        res = '바뀜 → ' + '; '.join(changed[(ln, nt)]) if (ln, nt) in changed else '불변(P0=V1E 표기)'
    elif (ln, nt) in changed:
        res = '바뀜 → ' + '; '.join(changed[(ln, nt)])
    outrows.append({'원고행': ln, '구분': r['구분'], '절': r['절'], '숫자': r['숫자'], '원천': r['원천'], '부록ID': r['부록ID'], '원천 스크립트': r['원천 스크립트'],
                    'US·EP 행 사용 여부': r['US·EP 행 사용'], '재실행 여부': r['재실행'], 'V1E 결과': res, '연결 방법': method, '근거': r['근거']})
with open(W + r'\source_map.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(outrows[0])); w.writeheader(); w.writerows(outrows)
from collections import Counter
S = dict(n=len(outrows), use=Counter(r['US·EP 행 사용 여부'] for r in outrows), method=Counter(r['연결 방법'] for r in outrows),
         unresolved=[(r['원고행'], r['숫자']) for r in outrows if r['US·EP 행 사용 여부'] in ('확인 필요', '미분류')],
         changed_tokens=sum(1 for r in outrows if r['V1E 결과'].startswith('바뀜')),
         used_not_rerun=Counter((r['원천 스크립트']) for r in outrows if r['US·EP 행 사용 여부'] == '사용' and r['재실행 여부'].startswith('미재실행')),
         used_not_rerun_lines=sorted(set(r['원고행'] for r in outrows if r['US·EP 행 사용 여부'] == '사용' and r['재실행 여부'].startswith('미재실행'))),
         changed_lines=sorted(set(r['원고행'] for r in outrows if r['V1E 결과'].startswith('바뀜'))))
jdump(S, 's7b_summary.json')
out(json.dumps(S, ensure_ascii=False, default=str)[:3000])
