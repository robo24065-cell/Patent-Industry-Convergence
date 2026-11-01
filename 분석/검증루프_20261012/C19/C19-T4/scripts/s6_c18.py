# -*- coding: utf-8 -*-
# C19-T4 s6: C18 ERRATA 본문 항목(E-C18T1-01, O1–O9)과 부록 항목(A1–A10 대표 문자열, B1)의 v19/v20 md·v20 docx·pdf 존재 확인.
import os, re, json
import docx, fitz
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.dirname(os.path.dirname(D))
R = os.path.join(os.path.dirname(B), '보고서_20261012')
V19 = open(os.path.join(R, 'drafts', 'manuscript_for_v19.md'), encoding='utf-8').read().split('\n')
V20 = open(os.path.join(R, 'manuscript_v3.md'), encoding='utf-8').read().split('\n')
def nows(s):
    return re.sub(r'\s+', '', s.replace('**', '').replace('*', '').replace(' ', ' ').replace('­', ''))
d = docx.Document(os.path.join(R, '보고서_초안_v20.docx'))
parts = [p.text for p in d.paragraphs] + [c.text for t in d.tables for row in t.rows for c in row.cells]
DX = nows('\n'.join(parts))
PF = nows(''.join(pg.get_text() for pg in fitz.open(os.path.join(R, '보고서_초안_v20.pdf'))))
items = [
 ('E-C18T1-01', '2024년 전후 일부 등록번호가 비어', '2024–25년 일부 등록번호가 비어 2024년 등록이 공식보다 약 19%(2025년 47%) 적다', True),
 ('O1', '(빼거나 합쳐도 skill 0.38–0.42)', '(신설 분류 셀을 빼거나 합쳐도 skill 0.38–0.42)', True),
 ('O2', '물가·산업연관표·생산지수는', '물가·산업연관표·광공업지수는', True),
 ('O3a', '법인 연결표를 빼면', '법인 연결표·DART 업종을 빼면', True),
 ('O3b', '국민연금 사업장은 상호 변경 키를', '국민연금은 상호 변경 키를', True),
 ('O4', '기운다(연결률', '기운다(N6 미연결 기업은 결과 0; 연결률', True),
 ('O5', 'T1 위험쌍 조건도', 'T1 법인·위험쌍 조건도', True),
 ('O6', '| KR 출원의 9.34%, 등록의 10.95% |', '| KR 출원(실용신안 포함)의 9.34%, 등록의 10.95% |', True),
 ('O7a', '0에 닿는 T4(−0.003)', '0 밑인 T4(−0.003)', True),
 ('O7b', 'CI 하한이 0에 닿고(−0.003)', 'CI 하한이 0 밑이고(−0.003)', True),
 ('O8', '약 46–48%', '약 47–48%', True),
 ('O9', '권리범위확인심판 예고등록이다', '권리범위확인심판 예고등록(직권)이다', True),
 ('A2', '| CIRC-AGREE-CC | 표지·요약, 1.3, 3.6.1, 3.6.4 |', '| CIRC-AGREE-CC | 표지·요약, 1.3 |', False),
 ('A-LINK-NPS', '| LINK-NPS | 1.4, 2.5 |', '| LINK-NPS | 1.4 |', False),
 ('A-NTIS-SRC', '| NTIS-SRC | 4.4 |', '| NTIS-SRC | 데이터 출처 |', False),
 ('A-C-LAND-prov', '| C-LAND-prov | 표지·요약, 4.4 |', '| C-LAND-prov | 표지·요약, 3.1.1, 4.1 |', False),
 ('A-pkr_altA', None, 'C14\\C14-T2\\pkr_ipc_altA.csv', False),
 ('A-s4_horizon', None, 'C15-T1\\cache\\s4_horizon.csv', False),
 ('B1', None, '| C18-T1-REGGAP | 4.4 |', False),
]
out = []
N19 = [nows(x) for x in V19]; N20 = [nows(x) for x in V20]
for k, old, new, body in items:
    rec = {'item': k, 'body': body}
    if old:
        o = nows(old)
        rec['old_v19'] = [i + 1 for i, x in enumerate(N19) if o in x][:4]
        rec['old_v20'] = [i + 1 for i, x in enumerate(N20) if o in x][:4]
    n = nows(new)
    rec['new_v19'] = [i + 1 for i, x in enumerate(N19) if n in x][:4]
    rec['new_v20'] = [i + 1 for i, x in enumerate(N20) if n in x][:4]
    if body:
        rec['new_in_v20_docx'] = n in DX
        rec['new_in_v20_pdf'] = n in PF
        if old:
            rec['old_in_v20_docx'] = nows(old) in DX
    out.append(rec)
json.dump(out, open(os.path.join(D, 'cache', 's6_c18.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for r in out:
    print(r['item'], {k: v for k, v in r.items() if k not in ('item', 'body')})
