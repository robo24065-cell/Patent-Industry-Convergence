# C19-T3 s13: facts_patch_v20 '적용 가능' 판정(PREREG 6절) — 칸 수·형식·LINK-IO·순환 근거·대상 밖 행 불변·잔여 불일치
import os, sys, csv, io, json, re, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
WK = W + r'\facts_work'; R = {}
HDR = ['id', '주제', '수치', '단위', '정확한 문장 조각', '출처 파일 경로', '조건·경고', 'note']
PH = ['파일', 'id', '열', '현재값', '새값', '근거', '출처바퀴', '종류']
def rows_of(raw):
    t = raw.decode('utf-8-sig'); return list(csv.reader(io.StringIO(t.replace('\r\n', '\n'))))
fmt = {}
for name, want_bom, want_crlf in [('facts.csv', True, True), ('facts_supplement.csv', True, True), ('facts_supplement_R12.csv', False, False)]:
    a = open(FACTS + '\\' + name, 'rb').read(); b = open(WK + '\\' + name, 'rb').read()
    ra, rb = rows_of(a), rows_of(b)
    bad = [i + 1 for i, r in enumerate(rb) if len(r) != len(HDR)]
    lf_only = b.count(b'\n') - b.count(b'\r\n')
    da = {r[0]: r for r in ra[1:] if r}; db = {r[0]: r for r in rb[1:] if r}
    fmt[name] = dict(bom=b.startswith(b'\xef\xbb\xbf'), want_bom=want_bom, crlf_lines=b.count(b'\r\n'), lf_only_lines=lf_only, want_crlf=want_crlf,
                     format_ok=(b.startswith(b'\xef\xbb\xbf') == want_bom) and ((lf_only == 0) if want_crlf else (b.count(b'\r\n') == 0)),
                     header=rb[0] == HDR, rows=len(rb) - 1, bad_field_rows=bad, src_bad_field_rows=[i + 1 for i, r in enumerate(ra) if len(r) != len(HDR)],
                     changed=sorted(k for k in da if k in db and da[k] != db[k]), added=sorted(k for k in db if k not in da), removed=sorted(k for k in da if k not in db))
R['format'] = fmt
# 패치 csv
P = list(csv.reader(io.StringIO(open(W + r'\facts_patch_v20.csv', encoding='utf-8-sig').read())))
R['patch'] = dict(header=P[0] == PH, n=len(P) - 1, bad_rows=[i + 1 for i, r in enumerate(P) if len(r) != len(PH)])
add_bad = []
for r in P[1:]:
    d = dict(zip(PH, r))
    if d['종류'] == '행 추가':
        f = next(csv.reader(io.StringIO(d['새값'])))
        if len(f) != len(HDR) or not f[2] or not f[3]: add_bad.append((d['id'], len(f), f[2:4]))
R['patch']['add_rows'] = sum(1 for r in P[1:] if r[7] == '행 추가'); R['patch']['add_rows_bad'] = add_bad
# 패치 대상 밖 행 불변
patched = set()
for r in P[1:]:
    d = dict(zip(PH, r)); patched.add((d['파일'], d['id'].split('→')[-1])); patched.add((d['파일'], d['id'].split('→')[0]))
outside = []
for name in fmt:
    for k in fmt[name]['changed'] + fmt[name]['added'] + fmt[name]['removed']:
        if (name, k) not in patched: outside.append((name, k))
R['outside_patch_changes'] = outside
# LINK-IO
fw = {r[0]: dict(zip(HDR, r)) for r in rows_of(open(WK + r'\facts.csv', 'rb').read())[1:]}
li = fw['LINK-IO']['정확한 문장 조각']
R['LINK-IO'] = dict(text=li, has_ellipsis='…' in li, paren_balanced=li.count('(') == li.count(')'), tail_kept=li.endswith('KSIC→310), 제품코드 수 비례 가중)'),
                    front_new=li.startswith('한국은행 2010–2015–2020 접속 산업연관표의 2010년 표(2020 분류, 통합소분류 165; 파일 수정 2026-06-30)'))
R['ellipsis_in_frags'] = [k for k, v in fw.items() if '…' in v['정확한 문장 조각']]
# 순환 18행 근거
ev = json.load(open(C + r'\s11_evidence.json', encoding='utf-8'))['verdict']
C18L = ['LINK-IO', 'T3NC-MG', 'C-LAND-free_F', 'C-LAND-txt_F', 'C-AGREE-US', 'C-AGREE-JP', 'C-AGREE-EP', 'C-AGREE-RANGE', 'C-RCA-TXT', 'C-RCA-FREE-IPC',
        'C-T3-AUCS', 'C-T3-RET', 'C-T3-drop_ros-hyb_txt', 'C-T3-drop_ros-hyb_free', 'DATA-REG-EVENTS', 'DATA-KR-REG', 'LINK-NTIS', 'C-COUNTRY-SHARE']
R['circular'] = dict(n=len(C18L), no_evidence=[k for k in C18L if ev[k]['근거'] != '있음'], synced_v20_no_evidence=[k for k in ev if k not in C18L and ev[k]['근거'] != '있음'])
res = json.load(open(C + r'\s12_c19work_v20.json', encoding='utf-8'))
R['residual'] = dict(v20=f"{res['A또는B']}/{res['대조행']}", c18_ref='4/204', items=res['잔여'])
ok = (all(v['format_ok'] and v['header'] and not v['bad_field_rows'] for v in fmt.values()) and R['patch']['header'] and not R['patch']['bad_rows'] and not add_bad
      and not outside and not R['LINK-IO']['has_ellipsis'] and R['LINK-IO']['paren_balanced'] and R['LINK-IO']['tail_kept'] and not R['circular']['no_evidence']
      and res['A또는B'] / res['대조행'] <= 4 / 204 + 1e-12 and res['A또는B'] <= 4)
R['적용가능'] = ok
R['sha1'] = {n: sha1(WK + '\\' + n) for n in fmt}; R['sha1_patch'] = sha1(W + r'\facts_patch_v20.csv'); R['src_sha1'] = {n: sha1(FACTS + '\\' + n) for n in fmt}
jdump(R, 's13_format_check.json')
out(json.dumps({k: v for k, v in R.items() if k not in ('LINK-IO',)}, ensure_ascii=False, default=str)[:4000])
out('LINK-IO', R['LINK-IO'])
