# C19-T3 s14: 시험 빌드(PREREG 7절). C18-T3 s5_build.py 절차(FIG_NATIVE=1, md2docx.js·render.ps1 사본) 그대로, 기준만 v20(40406e29, 보고서_초안_v20).
# 사용: python -I s14_build.py N0 B1 B2 B2s A13
import os, sys, json, subprocess, hashlib, zipfile, shutil, re
import fitz
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
R = REP; T = W + r'\build_test'; os.makedirs(T, exist_ok=True)
src = '\n'.join(manu_lines()) if False else open(MANU, encoding='utf-8').read()
assert sha1(MANU) == MANU_SHA1
for f in ['md2docx.js', 'render.ps1']:
    if not os.path.exists(os.path.join(T, f)): shutil.copy2(os.path.join(R, 'build', f), os.path.join(T, f))
    assert open(os.path.join(T, f), 'rb').read() == open(os.path.join(R, 'build', f), 'rb').read(), f
SRCNOTE = '; 제공 데이터 그대로(번호 형식 이중 수록 포함) 계산 — 이중 수록 제거(V1E) 값: (분석)\\검증루프_20261012\\C19\\C19-T3\\diff_numbers.csv'
def appx_src(t, rid, add):
    lines = t.split('\n'); hit = [k for k, s in enumerate(lines) if s.startswith('| ' + rid + ' |')]
    assert len(hit) == 1, (rid, len(hit)); s = lines[hit[0]]; assert s.endswith(' |')
    lines[hit[0]] = s[:-2] + add + ' |'; return '\n'.join(lines), len(add)
E = {
 # 대안 B 부록(제출 docx 밖)
 'BA1': [('제공 융합데이터 5개국 메타 15,141,159행(1행=출원×KSIC 쌍, 원본 TSV 50개 29.5GB)', '제공 융합데이터 5개국 메타 15,141,159행(1행=출원×KSIC 쌍, 원본 TSV 50개 29.5GB; 번호 형식 이중 수록 US 2,099·EP 7,357행 포함, 고유 문서 기준 15,131,703)')],
 'BA2': [('EP 5.3%(806,905) |', 'EP 5.3%(806,905); 행 수는 이중 수록 포함(고유 문서 US 2,380,849·EP 799,548; 비중 표기 불변) |')],
 'BA3': 'SRC:CIRC-AGREE-CC,CIRC-LINKS,SP-FCST,MAP-LEADLAG,MAP-SYNC,MAP-rho_patRTAchange_expRCAchange_0817,SP-VAR-한국기원점유_변화pp',
 # 대안 B 본문(선택): 73행 표 2 주 끝
 'BB': [('CN 문서의 48.8%는 실용신안이다.', 'CN 문서의 48.8%는 실용신안이다. 행 수는 번호 형식만 다른 같은 출원의 이중 수록을 포함한다(고유 문서 US 2,380,849·EP 799,548).')],
 'BBs': [('CN 문서의 48.8%는 실용신안이다.', 'CN 문서의 48.8%는 실용신안이다. 행 수는 번호 형식 이중 수록 포함(고유 US 2,380,849·EP 799,548).')],
}
# 참고: 대안 A 13행(diff_numbers.csv 치환) — 사전 규칙상 채택 아님, 지면 영향 기록용
import csv
DN = list(csv.DictReader(open(W + r'\diff_numbers.csv', encoding='utf-8-sig')))
def apply_A(t):
    lines = t.split('\n'); n = 0
    for d in DN:
        k = int(d['원고행']) - 1
        assert d['원래문자열'] in lines[k], (d['원고행'], d['원래문자열'])
        lines[k] = lines[k].replace(d['원래문자열'], d['새문자열'], 1); n += len(d['새문자열']) - len(d['원래문자열'])
    return '\n'.join(lines), n
V = {'N0': [], 'B1': ['BA1', 'BA2', 'BA3'], 'B2': ['BA1', 'BA2', 'BA3', 'BB'], 'B2s': ['BA1', 'BA2', 'BA3', 'BBs'], 'A13': ['A']}
BODYKEYS = {'BB', 'BBs', 'A'}
def apply(t, keys):
    dch = {}
    for k in keys:
        if k == 'A':
            t, n = apply_A(t); dch[k] = n; continue
        spec = E[k]; specs = spec if isinstance(spec, list) else [spec]; tot = 0
        for sp in specs:
            if isinstance(sp, str) and sp.startswith('SRC:'):
                for rid in sp[4:].split(','):
                    t, n = appx_src(t, rid, SRCNOTE); tot += n
            else:
                o, n2 = sp; assert t.count(o) == 1, (k, o, t.count(o)); t = t.replace(o, n2); tot += len(n2) - len(o)
        dch[k] = tot
    return t, dch
env = dict(os.environ, FIG_NATIVE='1', NODE_PATH=r'C:\Users\PC\AppData\Roaming\npm\node_modules')
outp = os.path.join(T, 'test_builds.json')
res = json.load(open(outp, encoding='utf-8')) if os.path.exists(outp) else {}
def pageinfo(d):
    bots, firsts = [], []
    for pg in d:
        ph = pg.rect.height; ys = []; first = None
        for b in pg.get_text('dict')['blocks']:
            for l in b.get('lines', []):
                txt = ''.join(s['text'] for s in l['spans']).strip()
                if txt and l['bbox'][1] < ph * 0.93:
                    ys.append(l['bbox'][3])
                    if first is None or l['bbox'][1] < first[0]: first = (l['bbox'][1], txt[:40])
        bots.append(round(max(ys), 1) if ys else 0); firsts.append(first[1] if first else '')
    return bots, firsts
base_pdf = fitz.open(R + r'\보고서_초안_v20.pdf'); base_b, base_f = pageinfo(base_pdf)
z20 = zipfile.ZipFile(R + r'\보고서_초안_v20.docx'); xml20 = z20.read('word/document.xml')
med20 = sorted(hashlib.sha1(z20.read(n)).hexdigest()[:8] for n in z20.namelist() if n.startswith('word/media/') and not n.endswith('/'))
res['_v20_base'] = dict(pdf_pages=base_pdf.page_count, page_bottoms=base_b, last_bottom=base_b[-1], slack_pt=round(785.2 - base_b[-1], 1), page_firsts=base_f,
                        n_para=xml20.count(b'<w:p>') + xml20.count(b'<w:p '), media=med20, n_media=len(med20), docxml_sha1=hashlib.sha1(xml20).hexdigest()[:8])
for k in sys.argv[1:]:
    t, dch = apply(src, V[k])
    md = os.path.join(T, f'ms_{k}.md'); open(md, 'w', encoding='utf-8', newline='\n').write(t)
    docx = os.path.join(T, f'{k}.docx')
    r1 = subprocess.run(['node', os.path.join(T, 'md2docx.js'), md, R + r'\figs', docx], env=env, capture_output=True, text=True, encoding='utf-8')
    r2 = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', os.path.join(T, 'render.ps1'), docx], capture_output=True, text=True, encoding='utf-8', errors='replace')
    mword = re.search(r'PAGES:\s*(\d+)', r2.stdout or '')
    d = fitz.open(docx.replace('.docx', '.pdf')); b, fl = pageinfo(d)
    z = zipfile.ZipFile(docx); xml = z.read('word/document.xml')
    med = sorted(hashlib.sha1(z.read(n)).hexdigest()[:8] for n in z.namelist() if n.startswith('word/media/') and not n.endswith('/'))
    npara = xml.count(b'<w:p>') + xml.count(b'<w:p ')
    body_d = sum(v for kk, v in dch.items() if kk in BODYKEYS); appx_d = sum(v for kk, v in dch.items() if kk not in BODYKEYS)
    res[k] = dict(edits=V[k], d_chars_body=body_d, d_chars_appendix=appx_d, per_edit=dch, pdf_pages=d.page_count, word_pages=int(mword.group(1)) if mword else None,
                  last_bottom=b[-1], slack_pt=round(785.2 - b[-1], 1), page_bottoms=b, page_firsts=fl,
                  pages_differing_from_v20=[i + 1 for i, (x, y) in enumerate(zip(b, base_b)) if abs(x - y) > 0.5] + ([f'쪽수 {d.page_count}'] if d.page_count != len(base_b) else []),
                  firsts_same_as_v20=(fl == base_f), docxml_same_as_v20=(xml == xml20), docxml_sha1=hashlib.sha1(xml).hexdigest()[:8], n_para=npara,
                  media_same_as_v20=(med == med20), n_media=len(med), md_sha1=hashlib.sha1(t.encode('utf-8')).hexdigest()[:8],
                  node_err=(r1.stderr or '')[-300:], render_out=(r2.stdout or '')[-120:])
    out(k, 'pdf', d.page_count, 'word', res[k]['word_pages'], 'last', b[-1], 'slack', res[k]['slack_pt'], 'dbody', body_d, 'dappx', appx_d,
        'diffpages', res[k]['pages_differing_from_v20'], 'firsts_same', res[k]['firsts_same_as_v20'], 'xml_same', res[k]['docxml_same_as_v20'], 'para', npara,
        'media', res[k]['media_same_as_v20'], len(med), (r1.stderr or '')[-200:])
    d.close(); z.close()
    json.dump(res, open(outp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
out('v20 base', {k: v for k, v in res['_v20_base'].items() if k not in ('page_bottoms', 'page_firsts')})
