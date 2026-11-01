# C20-T3 공용: 경로·로더·숫자 토큰·동치 규칙(PREREG 2절). 입력은 읽기 전용, 쓰기는 C20\C20-T3만.
import os, sys, re, io, csv, json, hashlib, glob
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN, InvalidOperation
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
ROOT = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
AN = ROOT + r'\분석'
MC = AN + r'\모델경쟁_20260927'
VL = AN + r'\검증루프_20261012'
REP = AN + r'\보고서_20261012'
MANU = REP + r'\manuscript_v3.md'
MANU_SHA1 = 'be2e0e1da47da20ddf79bdd5f2f7d67152454b21'
FACTS_NOW = REP + r'\facts'
FACTS_ORIG = REP + r'\facts\backup_20261017'
ORIG_SHA1 = {'facts.csv': 'd8d3c2d6f08aa34e5751558c12f68750c2b6ba92', 'facts_supplement.csv': 'c7fb7d78d4cc48465b05ab749af7b1eaffa2e532',
             'facts_supplement_R12.csv': '1b6fd7c4c62c45cc2fb6ef06e641b06f7638b319'}
NOW_SHA1 = {'facts.csv': '348744b0f87e8aa8dc59bed090876f24dccdefbb', 'facts_supplement.csv': '6f2657c0711ca3a590fb91ceaad8b3fb47a195f5',
            'facts_supplement_R12.csv': '97e750eeb4d20a5af7cc8d5d876965baf868c390'}
C19T3 = VL + r'\C19\C19-T3'
PATCH20 = C19T3 + r'\facts_patch_v20.csv'
PATCH20_SHA1 = '7107b01b7667cde8ca676eb22feed2cd0b869fe8'
W = VL + r'\C20\C20-T3'
C = W + r'\cache'
FILES = ['facts.csv', 'facts_supplement.csv', 'facts_supplement_R12.csv']
HDR = ['id', '주제', '수치', '단위', '정확한 문장 조각', '출처 파일 경로', '조건·경고', 'note']
KNOWN4 = {'C-RCA-SUB': ('-0.015', '-0.024'), 'T3NC-txt_vs_none': ('-0.0023', '-0.0020'),
          'T3EXT-prov-txt-종사자수': ('+0.114', '+0.106'), 'T3EXT-34': ('+0.115', '+0.111')}


def sha1(p):
    return hashlib.sha1(open(p, 'rb').read()).hexdigest()


def jdump(o, name):
    json.dump(o, open(os.path.join(C, name), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)


def read_rows(p):
    s = open(p, 'rb').read().decode('utf-8-sig')
    return list(csv.reader(io.StringIO(s, newline='')))


def manu_lines():
    assert sha1(MANU) == MANU_SHA1, 'manuscript changed'
    return open(MANU, encoding='utf-8').read().replace('\r\n', '\n').split('\n')


def appendix(ms):
    st = next(i for i, s in enumerate(ms) if s.startswith('# [부록'))
    en = next(i for i, s in enumerate(ms) if s.startswith('# [편집 메모'))
    APP = {}
    for i in range(st, en):
        s = ms[i]
        if s.startswith('| ') and not s.startswith('| ---'):
            c = [x.strip() for x in s.strip().strip('|').split(' | ')]
            if len(c) >= 4 and c[0] != 'id':
                key = re.sub(r'\s*\((보충|보충 R12)\)$', '', c[0])
                APP[key] = dict(line=i + 1, sec=c[1], val=' | '.join(c[2:-1]), src=c[-1])
    return APP, st, en


def sections(ms, st):
    """절 번호 → (시작 줄 idx, 끝 idx). '표지·요약' = 0..'# 1.' 직전, '데이터 출처' = 해당 ## 절."""
    heads = [(i, s) for i, s in enumerate(ms[:st]) if s.startswith('#')]
    out = {}
    for k, (i, s) in enumerate(heads):
        lvl = len(s) - len(s.lstrip('#'))
        m = re.match(r'#+\s+(\d+(?:\.\d+)*)', s)
        name = m.group(1) if m else s.lstrip('#').strip()
        j = next((h for h, t in heads[k + 1:] if (len(t) - len(t.lstrip('#'))) <= lvl), st)
        out[name] = (i, j)
    first1 = next(i for i, s in heads if s.startswith('# 1.'))
    out['표지·요약'] = (0, first1)
    out['요약'] = out.get('요약', (0, first1))
    return out


NUM = re.compile(r'(?<![A-Za-z0-9_])[−\-+]?\d[\d,]*(?:\.\d+)?')


def toks(s, keep_sign=True):
    """숫자 토큰(정규화 문자열) 목록. '–' '~' '/' 는 구분자."""
    out = []
    s = s.replace('–', ' – ').replace('~', ' ~ ').replace('/', ' / ').replace('→', ' → ')
    for m in NUM.finditer(s):
        t = m.group(0).replace('−', '-').replace('+', '').rstrip(',')  # C20 재개: 끝 쉼표를 먼저 떼야 '392,048,'가 천 단위로 인식됨
        if re.fullmatch(r'-?\d{1,3}(,\d{3})+(\.\d+)?', t):
            t = t.replace(',', '')
        else:
            t = t.rstrip(',')
            if ',' in t:  # '1,2' 같은 나열은 쪼갬
                parts = t.split(',')
                out.extend([p for p in parts if p and p != '-'])
                continue
        if not keep_sign:
            t = t.lstrip('-')
        if t in ('', '-'):
            continue
        out.append(t)
    return out


def dec(t):
    try:
        return Decimal(t)
    except InvalidOperation:
        return None


def ndp(t):
    return len(t.split('.')[1]) if '.' in t else 0


def _rnd(x, d, mode):
    q = Decimal(1).scaleb(-d)
    return x.quantize(q, rounding=mode)


def eq_tok(a, b, pct=False):
    """PREREG R1–R3. a, b 정규화 토큰(부호 포함). 반환: 'R1'|'R2'|'R3'|''"""
    if a == b:
        return 'R1'
    A, B = dec(a), dec(b)
    if A is None or B is None:
        return ''
    if A == B:
        return 'R1'
    da, db = ndp(a), ndp(b)

    def r2(x, dx, y, dy):
        # 자릿수 많은 쪽을 적은 쪽 자릿수로 반올림
        if dx >= dy:
            return any(_rnd(x, dy, m) == y for m in (ROUND_HALF_UP, ROUND_HALF_EVEN))
        return any(_rnd(y, dx, m) == x for m in (ROUND_HALF_UP, ROUND_HALF_EVEN))
    if da != db and r2(A, da, B, db):
        return 'R2'
    if pct:
        for f in (Decimal(100), Decimal('0.01')):
            X = A * f
            dX = max(0, da - 2) if f == 100 else da + 2   # 배율 후 a의 표기 자릿수
            if X == B or r2(X, dX, B, db):
                return 'R3'
    return ''


def match_any(t, pool, pct=False, signless=False):
    for p in pool:
        a, b = (t.lstrip('-'), p.lstrip('-')) if signless else (t, p)
        r = eq_tok(a, b, pct)
        if r:
            return r, p
    return '', ''


def resolve_paths(cell):
    """출처 칸 → 실제 경로 목록 [(원문 조각, 경로 또는 None)]"""
    out = []
    prev_dir = None
    for piece in re.split(r';', cell):
        for sub in re.split(r',\s+', piece):
            s = sub.strip()
            if not s:
                continue
            s = re.sub(r'\s*\([^)]*\)\s*$', '', s).strip()   # 끝 괄호 설명 제거
            s = re.sub(r'\s+\S*\s*행$', '', s)
            if not s or s.startswith('<'):
                continue
            cand = None
            if s.startswith('분석\\'):
                cand = ROOT + '\\' + s
            elif s.startswith('(MC)\\'):
                cand = MC + s[4:]
            elif s.startswith('(분석)\\'):
                cand = AN + s[4:]
            elif s.startswith('검증루프_20261012\\') or s.startswith('모델경쟁_20260927\\') or s.startswith('보고서_20261012\\'):
                cand = AN + '\\' + s
            elif re.match(r'C\d\d\\', s):
                cand = VL + '\\' + s
            elif prev_dir and re.match(r'^[\w\-\.\*\\가-힣]+\.(json|csv|txt|md|parquet|py|tsv|log|xlsx|sqlite)$', s):
                cand = prev_dir + '\\' + s
            if cand is None:
                out.append((s, None))
                continue
            cand = cand.split(' ')[0] if not os.path.exists(cand) and os.path.exists(cand.split(' ')[0]) else cand
            if '*' in cand:
                g = sorted(glob.glob(cand))
                for x in g:
                    out.append((s, x))
                if not g:
                    out.append((s, cand))
            else:
                out.append((s, cand))
            prev_dir = os.path.dirname(cand)
    return out


TEXT_EXT = ('.json', '.csv', '.txt', '.md', '.tsv', '.log', '.py')
_src_cache = {}


def src_numbers(path):
    if path in _src_cache:
        return _src_cache[path]
    if not os.path.exists(path):
        r = ('파일 없음', '', [])
    elif not path.lower().endswith(TEXT_EXT):
        r = ('검사 불가', '', [])
    elif os.path.getsize(path) > 80_000_000:
        r = ('검사 불가(대용량)', '', [])
    else:
        t = open(path, encoding='utf-8', errors='replace').read()
        nums = set()
        for src_t in (t, re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', t)):
            for m in re.finditer(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', src_t):
                nums.add(m.group(0))
        r = ('읽음', t, sorted(nums))
    _src_cache[path] = r
    return r


def src_has(tok, nums_txt):
    """원천 텍스트 숫자 x 가운데 x, x×100, |x|를 tok 자릿수로 반올림해 같아지는 것이 있는가(PREREG C3)."""
    t = tok.lstrip('-')
    T = dec(t)
    if T is None:
        return ''
    d = ndp(t)
    status, txt, nums = nums_txt
    for x in nums:
        try:
            X = abs(Decimal(x))
        except InvalidOperation:
            continue
        if X == T:
            return 'same'
        for f, lab in ((Decimal(1), 'round'), (Decimal(100), 'x100')):
            Y = X * f
            try:
                if any(_rnd(Y, d, m) == T for m in (ROUND_HALF_UP, ROUND_HALF_EVEN)):
                    return lab
            except InvalidOperation:
                pass
    if tok in txt or t in txt:
        return 'string'
    return ''
