# -*- coding: utf-8 -*-
"""C09-T2 재계산 공용 함수: 원천 로더, 판정(PREREG 3절)."""
import json, csv, os, re
from decimal import Decimal, ROUND_HALF_UP
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AN = os.path.abspath(os.path.join(BASE, '..', '..', '..'))
MC = os.path.join(AN, '모델경쟁_20260927')
REP = os.path.join(AN, '보고서_20261012')
LOOP = os.path.join(AN, '검증루프_20261012')
CUR = open(os.path.join(BASE, 'cache', 'manuscript_now_88d77045.md'), encoding='utf-8').read().split('\n')

_cache = {}
def P(rel):
    rel = rel.replace('(MC)', MC).replace('(LOOP)', LOOP).replace('(REP)', REP)
    return rel if os.path.isabs(rel) else os.path.join(AN, rel)

def J(rel, *keys):
    p = P(rel)
    if p not in _cache:
        _cache[p] = json.load(open(p, encoding='utf-8'))
    o = _cache[p]
    for k in keys:
        o = o[k]
    return o

def CSV(rel):
    p = P(rel)
    if ('CSV', p) not in _cache:
        _cache[('CSV', p)] = list(csv.DictReader(open(p, encoding='utf-8-sig')))
    return _cache[('CSV', p)]

def ROW(rel, **cond):
    rows = [r for r in CSV(rel) if all(str(r.get(k, '')).strip() == str(v) for k, v in cond.items())]
    if len(rows) != 1:
        raise KeyError(f'ROW {rel} {cond} -> {len(rows)} rows')
    return rows[0]

def F(x):
    return float(str(x).replace(',', '').replace('−', '-').replace('%', '').strip())

def TXT(rel):
    p = P(rel)
    if ('TXT', p) not in _cache:
        _cache[('TXT', p)] = open(p, encoding='utf-8-sig', errors='replace').read()
    return _cache[('TXT', p)]

def tokval(tok):
    t = tok.strip().rstrip(',').replace(',', '').replace('−', '-').replace('+', '').replace('≤', '')
    m = re.match(r'-?[0-9]*\.?[0-9]+(?:[eE]-?[0-9]+)?', t.replace('−', '-'))
    return float(m.group(0))

def decs(tok):
    t = tok.strip().rstrip(',').replace('≤', '')
    m0 = re.match(r'[+\-−]?[0-9,]*\.?[0-9]+(?:[eE][−\-+]?[0-9]+)?', t)
    t = m0.group(0) if m0 else t
    if re.search(r'[eE]', t):
        m = re.match(r'[+\-−]?(\d+)(?:\.(\d+))?[eE]([−\-+]?\d+)', t)
        mant = len(m.group(2) or '')
        ex = int(m.group(3).replace('−', '-'))
        return mant - ex
    return len(t.split('.')[1]) if '.' in t else 0

def judge(tok, val, absval=False):
    """반환 (verdict, rounded_str, diff). 허용 오차 = 0.5×10^-d (경계 포함)."""
    if isinstance(val, str):
        core = lambda x: (re.match(r'^[≤+\-−]?[0-9,./e−\-]+', x.strip()) or re.match(r'.*', x)).group(0).replace(',', '').rstrip('.')
        return ('정확' if core(val) == core(tok) else '오류'), val, None
    y = tokval(tok)
    d = decs(tok)
    v = abs(val) if absval else val
    yy = abs(y) if absval else y
    tol = 0.5 * 10 ** (-d)
    diff = v - yy
    q = Decimal(1).scaleb(-d) if d >= 0 else Decimal(1)
    try:
        rnd = str(Decimal(repr(v)).quantize(q, rounding=ROUND_HALF_UP))
    except Exception:
        rnd = repr(v)
    ok = abs(diff) <= tol + 1e-12 * max(1, abs(v))
    return ('정확' if ok else '오류'), rnd, diff

RESULTS = []
def chk(line, toks, vals, src, note='', pct=False, absval=False, defn='', srctype='원천', verdict_override=None):
    """line의 토큰 목록(toks)을 원천 값(vals)과 대조. defn: 정의·조건 불일치 설명(있으면 '부분')."""
    if not isinstance(toks, (list, tuple)):
        toks = [toks]; vals = [vals]
    text = CUR[line - 1]
    for t, v in zip(toks, vals):
        present = t.rstrip(',') in text
        if isinstance(v, Exception):
            RESULTS.append(dict(line=line, token=t, src_value='', rounded='', verdict='막힘', source=src, srctype=srctype,
                                note=f'{note} | 원천 오류: {v}'[:400], present=present)); continue
        vv = v * 100 if (pct and not isinstance(v, str)) else v
        vd, rnd, diff = judge(t, vv, absval=absval)
        if vd == '정확' and defn:
            vd = '부분'
        if verdict_override:
            vd = verdict_override
        RESULTS.append(dict(line=line, token=t, src_value=(vv if isinstance(vv, str) else round(vv, 10)), rounded=rnd,
                            verdict=vd, source=src, srctype=srctype,
                            note=(note + (' | 정의: ' + defn if defn else '')).strip(' |')[:500], present=present))

def safe(fn):
    try:
        return fn()
    except Exception as e:
        return e
