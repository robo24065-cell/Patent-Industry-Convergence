# C19-T3 verify V2: facts_patch_v20을 csv 모듈 없이 바이트 상태 기계로 읽고, 독립 적용기로 원본 facts에 다시 적용해 facts_work와 바이트 대조
import os, sys, json
sys.stdout.reconfigure(encoding='utf-8')
AN = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석'
FD = AN + r'\보고서_20261012\facts'; W = AN + r'\검증루프_20261012\C19\C19-T3'; WK = W + r'\facts_work'
def parse(b):
    """RFC4180 상태 기계: 따옴표·이중 따옴표·CR/LF. 반환: (BOM 여부, 줄바꿈 종류 집합, 레코드 목록)"""
    bom = b.startswith(b'\xef\xbb\xbf'); t = b[3:].decode('utf-8') if bom else b.decode('utf-8')
    recs, rec, fld, i, n, inq, eols = [], [], [], 0, len(t), False, set()
    while i < n:
        ch = t[i]
        if inq:
            if ch == '"':
                if i + 1 < n and t[i + 1] == '"': fld.append('"'); i += 2; continue
                inq = False; i += 1; continue
            fld.append(ch); i += 1; continue
        if ch == '"' and not fld: inq = True; i += 1; continue
        if ch == ',': rec.append(''.join(fld)); fld = []; i += 1; continue
        if ch == '\r' and i + 1 < n and t[i + 1] == '\n':
            rec.append(''.join(fld)); recs.append(rec); rec, fld = [], []; eols.add('CRLF'); i += 2; continue
        if ch == '\n':
            rec.append(''.join(fld)); recs.append(rec); rec, fld = [], []; eols.add('LF'); i += 1; continue
        fld.append(ch); i += 1
    if fld or rec: rec.append(''.join(fld)); recs.append(rec)
    assert not inq, 'unterminated quote'
    return bom, eols, recs
def ser(recs, bom, eol):
    def q(f):
        return '"' + f.replace('"', '""') + '"' if any(c in f for c in ',"\r\n') else f
    s = ''.join(','.join(q(f) for f in r) + eol for r in recs)
    return (b'\xef\xbb\xbf' if bom else b'') + s.encode('utf-8')
R = {}
pb = open(W + r'\facts_patch_v20.csv', 'rb').read(); pbom, peol, P = parse(pb)
H = P[0]; R['patch'] = dict(bom=pbom, eol=sorted(peol), header=H, n=len(P) - 1, bad=[i + 1 for i, r in enumerate(P) if len(r) != len(H)])
PD = [dict(zip(H, r)) for r in P[1:]]
add_bad = []
for d in PD:
    if d['종류'] in ('행 추가', '행 대체'):
        _, _, rr = parse(d['새값'].encode('utf-8'))
        if len(rr) != 1 or len(rr[0]) != 8: add_bad.append((d['id'], [len(x) for x in rr]))
R['patch']['add_or_replace_new_bad'] = add_bad
# 독립 적용
OUTS = {}
for name, eol in [('facts.csv', '\r\n'), ('facts_supplement.csv', '\r\n'), ('facts_supplement_R12.csv', '\n')]:
    bom, eols, recs = parse(open(FD + '\\' + name, 'rb').read())
    hdr = recs[0]; body = recs[1:]
    for d in [x for x in PD if x['파일'] == name]:
        k = d['종류']
        if k == '형식':
            for j, r in enumerate(body):
                if r[0] == d['id'] and len(r) != len(hdr): body[j] = r[:len(hdr) - 1] + [','.join(r[len(hdr) - 1:])]
        elif k in ('설정', '치환', '덧붙임'):
            col = hdr.index(d['열']); hit = [j for j, r in enumerate(body) if r[0] == d['id']]
            assert len(hit) == 1, (name, d['id'], len(hit)); body[hit[0]][col] = d['새값']
        elif k == '행 추가':
            body.append(parse(d['새값'].encode('utf-8'))[2][0])
        elif k == '행 대체':
            old, new = d['id'].split('→'); hit = [j for j, r in enumerate(body) if r[0] == old]
            assert len(hit) == 1; body[hit[0]] = parse(d['새값'].encode('utf-8'))[2][0]
    out_b = ser([hdr] + body, bom, eol)
    wk = open(WK + '\\' + name, 'rb').read()
    wbom, weol, wrecs = parse(wk)
    R[name] = dict(src_bom=bom, src_eol=sorted(eols), work_bom=wbom, work_eol=sorted(weol), work_bad=[i + 1 for i, r in enumerate(wrecs) if len(r) != 8],
                   indep_equal_bytes=(out_b == wk), n_rows=len(wrecs) - 1)
R['ok'] = (not R['patch']['bad'] and not add_bad and all(R[n]['indep_equal_bytes'] and not R[n]['work_bad'] for n in ('facts.csv', 'facts_supplement.csv', 'facts_supplement_R12.csv'))
           and R['facts.csv']['work_bom'] and R['facts.csv']['work_eol'] == ['CRLF'] and R['facts_supplement.csv']['work_bom'] and R['facts_supplement.csv']['work_eol'] == ['CRLF']
           and (not R['facts_supplement_R12.csv']['work_bom']) and R['facts_supplement_R12.csv']['work_eol'] == ['LF'])
json.dump(R, open(W + r'\verify\v2_facts_indep.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1)[:2500])
