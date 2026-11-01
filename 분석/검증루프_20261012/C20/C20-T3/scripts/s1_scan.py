# C20-T3 s1: 원본(backup) + facts_patch_v20 + 알려진 4칸 → 상태 S. S = 현재 적용본인지 확인하고,
# 범위 P(패치 130행 대상) ∪ A(v21 부록 ID) 행마다 PREREG 3절 C1–C4를 기계 대조한다. 출력: cache/s1_scan.json, cache/s1_scan_rows.csv
import os, sys, csv, io, json, re, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib20 import *

for f in FILES:
    assert sha1(FACTS_ORIG + '\\' + f) == ORIG_SHA1[f], f
assert sha1(PATCH20) == PATCH20_SHA1
ms = manu_lines()
APP, st, en = appendix(ms)
SEC = sections(ms, st)

orig = {f: read_rows(FACTS_ORIG + '\\' + f) for f in FILES}
patch = read_rows(PATCH20)
PH, PR = patch[0], patch[1:]
assert PH == ['파일', 'id', '열', '현재값', '새값', '근거', '출처바퀴', '종류'] and len(PR) == 130


def apply(base, prows, extra4=True):
    st_ = {f: [list(r) for r in base[f]] for f in FILES}
    prob = []
    for p in prows:
        f, pid, col, cur, new, ev, src, kind = p
        rows = st_[f]; H = rows[0]
        if kind == '형식':
            for j, r in enumerate(rows[1:], 1):
                if r[0] == pid and len(r) != len(H):
                    rows[j] = r[:len(H) - 1] + [','.join(r[len(H) - 1:])]
            continue
        if kind == '변경 없음' or col == '-':
            continue
        if col in H:
            idx = [k for k, r in enumerate(rows) if r and r[0] == pid]
            if len(idx) != 1:
                prob.append((pid, 'id수', len(idx))); continue
            c = H.index(col)
            if rows[idx[0]][c] != cur:
                prob.append((pid, col, '현재값 불일치'))
            rows[idx[0]][c] = new
        elif col == '(행 추가)':
            rows.append(next(csv.reader([new])))
        elif col == '(행 대체)':
            a, b = pid.split('→')
            idx = [k for k, r in enumerate(rows) if r and r[0] == a]
            assert len(idx) == 1
            rows[idx[0]] = next(csv.reader([new]))
        else:
            prob.append((pid, '알 수 없는 열', col))
    if extra4:
        H = st_['facts.csv'][0]; c = H.index('수치')
        for k, (old, new) in KNOWN4.items():
            idx = [j for j, r in enumerate(st_['facts.csv']) if r[0] == k]
            assert len(idx) == 1
            r = st_['facts.csv'][idx[0]]
            assert r[c] == old, (k, r[c])
            r[c] = new
    return st_, prob


S, prob = apply(orig, PR, True)
S0, _ = apply(orig, PR, False)
now = {f: read_rows(FACTS_NOW + '\\' + f) for f in FILES}
same_now = {f: S[f] == now[f] for f in FILES}

# 범위
P = set()
for p in PR:
    pid = p[1].split('→')[-1]
    P.add((p[0], pid))
recs = {}
for f in FILES:
    H = S[f][0]
    for r in S[f][1:]:
        recs.setdefault(r[0], []).append((f, dict(zip(H, r))))
origrec = {}
for f in FILES:
    H = orig[f][0]
    for r in orig[f][1:]:
        rr = r[:len(H) - 1] + [','.join(r[len(H) - 1:])] if len(r) > len(H) else r
        origrec[(f, r[0])] = dict(zip(H, rr))
dup = {k: [x[0] for x in v] for k, v in recs.items() if len(v) > 1}
A = set()
for k in APP:
    for f, _ in recs.get(k, []):
        A.add((f, k))
app_missing = [k for k in APP if k not in recs]
scope = sorted(P | A)

# 패치가 바꾼 칸
pchg = {}
for p in PR:
    pchg.setdefault((p[0], p[1].split('→')[-1]), []).append(p)

out = []
for f, i in scope:
    rr = [d for ff, d in recs.get(i, []) if ff == f]
    if not rr:
        out.append(dict(파일=f, id=i, 범위='P', 상태='S에 없음')); continue
    d = rr[0]
    val, unit, frag, cond, srccell = d['수치'], d['단위'], d['정확한 문장 조각'], d['조건·경고'], d['출처 파일 경로']
    pct = ('%' in unit) or ('%' in frag)
    vt = toks(val); ft = toks(frag); ct = toks(cond)
    # C1
    c1_miss = []; c1_how = []; c1_signonly = []
    for t in vt:
        how, hit = match_any(t, ft, pct)
        if how:
            c1_how.append(f'{t}:{how}')
        else:
            h2, _ = match_any(t, ft, pct, signless=True)
            if h2:
                c1_signonly.append(t)
            c1_miss.append(t)
    o = origrec.get((f, i))
    c1_origin = ''
    if c1_miss:
        if o is None:
            c1_origin = '신규 행'
        else:
            ot = toks(o['정확한 문장 조각']); ov = toks(o['수치'])
            miss_o = [t for t in ov if not match_any(t, ot, pct)[0]]
            c1_origin = '원본 기원' if miss_o else '패치 기원'
    # C2
    c2 = {}
    if i in APP:
        av = APP[i]['val']
        for ks in (False, True):
            fts = set(toks(frag, ks)); ats = set(toks(av, ks)); cts = set(toks(cond, ks))
            c2[f'A{int(ks)}'] = sorted(fts - ats); c2[f'B{int(ks)}'] = sorted(ats - (fts | cts))
        # 수치 토큰 ↔ 부록 값 칸 또는 쓰인 절 본문
        body = []
        for sname in re.split(r',\s*', APP[i]['sec']):
            sname = sname.strip()
            if sname in SEC:
                a, b = SEC[sname]; body += ms[a:b]
        bt = toks('\n'.join(body)); at_ = toks(av)
        c2_val_miss = []
        for t in vt:
            if not match_any(t, at_, pct)[0] and not match_any(t, bt, pct)[0]:
                c2_val_miss.append(t)
        c2['수치_원고없음'] = c2_val_miss
        c2['절없음'] = [s for s in re.split(r',\s*', APP[i]['sec']) if s.strip() not in SEC]
    # C3
    paths = resolve_paths(srccell)
    pst = []; srcnums = []
    for raw, p in paths:
        if p is None:
            pst.append(f'{raw}=해석 불가'); continue
        r = src_numbers(p)
        pst.append(f'{os.path.basename(p)}={r[0]}')
        if r[0] == '읽음':
            srcnums.append(r)
    readable = bool(srcnums)
    def found(t):
        for r in srcnums:
            h = src_has(t, r)
            if h:
                return h
        return ''
    c3_val_miss = [t for t in vt if readable and not found(t)]
    c3_frag_miss = [t for t in dict.fromkeys(ft) if readable and not found(t)]
    # C4
    c4 = []
    for p in pchg.get((f, i), []):
        if p[2] == '정확한 문장 조각' and o is not None:
            oldf = toks(p[3]); newf = toks(p[4])
            for t in toks(o['수치']):
                if match_any(t, oldf, pct)[0] and not match_any(t, newf, pct)[0]:
                    if not any(q[2] == '수치' for q in pchg.get((f, i), [])) and i not in KNOWN4:
                        c4.append(f'수치 {t} 옛 조각에만')
            changed = [t for t in dict.fromkeys(newf) if t not in oldf]
            if readable:
                nf = [t for t in changed if not found(t)]
                if nf:
                    c4.append('바뀐 조각 숫자 출처 칸에서 못 찾음: ' + ' '.join(nf))
            elif changed:
                c4.append('바뀐 조각 숫자 있음, 출처 칸 읽을 수 없음')
    flags = []
    if c1_miss: flags.append('C1')
    if i in APP and (c2['A0'] or c2['B0']): flags.append('C2')
    if i in APP and c2['수치_원고없음']: flags.append('C2v')
    if c3_val_miss: flags.append('C3v')
    if not readable: flags.append('C3x')
    if c4: flags.append('C4')
    out.append(dict(파일=f, id=i, 범위=('P' if (f, i) in P else '') + ('A' if (f, i) in A else ''), 부록행=APP.get(i, {}).get('line', ''),
                    수치=val, 단위=unit, C1_일치=' '.join(c1_how), C1_미일치=' '.join(c1_miss), C1_부호만=' '.join(c1_signonly), C1_기원=c1_origin,
                    C2_A=' '.join(c2.get('A0', [])), C2_B=' '.join(c2.get('B0', [])), C2_A_부호=' '.join(c2.get('A1', [])), C2_B_부호=' '.join(c2.get('B1', [])),
                    C2_수치원고없음=' '.join(c2.get('수치_원고없음', [])), C2_절없음=' '.join(c2.get('절없음', [])),
                    C3_파일=' ; '.join(pst), C3_수치원천없음=' '.join(c3_val_miss), C3_조각원천없음=' '.join(c3_frag_miss), C3_조각토큰수=len(set(ft)),
                    C4=' | '.join(c4), 플래그=' '.join(flags)))

S_sum = dict(same_now=same_now, apply_problems=prob, n_scope=len(scope), n_P=len(P), n_A=len(A), app_ids=len(APP), app_missing=app_missing,
             dup_ids=dup, flags={k: sum(1 for r in out if k in r.get('플래그', '').split()) for k in ['C1', 'C2', 'C2v', 'C3v', 'C3x', 'C4']},
             c1_origin={k: sum(1 for r in out if r.get('C1_기원') == k) for k in ['원본 기원', '패치 기원', '신규 행']})
jdump(S_sum, 's1_scan.json')
with open(C + r'\s1_scan_rows.csv', 'w', encoding='utf-8-sig', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[1].keys()), extrasaction='ignore'); w.writeheader(); w.writerows(out)
# 상태 S 보관(다음 단계 입력)
json.dump({f: S[f] for f in FILES}, open(C + r'\s1_state_S.json', 'w', encoding='utf-8'), ensure_ascii=False)
print(json.dumps(S_sum, ensure_ascii=False, indent=1))
