# 각 diff 블록(v4->v7, v8->now)의 새 행을 현재 원고(88abe4e7) 행 번호로 사상하고 현재 문장을 붙인다
import difflib, json, sys, os
sys.stdout.reconfigure(encoding="utf-8")
W = sys.argv[1]; C = os.path.join(W, 'cache')
seq = ['manuscript_v3_before_C03','manuscript_for_v5','manuscript_for_v6','manuscript_for_v7','manuscript_for_v8','manuscript_v3_now_88abe4e7']
T = {n: open(os.path.join(C, n+'.md'), encoding='utf-8').read().split('\n') for n in seq}
def linemap(a, b):
    m = {}
    for op,i1,i2,j1,j2 in difflib.SequenceMatcher(None, T[a], T[b], autojunk=False).get_opcodes():
        if op == 'equal' or (op == 'replace' and i2-i1 == j2-j1):
            for k in range(i2-i1): m[i1+k] = j1+k
    return m
maps = {seq[k]: linemap(seq[k], seq[k+1]) for k in range(len(seq)-1)}
def to_now(ver, idx0):
    k = seq.index(ver); i = idx0
    while k < len(seq)-1:
        i = maps[seq[k]].get(i)
        if i is None: return None
        k += 1
    return i
blocks = json.load(open(os.path.join(C, 'blocks_all_88abe4e7.json'), encoding='utf-8'))
now = T[seq[-1]]
out = []
for n, b in enumerate(blocks):
    src = b['pair'].split('->')[1]
    rows = []
    for j in range(b['new'][0]-1, b['new'][1]):
        c = to_now(src, j)
        rows.append(dict(new_line=j+1, now_line=(c+1 if c is not None else None), now_text=(now[c] if c is not None else None),
                         same_as_now=(c is not None and now[c] == T[src][j])))
    if b['op'] == 'delete':
        # 삭제 블록: 앞 행의 현재 위치만
        c = to_now(src, b['new'][0]-2)
        rows.append(dict(new_line=None, now_line=(c+1 if c is not None else None) and c+1, now_text='(삭제 블록; 직전 행 현재 위치 %s)' % (c+1 if c is not None else None), same_as_now=None))
    out.append(dict(block=f'B{n:02d}', pair=b['pair'], op=b['op'], old=b['old'], new=b['new'], now=rows))
json.dump(out, open(os.path.join(C, 'blocks_now_map.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for o in out:
    if o['block'] > 'B49': continue
    for r in o['now']:
        print(o['block'], r['new_line'], '->', r['now_line'], 'SAME' if r['same_as_now'] else 'CHANGED_LATER', '|', (r['now_text'] or '')[:3000])
