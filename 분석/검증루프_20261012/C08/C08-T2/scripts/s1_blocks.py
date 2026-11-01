# 연속 판 diff 블록을 행 단위로 뽑고, 행 안 변경 부분만 [-old-]{+new+}로 표시
import difflib, json, sys, os, re
sys.stdout.reconfigure(encoding="utf-8")
W = sys.argv[1]
C = os.path.join(W, 'cache')
seq = ['manuscript_v3_before_C03','manuscript_for_v5','manuscript_for_v6','manuscript_for_v7','manuscript_for_v8','manuscript_v3_start']
def rd(n): return open(os.path.join(C, n+'.md'), encoding='utf-8').read().split('\n')
def tok(s): return re.findall(r'\s+|[^\s]+', s)
def inline(a, b):
    ta, tb = tok(a), tok(b); out=[]
    for op,i1,i2,j1,j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if op=='equal':
            seg=''.join(ta[i1:i2]); out.append(seg if len(seg)<60 else seg[:25]+' … '+seg[-25:])
        else:
            if i2>i1: out.append('[-'+''.join(ta[i1:i2])+'-]')
            if j2>j1: out.append('{+'+''.join(tb[j1:j2])+'+}')
    return ''.join(out)
blocks=[]
for k in range(len(seq)-1):
    A, B = rd(seq[k]), rd(seq[k+1])
    sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
    for op,i1,i2,j1,j2 in sm.get_opcodes():
        if op=='equal': continue
        bid = f'{k}-{len(blocks)}'
        blocks.append(dict(pair=f'{seq[k]}->{seq[k+1]}', op=op, old=(i1+1,i2), new=(j1+1,j2), old_text=A[i1:i2], new_text=B[j1:j2]))
json.dump(blocks, open(os.path.join(C,'blocks_all.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=1)
sel = sys.argv[2] if len(sys.argv)>2 else None
for n,b in enumerate(blocks):
    if sel and sel not in b['pair'].split('->')[0]: continue
    print(f"### B{n:02d} {b['pair']} {b['op']} old{b['old']} new{b['new']}")
    if b['op']=='replace' and len(b['old_text'])==len(b['new_text']):
        for a,c in zip(b['old_text'], b['new_text']): print('  ~ '+inline(a,c))
    else:
        for a in b['old_text']: print('  - '+a)
        for c in b['new_text']: print('  + '+c)
