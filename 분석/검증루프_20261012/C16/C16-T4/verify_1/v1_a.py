import csv, io, re, sys, json, hashlib
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
R = AN + r"\보고서_20261012"; T = AN + r"\검증루프_20261012\C16\C16-T4"
FILES = ['facts.csv', 'facts_supplement.csv', 'facts_supplement_R12.csv']
out = {}
# 1-2) structure
def raw_rows(p):
    return list(csv.reader(io.StringIO(open(p, encoding='utf-8-sig').read())))
for f in FILES:
    for lab, d in [('orig', R + r'\facts'), ('work', T + r'\facts_work')]:
        p = d + '\\' + f
        rr = raw_rows(p); h = rr[0]
        bad = [(i + 1, len(r)) for i, r in enumerate(rr[1:]) if len(r) != len(h)]
        short = [x for x in bad if x[1] < len(h)]
        ids = [r[0] for r in rr[1:] if r]
        dup = sorted(set(i for i in ids if ids.count(i) > 1))
        out[f'{lab}:{f}'] = dict(sha1=hashlib.sha1(open(p, 'rb').read()).hexdigest()[:8], nrows=len(rr) - 1, ncol=len(h), badcols=bad[:20], n_bad=len(bad), n_short=len(short), dup_ids=dup)
# diff
def ld(p):
    rr = raw_rows(p); h = rr[0]
    rows = [r[:len(h)-1] + [','.join(r[len(h)-1:])] if len(r) >= len(h) else r + ['']*(len(h)-len(r)) for r in rr[1:] if r]
    return pd.DataFrame(rows, columns=h)
patch = pd.read_csv(T + r'\facts_patch_consolidated.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig')
diffs = []
for f in FILES:
    a = ld(R + r'\facts\\' + f)
    b = ld(T + r'\facts_work\\' + f)
    assert list(a.columns) == list(b.columns), (f, a.columns, b.columns)
    # map id->rows (allow duplicates by position within id)
    ga = {}; gb = {}
    for i, r in a.iterrows(): ga.setdefault(r['id'], []).append(r)
    for i, r in b.iterrows(): gb.setdefault(r['id'], []).append(r)
    for k in set(ga) | set(gb):
        la, lb = ga.get(k, []), gb.get(k, [])
        if len(la) != len(lb):
            diffs.append(dict(file=f, id=k, col='(rows)', old=len(la), new=len(lb)))
            if la and lb:
                pass
            continue
        for ra, rb in zip(la, lb):
            for c in a.columns:
                if ra[c] != rb[c]:
                    diffs.append(dict(file=f, id=k, col=c, old=ra[c], new=rb[c]))
    # row order change
    out[f'order_same:{f}'] = [x for x in a['id']] == [x for x in b['id'] if x in set(a['id'])]
# compare diffs to patch log
plog = set()
final_new = {}
for _, r in patch.iterrows():
    ids = r['id'].split('→')
    for i in ids: plog.add((r['파일'], i, r['열']))
    final_new[(r['파일'], ids[-1], r['열'])] = r['새값']
unlogged = []
mismatch_final = []
for d in diffs:
    if d['col'] == '(rows)':
        key_any = any((d['file'], d['id'], c) in plog for c in ['(행)', '(행 추가)', '(행 대체)']) or any(k[0] == d['file'] and k[1] == d['id'] for k in plog)
        if not key_any: unlogged.append(d)
        continue
    if (d['file'], d['id'], d['col']) not in plog and not any(k[0] == d['file'] and k[1] == d['id'] and k[2] in ('(행)', '(형식)', '전체', '(행 추가)', '(행 대체)') for k in plog):
        unlogged.append(d)
    else:
        k = (d['file'], d['id'], d['col'])
        if k in final_new and final_new[k] != d['new']:
            mismatch_final.append(dict(**d, logged_new=final_new[k]))
out['n_diff_cells'] = len(diffs)
out['n_diff_rows'] = len(set((d['file'], d['id']) for d in diffs))
out['unlogged'] = unlogged
out['final_value_mismatch'] = mismatch_final[:20]
out['n_final_value_mismatch'] = len(mismatch_final)
out['patch_cols'] = patch['열'].value_counts().to_dict()
out['patch_kinds'] = patch['종류'].value_counts().to_dict()
# logged items that produced no diff
dset = set((d['file'], d['id'], d['col']) for d in diffs) | set((d['file'], d['id']) for d in diffs)
nodiff = []
for _, r in patch.iterrows():
    i = r['id'].split('→')[-1]
    if (r['파일'], i, r['열']) not in dset and (r['파일'], i) not in dset:
        nodiff.append((r['파일'], r['id'], r['열'], r['종류']))
out['logged_without_diff'] = nodiff
pd.DataFrame(diffs).to_csv(T + r'\verify_1\v1_cell_diff.csv', index=False, encoding='utf-8-sig')
json.dump(out, open(T + r'\verify_1\v1_a_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:6000])
