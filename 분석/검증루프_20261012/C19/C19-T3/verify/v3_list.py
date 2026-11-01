import csv, json, sys, os, re, random
sys.stdout.reconfigure(encoding='utf-8')
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(W, 'scripts'))
import importlib.util
spec = importlib.util.spec_from_file_location('s7', os.path.join(W, 'scripts', 's7_source_map.py'))
src = open(os.path.join(W, 'scripts', 's7_source_map.py'), encoding='utf-8').read()
rules_txt = src[src.find('RULES = ['):src.find(']\ndef classify')+1]
ns = {}
exec('R_MAP=R_LO=NO=""\n' + rules_txt, ns)
RULES = ns['RULES']
def classify(p):
    for pat, scr, use, rr, why in RULES:
        if re.search(pat, p): return use
    return '미분류'
rows = list(csv.DictReader(open(os.path.join(W, 'cache', 's3_sources.csv'), encoding='utf-8-sig')))
scripts = {}
for r in rows:
    use = classify(r['src_abs'])
    if use != '사용 안 함': continue
    for p in json.loads(r['producers']):
        scripts.setdefault(p['s'], set()).add(os.path.basename(r['src_abs'].split(' ')[0]))
L = sorted(scripts)
print(len(L))
samp = sorted(random.Random(20261017).sample(L, min(20, len(L))))
json.dump(dict(all=[(s, sorted(v)) for s, v in scripts.items()], sample=samp), open(os.path.join(W, 'verify', 'v3_list.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for s in samp: print(s, sorted(scripts[s])[:3])
