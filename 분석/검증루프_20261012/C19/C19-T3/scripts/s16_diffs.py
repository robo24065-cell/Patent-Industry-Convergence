import difflib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
a = open(MANU, encoding='utf-8').read().split('\n')
for k in ['B1', 'B2', 'A13']:
    b = open(os.path.join(W, 'build_test', f'ms_{k}.md'), encoding='utf-8').read().split('\n')
    d = list(difflib.unified_diff(a, b, 'manuscript_v3.md(v20 40406e29)', f'ms_{k}.md', n=0, lineterm=''))
    open(os.path.join(W, 'build_test', f'diff_v20_to_{k}.diff'), 'w', encoding='utf-8').write('\n'.join(d) + '\n')
    out(k, len(d))
