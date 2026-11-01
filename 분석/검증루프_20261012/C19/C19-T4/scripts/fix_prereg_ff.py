import os
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(D, 'PREREG.txt')
s = open(p, encoding='utf-8').read()
n = s.count('\x0c')
s = s.replace('cache\x0cinal_overrides', 'cache\\final_overrides').replace('scripts\x0cix_json', 'scripts\\fix_json')
open(p, 'w', encoding='utf-8').write(s)
print(n, s.count('\x0c'))
