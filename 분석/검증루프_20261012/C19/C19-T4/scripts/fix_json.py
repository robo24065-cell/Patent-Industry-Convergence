import json, os
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(D, 'cache', 'final_overrides.json')
s = open(p, encoding='utf-8').read()
s = s.replace('\\', '\\\\')
d = json.loads(s)
json.dump(d, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(len(d))
