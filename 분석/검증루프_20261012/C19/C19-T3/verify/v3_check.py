# V3: '사용 안 함' 표본 20개 스크립트 전문 점검 — US·EP 행을 담은 입력(메타·fam_reg·base_rows·alt_assign·rec_conc·txt_assign·ep_firm·fam_link·panel)을 읽는 줄과 cc 필터 줄을 모두 뽑는다
import os, sys, re, json
sys.stdout.reconfigure(encoding='utf-8')
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AN = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석'
L = json.load(open(os.path.join(W, 'verify', 'v3_list.json'), encoding='utf-8'))
SRCS = re.compile(r"_shared[\/]+meta|meta[\/]\*\.parquet|\bMETA\b|fam_reg|base_rows|alt_assign|alt_\*|rec_conc|txt_assign|ep_firm|fam_link|panel_ksic_cc_year|m1_counts|m1_origin|fdoc|family", re.I)
CC = re.compile(r"cc\s*(=|==|<>|!=|in)\s*\(?['\"]|\.cc\s*==|cc\.isin|'US'|'EP'|\"US\"|\"EP\"")
rep = {}
for s in L['sample']:
    p = os.path.join(AN, s); t = open(p, encoding='utf-8', errors='replace').read().split('\n')
    hits = [(i + 1, x.strip()[:200]) for i, x in enumerate(t) if (SRCS.search(x) or CC.search(x)) and not x.strip().startswith('#')]
    rep[s] = dict(n_lines=len(t), hits=hits)
json.dump(rep, open(os.path.join(W, 'verify', 'v3_hits.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for s, v in rep.items():
    print('=====', s, v['n_lines'])
    for h in v['hits'][:25]: print('   ', h)
