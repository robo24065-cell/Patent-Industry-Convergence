# C16-T3 s5: 정규식 오분류 유형별 모집단 규모(전수, 공개연도별) — 표본 감사 보조
# 사용: python -I s5_regex_pop.py <meta_glob> <cache_dir>
# FP 후보: 출원인1이 한자 2–3자 이름인데 정규식이 조직(단일 글자 院/所/厂/局/会 때문)
# FN 후보: 출원인1에 部队/实验室/商行/合作社/经营部/门市部/工作室/学校/植物园/站 이 있는데 정규식 비조직, 또는 키 길이<2로 탈락
import sys, os, re, json
import pandas as pd, numpy as np, duckdb
M, C = sys.argv[1].replace('\\', '/'), sys.argv[2]
sys.stdout.reconfigure(encoding='utf-8')
d = pd.read_parquet(os.path.join(C, 'cn_docs.parquet'), columns=['appl_no', 'py', 'pub', 'app1', 'app1_org', 'n_orgkey', 'foreign'])
d = d[d.pub.notna() & d.py.between(2015, 2024)].copy()
u = pd.read_parquet(os.path.join(C, 'cn_names.parquet')).set_index('nm')
han23 = d.app1.str.fullmatch(r'[一-鿿]{2,3}')
d['fp_cand'] = han23 & d.app1_org
FNW = r'部队|实验室|商行|合作社|经营部|门市部|工作室|学校|植物园|站$|银行|事务所'
d['fn_lex'] = d.app1.str.contains(FNW) & ~d.app1_org
d['fn_key'] = d.app1.map(u.org).fillna(False).astype(bool) & ~d.app1_org     # 정규식은 맞췄지만 키 길이<2로 탈락
d['person_like'] = han23 & ~d.app1_org
g = d.groupby('py').agg(n=('appl_no', 'size'), fp_cand=('fp_cand', 'mean'), fn_lex=('fn_lex', 'mean'), fn_key=('fn_key', 'mean'),
                        person_like=('person_like', 'mean'), app1_org=('app1_org', 'mean'))
g['app1_org_corrected'] = g.app1_org - g.fp_cand + g.fn_lex + g.fn_key
print(g.round(5).to_string())
ex = {int(y): {'fp': d[(d.py == y) & d.fp_cand].app1.sample(min(15, int(((d.py == y) & d.fp_cand).sum())), random_state=1).tolist(),
               'fn_lex': d[(d.py == y) & d.fn_lex].app1.sample(min(10, int(((d.py == y) & d.fn_lex).sum())), random_state=1).tolist()} for y in [2015, 2019, 2024]}
print(json.dumps(ex, ensure_ascii=False))
g.to_csv(os.path.join(C, 's5_regex_pop.csv'), encoding='utf-8-sig')
json.dump(ex, open(os.path.join(C, 's5_examples.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
