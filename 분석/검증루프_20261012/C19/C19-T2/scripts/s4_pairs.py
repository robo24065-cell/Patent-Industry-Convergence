# -*- coding: utf-8 -*-
r"""C19-T2 s4: 정규화 점검 표본 200쌍(과병합)·200쌍(누락) + 제공 해외 메타와의 대조.
 (i) 과병합: F1 충돌 100쌍 + F1x 추가 충돌 100쌍(seed 20261017). 쌍마다 구조 점검(같은 국가·같은 일련번호·같은 추정연도·자릿수/표기 차이만)과
     메타 대조: 같은 국가 해외 메타 행 중 (a) 자기 번호(rgst_no 정수, 또는 family 안 자기 표기)가 그 번호인 행, (b) family 문자열에 두 표기를 모두 적은 행 수.
 (ii) 누락: F1x 정규화 뒤에도 같은 pid 안에 같은 국가·같은 yr·같은 끝 6자리인 토큰쌍 → 유형별 수와 표본 200쌍.
 (iii) 위험 패턴: JP 8자리 0시작 중 값 < 2,000,000(레이와 공개번호 오독 가능), US 10자리·11자리 비율, KR 자기 문서 토큰.
 입력: cache/s1_tokens_keys.parquet, 메타(읽기만). 출력: pairs_check.csv, cache/s4_pairs.json, cache/s4_omission_examples.csv"""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
import numpy as np, pandas as pd

tk = pd.read_parquet(J(C, 's1_tokens_keys.parquet'))
R = {}
rng = np.random.default_rng(20261017)


def pairs_of(key):
    d = tk[tk.duplicated(['pid', key], keep=False)].sort_values(['pid', key, 'tok'])
    g = d.groupby(['pid', key])
    a = g.tok.first(); b = g.tok.last(); yr = g.yr.first(); cc = g.cc.first()
    return pd.DataFrame({'pid': a.index.get_level_values(0), 'key': a.index.get_level_values(1), 'tok_a': a.values, 'tok_b': b.values, 'yr': yr.values, 'cc': cc.values})


P1 = pairs_of('k1')
Px = pairs_of('kx')
Px = Px.merge(P1[['pid', 'tok_a', 'tok_b']], on=['pid', 'tok_a', 'tok_b'], how='left', indicator=True)
Px = Px[Px._merge == 'left_only'].drop(columns='_merge')
R['F1_pairs'] = int(len(P1)); R['F1x_extra_pairs'] = int(len(Px))
R['F1_pairs_by_cc'] = P1.cc.value_counts().to_dict(); R['F1x_extra_by_cc'] = Px.cc.value_counts().to_dict()
s1 = P1.iloc[rng.choice(len(P1), 100, replace=False)].assign(rule='F1')
sx = Px.iloc[rng.choice(len(Px), 100, replace=False)].assign(rule='F1x_extra')
S = pd.concat([s1, sx], ignore_index=True)


def serial(tok):
    cc, r = tok[:2], tok[2:]
    if cc == 'US' and len(r) == 11 and r[4] == '0':
        return r[:4] + r[5:]
    if cc == 'JP' and len(r) == 8 and r[0] != '0':
        return str(1988 + int(r[:2])) + r[2:]
    return r.lstrip('0')


S['same_cc'] = S.tok_a.str[:2] == S.tok_b.str[:2]
S['same_serial'] = [serial(a) == serial(b) for a, b in zip(S.tok_a, S.tok_b)]
S['len_a'] = S.tok_a.str.len() - 2; S['len_b'] = S.tok_b.str.len() - 2
S['form'] = S.cc + ':' + S[['len_a', 'len_b']].astype(str).agg('/'.join, axis=1)
# 메타 대조: 표본 토큰이 해외 메타 family 문자열에 나오는 행
toks = sorted(set(S.tok_a) | set(S.tok_b))
con = con3()
con.register('st', pd.DataFrame({'t': toks}))
hits = con.execute(f"""with f as (select cc, appl_no, rgst_no, unnest(string_split(family,'|')) t from read_parquet('{META}')
      where cc in ('US','JP','EP') and family is not null)
      select f.cc, f.appl_no, f.rgst_no, f.t from f join st using(t)""").df()
R['meta_rows_hit'] = int(len(hits))
byrow = hits.groupby(['cc', 'appl_no']).t.apply(set)
rg = hits.drop_duplicates(['cc', 'appl_no'])[['cc', 'appl_no', 'rgst_no']]
rg['rg_int'] = pd.to_numeric(rg.rgst_no, errors='coerce')
both, own = [], []
t2rows = hits.groupby('t').apply(lambda x: set(zip(x.cc, x.appl_no)), include_groups=False).to_dict()
rgmap = {(c, a): v for c, a, v in zip(rg.cc, rg.appl_no, rg.rg_int)}
for a, b in zip(S.tok_a, S.tok_b):
    ra, rb = t2rows.get(a, set()), t2rows.get(b, set())
    both.append(len(ra & rb))
    num = int(serial(a)) if serial(a).isdigit() else -1
    own.append(int(any(rgmap.get(x) == num for x in (ra | rb))))
S['meta_rows_with_a'] = [len(t2rows.get(a, set())) for a in S.tok_a]
S['meta_rows_with_b'] = [len(t2rows.get(b, set())) for b in S.tok_b]
S['meta_rows_listing_both'] = both
S['meta_own_rgst_match'] = own
S['verdict'] = np.where(S.same_cc & S.same_serial, '같은 문서(표기 차이)', '확인 필요')
S.to_csv(J(W, 'pairs_check.csv'), index=False, encoding='utf-8-sig')
for r_, g in S.groupby('rule'):
    R[f'sample_{r_}'] = dict(n=int(len(g)), same_cc_serial=int((g.same_cc & g.same_serial).sum()), yr_equal=int(len(g)),
                             any_meta_hit=int(((g.meta_rows_with_a + g.meta_rows_with_b) > 0).sum()),
                             meta_listing_both=int((g.meta_rows_listing_both > 0).sum()), own_rgst_match=int(g.meta_own_rgst_match.sum()),
                             forms=g.form.value_counts().to_dict())
out('sample', {k: v for k, v in R.items() if k.startswith('sample')})
# 메타 전체: 해외 기록 family 안에서도 같은 이중 표기가 나오는가(같은 행 안 F1·F1x 충돌 비율) — 제공 데이터 전반 특성
fam_meta = con.execute(f"""select cc, count(*) n,
      avg(case when family is null or family='' then null else (len(string_split(family,'|')) - len(list_distinct(list_transform(string_split(family,'|'), x -> substr(x,1,2) || ltrim(substr(x,3),'0'))))) end) mean_dup_f1,
      avg(case when family is null or family='' then null else (len(list_distinct(list_transform(string_split(family,'|'), x -> substr(x,1,2) || ltrim(substr(x,3),'0')))) < len(list_distinct(string_split(family,'|'))))::int end) share_rec_dup_f1
      from read_parquet('{META}') group by cc order by cc""").df()
R['meta_family_dup_by_cc'] = fam_meta.to_dict('records')
out('meta fam dup', fam_meta.to_string())
# (ii) 누락 후보
d = tk.drop_duplicates(['pid', 'kx']).copy()
d['t6'] = d.tok.str[-6:]
cand = d[d.duplicated(['pid', 'cc', 'yr', 't6'], keep=False)].sort_values(['pid', 'cc', 'yr', 't6', 'tok'])
g = cand.groupby(['pid', 'cc', 'yr', 't6'])
om = pd.DataFrame({'tok_a': g.tok.first(), 'tok_b': g.tok.last(), 'n': g.size()}).reset_index()
om['form'] = om.cc + ':' + (om.tok_a.str.len() - 2).astype(str) + '/' + (om.tok_b.str.len() - 2).astype(str)
R['omission_candidate_groups'] = int(len(om)); R['omission_forms'] = om.form.value_counts().head(15).to_dict()
om_s = om.iloc[rng.choice(len(om), min(200, len(om)), replace=False)] if len(om) else om
om_s.to_csv(J(C, 's4_omission_examples.csv'), index=False, encoding='utf-8-sig')
R['omission_tokens_share_of_all'] = float((om.n - 1).sum() / len(tk))
# (iii) 위험 패턴
r = tk.tok.str[2:]
jp0 = (tk.cc == 'JP') & (r.str.len() == 8) & r.str.startswith('0')
R['jp8_leading0'] = int(jp0.sum()); R['jp8_leading0_lt2M'] = int((jp0 & (pd.to_numeric(r.where(jp0), errors='coerce') < 2_000_000)).sum())
R['us_len_counts'] = tk[tk.cc == 'US'].tok.str.len().sub(2).value_counts().to_dict()
R['kr_tokens'] = int((tk.cc == 'KR').sum()); R['kr_len_counts'] = tk[tk.cc == 'KR'].tok.str.len().sub(2).value_counts().to_dict()
R['cc_share_tokens'] = tk.cc.value_counts(normalize=True).head(8).round(4).to_dict()
jdump(R, J(C, 's4_pairs.json'))
out('done', {k: R[k] for k in ['omission_candidate_groups', 'omission_forms', 'jp8_leading0', 'jp8_leading0_lt2M', 'kr_tokens', 'kr_len_counts']})
