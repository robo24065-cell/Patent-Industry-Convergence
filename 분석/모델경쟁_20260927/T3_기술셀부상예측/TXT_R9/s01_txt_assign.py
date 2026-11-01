# -*- coding: utf-8 -*-
"""R9 s01: T3 문서 풀(R3/cache_pub/base_rows, 5개국, 공개 ≤2021)의 레코드마다 TXT 5자리 배정(PREREG.txt 정의).
 그룹 = R7 연계표 코드(ESSENTIAL_R7/rec_conc, C27 팔과 동일) → 코드 안 KSIC 1개면 그대로, 2–5개면
   KR: KSIC 해설서 ↔ 발명 명칭 문자 2–3그램 TF-IDF argmax (어휘 적합 = 해설서 + 풀 KR 명칭 20만 표본, 공개 ≤2021)
   해외: 같은 패밀리 KR 문서(풀 안, 같은 코드)의 TXT 최빈 → 없으면 TXT-IPC 표(KR TXT 결과의 (코드,서브그룹)→(코드,메인그룹)→(코드,서브클래스)→코드 최빈)
 제공 라벨은 배정에 쓰지 않는다(일치율 기술통계만).
출력: txt_assign.parquet (cc, appl_no, code, ksic_txt, method), txt_assign_stats.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, duckdb
from sklearn.feature_extraction.text import TfidfVectorizer
HERE = os.path.dirname(os.path.abspath(__file__)); T3 = os.path.dirname(HERE); MC = os.path.dirname(T3)
BASE = os.path.dirname(os.path.dirname(MC))
SRC = os.path.join(T3, 'R3', 'cache_pub', 'base_rows.parquet').replace(os.sep, '/')
RC = os.path.join(T3, 'ESSENTIAL_R7', 'rec_conc.parquet').replace(os.sep, '/')
FAM = os.path.join(MC, '_ideas_R8', 'cache', 'fam_link.parquet').replace(os.sep, '/')
DEF = os.path.join(BASE, '공모전데이터', '외부데이터', 'KSIC해설', 'ksic50_definitions.json')
R8TXT = os.path.join(MC, '_ideas_R8', 'cache', 'kr_assign.parquet').replace(os.sep, '/')
OUT = os.path.join(HERE, 'txt_assign.parquet').replace(os.sep, '/')
t = time.time()
log = lambda *a: print(*a, f'[{time.time() - t:.0f}s]', flush=True)
km = pd.read_csv(os.path.join(T3, 'ESSENTIAL_R7', 'conc_ksic_map.csv'), dtype=str)[['ksic', 'code']]
C2K = km.groupby('code').ksic.apply(sorted).to_dict(); K50 = sorted(km.ksic)
single = {c: v[0] for c, v in C2K.items() if len(v) == 1}; multi = {c: v for c, v in C2K.items() if len(v) > 1}
log('코드', len(C2K), '단일', len(single), '복수', len(multi), '복수 코드 KSIC', sum(len(v) for v in multi.values()))
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false"); con.execute("set temp_directory='" + os.path.join(HERE, 'duck_tmp').replace(os.sep, '/') + "'")
con.register('km', km)
con.execute(f"""create table r as select b.cc, b.appl_no, b.ksic ksic_prov, b.year, rc.code, rc.sg
  from read_parquet('{SRC}') b join read_parquet('{RC}') rc using(cc, appl_no) where rc.code in (select code from km)""")
assert con.sql("select max(year) from r").fetchone()[0] <= 2021
log('풀 레코드(27코드 안)', con.sql("select cc, count(*) from r group by 1 order by 1").fetchall())
# ---- KR 텍스트 배정 ----
defs = json.load(open(DEF, encoding='utf-8'))
dtext = {d['code']: ' '.join((d.get(x) or '') for x in ('codenmKr', 'codeIndex', 'codeDefine', 'codeExample')) for d in defs}
assert set(K50) <= set(dtext)
kr = con.sql(f"""select r.appl_no, r.code, coalesce(b.title,'') title from r join read_parquet('{SRC}') b on b.cc='KR' and b.appl_no=r.appl_no
   where r.cc='KR'""").df()
log('KR 문서', len(kr))
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3), sublinear_tf=True, min_df=1)
vec.fit([dtext[k] for k in K50] + kr.title.sample(min(200000, len(kr)), random_state=0).tolist())
D = vec.transform([dtext[k] for k in K50]); kidx = {k: i for i, k in enumerate(K50)}
kr['ksic_txt'] = kr.code.map(single); kr['method'] = np.where(kr.ksic_txt.notna(), 'single', None)
for c, ks in multi.items():
    idx = np.where((kr.code == c).values)[0]
    if not len(idx): continue
    cols = [kidx[k] for k in ks]; out = np.empty(len(idx), dtype=object)
    for s in range(0, len(idx), 100000):
        ii = idx[s:s + 100000]; S = (vec.transform(kr.title.values[ii]) @ D[cols].T).toarray()
        out[s:s + len(ii)] = np.array(ks)[S.argmax(1)]   # 동점(전부 0 포함)이면 코드 안 첫 KSIC
    kr.loc[kr.index[idx], 'ksic_txt'] = out; kr.loc[kr.index[idx], 'method'] = 'text'
assert kr.ksic_txt.notna().all()
log('KR 배정 완료')
krt = kr[['appl_no', 'code', 'ksic_txt', 'method']].copy(); krt['cc'] = 'KR'; del kr
con.register('krt', krt)
# ---- 해외: 패밀리 상속 ----
con.execute(f"""create table fam as with f as (
   select f.cc, f.appl_no, k.ksic_txt, count(*) n from read_parquet('{FAM}') f join r on r.cc=f.cc and r.appl_no=f.appl_no
   join krt k on k.appl_no=f.kr_appl_no and k.code=r.code where r.cc<>'KR' and r.code in ({','.join("'" + c + "'" for c in multi)})
   group by 1,2,3)
 select cc, appl_no, arg_min(ksic_txt, -n*100000 + cast(ksic_txt as int)) ksic_txt from f group by 1,2""")
log('패밀리 상속', con.sql("select cc, count(*) from fam group by 1").fetchall())
# ---- TXT-IPC 표 (KR 텍스트 배정 결과만으로) ----
con.execute("""create table krk as select k.code, k.ksic_txt, r.sg, split_part(r.sg,'/',1) mg, left(r.sg,4) sc
   from krt k join r on r.cc='KR' and r.appl_no=k.appl_no where k."method"='text'""")
for lvl in ('sg', 'mg', 'sc'):
    con.execute(f"""create table tab_{lvl} as with g as (select code, {lvl} as k_, ksic_txt, count(*) n from krk where {lvl} is not null and {lvl}<>'' group by 1,2,3),
       tot as (select code, k_, sum(n) tot from g group by 1,2 having sum(n) >= 3)
       select g.code, g.k_ as key_, arg_min(g.ksic_txt, -g.n*100000 + cast(g.ksic_txt as int)) ksic_txt from g join tot using(code, k_) group by 1,2""")
con.execute("""create table tab_code as with g as (select code, ksic_txt, count(*) n from krk group by 1,2)
   select code, arg_min(ksic_txt, -n*100000 + cast(ksic_txt as int)) ksic_txt from g group by 1""")
con.execute(f"""create table fx as select r.cc, r.appl_no, r.code,
   coalesce(f.ksic_txt, a.ksic_txt, b.ksic_txt, c.ksic_txt, d.ksic_txt) ksic_txt,
   case when f.ksic_txt is not null then 'family' when a.ksic_txt is not null then 'ipc_sg' when b.ksic_txt is not null then 'ipc_mg'
        when c.ksic_txt is not null then 'ipc_sc' else 'code_mode' end as "method"
   from r left join fam f using(cc, appl_no)
   left join tab_sg a on a.code=r.code and a.key_=r.sg
   left join tab_mg b on b.code=r.code and b.key_=split_part(r.sg,'/',1)
   left join tab_sc c on c.code=r.code and c.key_=left(r.sg,4)
   left join tab_code d on d.code=r.code
   where r.cc<>'KR' and r.code in ({','.join("'" + c + "'" for c in multi)})""")
con.execute(f"""copy (
  select r.cc, r.appl_no, r.code, coalesce(k.ksic_txt, x.ksic_txt, km1.ksic) ksic_txt,
         coalesce(k."method", x."method", 'single') as "method", r.ksic_prov
  from r left join krt k on k.cc=r.cc and k.appl_no=r.appl_no
  left join fx x on x.cc=r.cc and x.appl_no=r.appl_no
  left join (select code, min(ksic) ksic from km group by 1 having count(*)=1) km1 on km1.code=r.code
) to '{OUT}' (format parquet)""")
S = con.sql(f"select * from read_parquet('{OUT}')")
assert con.sql(f"select count(*) from read_parquet('{OUT}') where ksic_txt is null").fetchone()[0] == 0
assert con.sql(f"select count(*) from read_parquet('{OUT}') o join km on km.ksic=o.ksic_txt where km.code<>o.code").fetchone()[0] == 0
# ---- 기술통계(사후, 제공 라벨과 대조) ----
st = {}
q = lambda s: con.sql(s.replace('OUT', f"read_parquet('{OUT}')")).df()
st['by_cc_method'] = q('select cc, "method", count(*) n, avg((ksic_txt=ksic_prov)::int) agree5 from OUT group by 1,2 order by 1,2').to_dict('records')
st['agree5_by_cc'] = q("select cc, count(*) n, avg((ksic_txt=ksic_prov)::int) agree5 from OUT group by rollup(cc) order by cc").to_dict('records')
st['agree5_multi34_by_cc'] = q(f"""select cc, count(*) n, avg((ksic_txt=ksic_prov)::int) agree5 from OUT where code in ({','.join("'" + c + "'" for c in multi)})
   group by rollup(cc) order by cc""").to_dict('records')
st['agree5_multi34_given_group_correct'] = q(f"""select cc, count(*) n, avg((ksic_txt=ksic_prov)::int) agree5 from OUT o join km on km.ksic=o.ksic_prov
   where o.code in ({','.join("'" + c + "'" for c in multi)}) and km.code=o.code group by rollup(cc) order by cc""").to_dict('records')
st['group_agree_by_cc'] = q("select cc, avg((km.code=o.code)::int) agree_code from OUT o join km on km.ksic=o.ksic_prov group by rollup(cc) order by cc").to_dict('records')
st['docs_by_ksic'] = q("""with a as (select ksic_txt k, count(*) n_txt, sum((cc='KR')::int) kr_txt from OUT group by 1),
    b as (select ksic_prov k, count(*) n_prov_in27, sum((cc='KR')::int) kr_prov from OUT group by 1)
    select coalesce(a.k,b.k) ksic, n_txt, kr_txt, n_prov_in27, kr_prov from a full join b on a.k=b.k order by 1""").to_dict('records')
st['n_ksic_with_txt_docs'] = int(con.sql(f"select count(distinct ksic_txt) from read_parquet('{OUT}')").fetchone()[0])
# R8 p1a TXT(KR)와의 일치(같은 방법·다른 연계표 구현·어휘 표본)
try:
    st['agree_with_R8_ktxt_KR'] = float(con.sql(f"""select avg((o.ksic_txt=a.k_txt)::int) from read_parquet('{OUT}') o join read_parquet('{R8TXT}') a
        on o.cc='KR' and a.appl_no=o.appl_no where a.k_txt is not null""").fetchone()[0])
except Exception as e:
    st['agree_with_R8_ktxt_KR'] = str(e)
json.dump(st, open(os.path.join(HERE, 'txt_assign_stats.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(pd.DataFrame(st['agree5_by_cc']).to_string()); print(pd.DataFrame(st['agree5_multi34_by_cc']).to_string())
print(pd.DataFrame(st['by_cc_method']).to_string()); print('n_ksic', st['n_ksic_with_txt_docs'], 'R8 KR 일치', st['agree_with_R8_ktxt_KR'])
log('end')
