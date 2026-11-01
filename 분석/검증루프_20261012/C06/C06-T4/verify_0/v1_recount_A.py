# verify_0/v1: C06-A P1 독립 재계산. lib6·cache를 쓰지 않고 원천(CORP_APPLICANT.txt, kr_patents.parquet, 내려받은 벤처 CSV)에서 다시 센다.
# 다른 구현: duckdb SQL로 상호 유일성·첫 노출일·창 결과를 계산.
import sys, re, io, json
sys.stdout.reconfigure(encoding='utf-8')
import duckdb, pandas as pd
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
W = B + r'\분석\검증루프_20261012\C06\C06-T4'
pat = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
strip = re.compile(r"[\s\(\)\[\]·\.,\-&'\"/]")
def nz(s):
    return strip.sub('', pat.sub('', str(s or '')).upper())
rows = []
for line in open(B + r'\공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt', encoding='utf-8').read().splitlines()[1:]:
    p = line.split('¶')
    if len(p) >= 5:
        biz = re.sub(r'\D', '', p[4])
        rows.append((nz(p[1].strip()), biz))
ca = pd.DataFrame(rows, columns=['n', 'biz'])
k = pd.read_parquet(B + r'\공모전데이터\작업데이터\kr_patents.parquet', columns=['출원번호', '공개일자', '등록일자', '출원인명'])
k.columns = ['a', 'p', 'r', 'nm']
k = k.drop_duplicates('a')
k['n'] = k.nm.map(nz)
for c in ('p', 'r'):
    k[c] = k[c].fillna('').astype(str).str.strip(); k.loc[k[c].str.len() != 8, c] = None
raw = open(W + r'\download\venture_20260521\venture.bin', 'rb').read().decode('utf-8-sig')
v = pd.read_csv(io.StringIO(raw), dtype=str)
v = pd.DataFrame({'n': v['업체명'].map(nz), 's': v['벤처유효시작일'].str.replace('-', ''), 'e': v['벤처유효종료일'].str.replace('-', ''), 'nr': v['신규_재확인']})
con = duckdb.connect(); con.sql('SET threads=2')
con.register('ca', ca); con.register('k', k); con.register('v', v)
con.sql("""CREATE TABLE ok AS SELECT n FROM ca WHERE length(n)>=2 GROUP BY n
           HAVING count(*)=sum(CASE WHEN length(biz)=10 THEN 1 ELSE 0 END)
              AND count(DISTINCT CASE WHEN length(biz)=10 THEN substr(biz,1,6) END)=1
              AND max(CASE WHEN length(biz)=10 AND substr(biz,4,2) BETWEEN '81' AND '88' THEN 1 ELSE 0 END)=1""")
con.sql("""CREATE TABLE fv AS SELECT k.n, min(least(coalesce(p,'99999999'), coalesce(r,'99999999'))) fv FROM k JOIN ok USING(n)
           WHERE p IS NOT NULL OR r IS NOT NULL GROUP BY k.n""")
O = [('20230430', '20231031'), ('20231031', '20240430'), ('20240430', '20241031'), ('20241031', '20250430'), ('20250430', '20251031'), ('20251031', None)]
res = []
for a, b in O:
    q = f"""SELECT count(*) n, sum(CASE WHEN EXISTS (SELECT 1 FROM v WHERE v.n=fv.n AND v.nr='신규' AND v.s>'{a}' AND v.s<='{b or '00000000'}') THEN 1 ELSE 0 END) pos
            FROM fv WHERE fv.fv<='{a}' AND NOT EXISTS (SELECT 1 FROM v WHERE v.n=fv.n AND v.s<='{a}' AND v.e>='{a}')"""
    n, pos = con.sql(q).fetchone()
    res.append(dict(t0=a, n=int(n), pos=int(pos) if b else None))
    print(res[-1], flush=True)
uniq = con.sql("""SELECT count(DISTINCT n) FROM v WHERE nr='신규' AND s>'20230430' AND s<='20251031' AND n IN (SELECT n FROM fv)""").fetchone()[0]
out = dict(by_origin=res, note='uniq_any_newcert_names_dev(위험집합 조건 없이 상한)=' + str(uniq))
json.dump(out, open(W + r'\verify_0\v1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(out)
