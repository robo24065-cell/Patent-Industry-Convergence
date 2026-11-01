# verify_0 독립 재계산: 작업자 캐시·lib7을 쓰지 않고 registry_full(ro)·kr_patents 원천에서 SQL(duckdb)로 P1 수를 다시 센다.
# 정의는 PREREG §2 + 변경 1. (1차 실행: 날짜 공백 문자열을 결측으로 안 봐 '' 날짜가 t0 이전으로 잡힌 결함 → trim·길이 8 조건으로 수정 후 재실행) 정규화 규칙만 문서 정의(C04 libA)대로 다시 적었다.
import sqlite3, sys, re, json, time
import pandas as pd, numpy as np, duckdb
sys.stdout.reconfigure(encoding='utf-8')
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
OUT = B + r'\분석\검증루프_20261012\C07\C07-T4\verify_0\v1_out.json'
con = sqlite3.connect('file:' + B + r'\수집파이프라인\registry_full.sqlite?mode=ro', uri=True)
t = time.time()
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def nz(s): return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", SUF.sub("", str(s or "")).upper())
own = pd.read_sql("select rgst_no, seq, ownerName, ownerNatl, ownerRgstCsDate d, ownerRgstCsName a, ownerRgstCsReason b from owner where ownerRgstCsDate < '20250101'", con)
own['n'] = own.ownerName.map(nz)
pat = pd.read_sql("select rgst_no, applNo, ksic, rgstDate, cndrtExptnDate, opnDate, pubDate from patent", con)
rgt = pd.read_sql("select rgst_no, seq, rgstCsDate d, rgstCsName nm from right where rgstCsDate < '20250101'", con)
app = pd.read_sql("select rgst_no, seq, applicantCd, applicantName from applicant", con)
app['n'] = app.applicantName.map(nz)
k = pd.read_parquet(B + r'\공모전데이터\작업데이터\kr_patents.parquet', columns=['출원번호', '공개일자', '등록일자', '출원인명', '세세분류코드'])
k.columns = ['appl_no', 'pub', 'rg', 'aname', 'ksic']; k['n'] = k.aname.map(nz); k['ksic'] = k.ksic.astype(str)
db = duckdb.connect(); db.execute("set threads=2; set memory_limit='3GB'")
for nm, df in [('own', own), ('pat', pat), ('rgt', rgt), ('app', app), ('k', k)]:
    db.register(nm, df)
# 유형: 이름별 출원인코드 첫 자리 최빈
pref = db.sql("select n, mode(substr(applicantCd,1,1)) p from app where applicantCd is not null group by n").df().set_index('n').p
GRP = ['삼성', '현대', '에스케이', 'SK', '엘지', 'LG', '롯데', '포스코', '지에스', 'GS', '한화', '케이티', 'KT', '한진', '두산', '씨제이', 'CJ',
       '엘에스', 'LS', '금호', '대림', '부영', '동부', '신세계', '효성', '대우', '영풍', '에쓰오일', '오씨아이', 'OCI', '코오롱', '케이씨씨', 'KCC',
       '한국타이어', '대성', '동국', '세아', '태광', '미래에셋', '한라', '이랜드', '한솔', '동양']
def grp(r):
    s = str(r or '').replace('주식회사', '').replace('(주)', '').replace('㈜', '').strip()
    return next((g for g in GRP if s.startswith(g)), None)
def indiv(n, raw, natl):
    p = pref.get(n)
    if isinstance(p, str) and p.isdigit():
        return p in ('4', '6')
    s = str(raw or '')
    if isinstance(natl, str) and natl not in ('대한민국', ''): return False
    if '대학' in s or '산학협력' in s or any(x in s for x in ['연구원', '연구소', '과학기술원', '연구재단', '대한민국', '광역시', '특별시', '도청', '시청', '군청', '구청', '공사', '공단', '진흥원', '재단',
                                                         '주식회사', '(주)', '㈜', '회사', '유한', '코리아', '테크', '산업', '전자', '기업', '협동조합', '법인']):
        return False
    return bool(re.fullmatch(r'[가-힣]{2,4}', s.replace(' ', '')))
# 날짜 묶음 대표(seq 최소)와 직전 묶음
gr = db.sql("""
 with g as (select rgst_no, d, arg_min(n, seq) n, arg_min(ownerName, seq) raw, arg_min(ownerNatl, seq) natl,
                   max(case when a='양도' and b='권리의 전부이전등록' then 1 else 0 end) acc from own group by rgst_no, d)
 select *, row_number() over (partition by rgst_no order by d) - 1 kk,
        lag(n) over (partition by rgst_no order by d) sn, lag(raw) over (partition by rgst_no order by d) sraw from g""").df()
ev = gr[(gr.kk > 0) & (gr.acc == 1)].copy()
def same(a, b):
    if not a or not b or not isinstance(a, str) or not isinstance(b, str): return False
    return a == b or (len(a) >= 3 and len(b) >= 3 and (a in b or b in a))
ev = ev[[not same(a, b) for a, b in zip(ev.n, ev.sn)]]
g1 = ev.raw.map(grp); g2 = ev.sraw.map(grp); ev = ev[~(g1.notna() & (g1 == g2))]
ev['ind'] = [indiv(a, b, c) for a, b, c in zip(ev.n, ev.raw, ev.natl)]
db.register('ev', ev)
# 포트폴리오 (이름, KSIC) 최초일: A 제공 문서, B 같은 registry 이름의 비개인 코드 문서, C owner 행
db.execute("""create temp table vis as select p.rgst_no, p.applNo appl_no, p.ksic,
   coalesce(least(nullif(trim(k.pub),''), nullif(trim(k.rg),'')), least(nullif(trim(p.opnDate),''), nullif(trim(p.pubDate),''), nullif(trim(p.rgstDate),''))) v
   from pat p left join k on k.appl_no = p.applNo""")
db.execute("""create temp table port as
  select n, ksic, min(dt) dt from (
    select n, ksic, least(nullif(trim(pub),''), nullif(trim(rg),'')) dt from k
    union all
    select c.n, v.ksic, v.v dt from (select distinct n, applicantCd from app where applicantCd is not null and substr(applicantCd,1,1) not in ('4','6')) c
       join app a2 on a2.applicantCd = c.applicantCd join vis v on v.rgst_no = a2.rgst_no
    union all
    select o.n, v.ksic, o.d dt from own o join vis v on v.rgst_no = o.rgst_no) where n <> '' and dt is not null and length(dt) = 8 group by n, ksic""")
db.execute("create temp table pn as select n, min(dt) dt from port group by n")
res = {}
for Y in range(2015, 2023):
    L = f'{Y}0101'; E = f'{Y + 1}0101'
    alive = db.sql(f"""
      with st as (select rgst_no, arg_max(case when nm in ('소멸등록','권리의 말소등록','말소등록','직권말소등록') then 1 else 0 end, d || lpad(cast(seq as varchar),6,'0')) dead
                  from rgt where nm in ('설정등록','국유특허(실용신안)설정등록','국유설정등록','소멸등록','회복등록','권리의 말소등록','말소등록','직권말소등록') and d < '{L}' group by rgst_no)
      select p.rgst_no, p.ksic from pat p left join st using(rgst_no)
      where p.rgstDate < '{L}' and (p.cndrtExptnDate is null or p.cndrtExptnDate = '' or p.cndrtExptnDate > '{L}') and coalesce(st.dead,0) = 0
        and p.rgst_no in (select distinct rgst_no from own)""").df()
    db.register('al', alive)
    fe = db.sql(f"""select e.rgst_no, arg_min(e.n, e.d) n, arg_min(e.ind, e.d) ind from ev e join al using(rgst_no)
                    where e.d >= '{L}' and e.d < '{E}' group by e.rgst_no""").df()
    db.register('fe', fe)
    yl = db.sql(f"""select fe.rgst_no, fe.n from fe join al using(rgst_no) join pn on pn.n = fe.n and pn.dt < '{L}'
                   left join port pt on pt.n = fe.n and pt.ksic = al.ksic and pt.dt < '{L}'
                   where not fe.ind and pt.n is null""").df()
    yl.assign(t0=Y).to_parquet(OUT.replace('v1_out.json', f'v1_pos_{Y}.parquet'), index=False); y = len(yl)
    res[Y] = {'risk': len(alive), 'first_acc_events': len(fe), 'pos': int(y)}
    print(Y, res[Y], round(time.time() - t), flush=True)
lk = db.sql("""with st as (select rgst_no, arg_max(case when nm in ('소멸등록','권리의 말소등록','말소등록','직권말소등록') then 1 else 0 end, d || lpad(cast(seq as varchar),6,'0')) dead
       from rgt where nm in ('설정등록','국유특허(실용신안)설정등록','국유설정등록','소멸등록','회복등록','권리의 말소등록','말소등록','직권말소등록') and d < '20250101' group by rgst_no)
   select count(*) from pat p left join st using(rgst_no) where p.rgstDate < '20250101' and (p.cndrtExptnDate is null or p.cndrtExptnDate='' or p.cndrtExptnDate > '20250101')
   and coalesce(st.dead,0)=0 and p.rgst_no in (select distinct rgst_no from own where d < '20250101')""").fetchone()[0]
r21 = res[2021]['pos'] / res[2021]['risk']; r22 = res[2022]['pos'] / res[2022]['risk']
out = {'per_origin': res, 'lock_risk': int(lk), 'lock_expected': lk * (r21 + r22) / 2, 'sec': round(time.time() - t)}
json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(out)
