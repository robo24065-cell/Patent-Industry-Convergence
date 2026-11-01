# -*- coding: utf-8 -*-
"""IPC 메인그룹/서브클래스·KSIC ↔ OpenAlex 분야(field/subfield) 실측 대응표.
RoS 인용쌍(신뢰도≥3)을 제공 데이터 셀에 연결하고, 인용된 논문(OpenAlex work) 중 무작위 표본의 primary_topic을 OpenAlex API로 조회.
OpenAlex는 일일 크레딧 한도(1,000요청/일, 다른 수집기와 공유)가 있어 표본 조회만 한다 — 남은 한도가 RESERVE 미만이면 즉시 멈춘다.
보내는 값: OpenAlex work ID(공개 식별자)와 mailto(사용자 승인, OpenAlex 전용)뿐. 제공 데이터 내용은 보내지 않는다.
사용: python r3ros_04_oafield.py [최대요청수=150] [RESERVE=350]
"""
import sys, json, time, requests, numpy as np, pandas as pd, duckdb
from pathlib import Path
EXT = Path(__file__).resolve().parent; E = EXT.as_posix()
MAXREQ = int(sys.argv[1]) if len(sys.argv) > 1 else 150
RESERVE = int(sys.argv[2]) if len(sys.argv) > 2 else 350
OUT = EXT / "r3ros_work" / "oa_sample.jsonl"; LOG = EXT / "r3ros_work" / "oa_fetch.log"
def log(m):
    with open(LOG, "a", encoding="utf-8") as f: f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + m + "\n")
con = duckdb.connect(); con.execute("set memory_limit='3500MB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
con.execute(f"set temp_directory='{E}/r3ros_work/tmp'")
con.execute(f"""create table cp as select distinct r.ksic, r.mg, f.us_pn from read_parquet('{E}/r3ros_work/fam_us.parquet') f
   join read_parquet('{E}/r3ros_work/rec.parquet') r using(cc, appl_no, ksic) where r.mg is not null""")
con.execute(f"""create table pr as select distinct c.pn us_pn, c.oaid from read_parquet('{E}/r3ros_work/cit.parquet') c
   where c.confscore>=3 and c.pn in (select distinct us_pn from cp)""")
works = con.sql("select distinct oaid from pr order by oaid").df().oaid.values
print("연결 US 특허가 인용한 서로 다른 논문", len(works), flush=True)
rng = np.random.default_rng(20260929)
samp = rng.permutation(works)
done = set()
if OUT.exists():
    for l in open(OUT, encoding="utf-8"):
        try: done.add(json.loads(l)["oaid"])
        except Exception: pass
queried = set()
if (EXT / "r3ros_work" / "oa_queried.txt").exists():
    queried = set(int(x) for x in open(EXT / "r3ros_work" / "oa_queried.txt").read().split())
todo = [w for w in samp if w not in queried]
nreq = 0; stop = "maxreq"
fo = open(OUT, "a", encoding="utf-8"); fq = open(EXT / "r3ros_work" / "oa_queried.txt", "a")
for i in range(0, len(todo), 100):
    if nreq >= MAXREQ: break
    b = todo[i:i + 100]
    url = "https://api.openalex.org/works"
    prm = {"filter": "openalex:" + "|".join(f"W{x}" for x in b), "select": "id,publication_year,primary_topic",
           "per-page": 100, "mailto": "robo24065@gmail.com"}
    for attempt in range(4):
        try:
            r = requests.get(url, params=prm, timeout=60); break
        except Exception as e:
            log(f"exception {type(e).__name__}"); time.sleep(5 * 2 ** attempt); r = None
    if r is None: stop = "exceptions"; break
    nreq += 1
    rem = r.headers.get("X-RateLimit-Remaining")
    if r.status_code != 200:
        log(f"HTTP {r.status_code} {r.text[:200]} remaining={rem}"); stop = f"http_{r.status_code}"; break
    for x in r.json().get("results", []):
        pt = x.get("primary_topic") or {}
        rec = dict(oaid=int(x["id"].rsplit("W", 1)[1]), year=x.get("publication_year"),
                   topic=(pt.get("display_name")), subfield_id=((pt.get("subfield") or {}).get("id", "") or "").rsplit("/", 1)[-1],
                   subfield=(pt.get("subfield") or {}).get("display_name"), field_id=((pt.get("field") or {}).get("id", "") or "").rsplit("/", 1)[-1],
                   field=(pt.get("field") or {}).get("display_name"), domain=(pt.get("domain") or {}).get("display_name"))
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fq.write(" ".join(str(int(x)) for x in b) + "\n"); fo.flush(); fq.flush()
    if rem is not None and int(rem) < RESERVE:
        stop = f"reserve_reached(remaining={rem})"; break
    time.sleep(0.6)
fo.close(); fq.close()
log(f"requests={nreq} stop={stop}"); print("요청", nreq, "중단사유", stop, flush=True)
# ---- 대응표 ----
oa = pd.DataFrame([json.loads(l) for l in open(OUT, encoding="utf-8")]).drop_duplicates("oaid")
qd = set(int(x) for x in open(EXT / "r3ros_work" / "oa_queried.txt").read().split())
print("조회 논문", len(qd), "응답(존재)", len(oa), "primary_topic 보유", int(oa.field.notna().sum()))
con.register("oa", oa[oa.field.notna()])
base = """select c.ksic, c.mg, split_part(c.mg,' ',1) sc, o.field_id, o.field, o.subfield_id, o.subfield, o.domain
          from pr join cp c using(us_pn) join oa o using(oaid)"""
mg = con.sql(f"select mg, sc, field_id, field, subfield_id, subfield, count(*) n_pairs from ({base}) group by all").df()
sc = con.sql(f"select sc, field_id, field, count(*) n_pairs from ({base}) group by all").df()
ks = con.sql(f"select ksic, field_id, field, count(*) n_pairs from ({base}) group by all").df()
for d, k in ((mg, "mg"), (sc, "sc"), (ks, "ksic")):
    d["share"] = d.n_pairs / d.groupby(k).n_pairs.transform("sum")
mg["level"] = "maingroup"
mg.to_parquet(EXT / "r3ros_ipc_mg_openalex_subfield.parquet", index=False)
sc.to_parquet(EXT / "r3ros_ipc_subclass_openalex_field.parquet", index=False)
ks.to_parquet(EXT / "r3ros_ksic_openalex_field.parquet", index=False)
tot = mg.groupby("mg").n_pairs.sum()
print("대응표: 메인그룹", mg.mg.nunique(), "(표본쌍≥30:", int((tot >= 30).sum()), ") / 서브클래스", sc.sc.nunique(), "/ KSIC", ks.ksic.nunique())
top = ks.sort_values("share", ascending=False).groupby("ksic").head(1).sort_values("ksic")
print(top[["ksic", "field", "share", "n_pairs"]].round(3).to_string(index=False))
