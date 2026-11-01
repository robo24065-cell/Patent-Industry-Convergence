# -*- coding: utf-8 -*-
"""P1 비순환 점검: '기술 g가 산업 k로 이동했다'는 사건이 실제 산업 k 기업의 출원으로 이루어지는가?
사건을 만드는 미래 창(t+1..t+3) KR 문서의 출원 법인 등록업종(DART induty_code, 2026-08 스냅샷 — 시점 한계)이 도착 산업과 맞는 비율을
제공 라벨 사건 / 연계표(CW) 사건 / 해설서(TXT) 사건끼리 비교한다(같은 27그룹 해상도). 평시(비사건) 문서의 같은 비율도 함께.
출원인 연결: N4 cache/app_firm.parquet src=registry(등록특허 정확 코드, 읽기만) → CORP_APPLICANT 법인번호 → DART. 개발 원점(t≤2018, 문서 ≤2021)만.
"""
import sys, sqlite3
sys.dont_write_bytecode = True
from r8_common import *
import duckdb

P = pd.read_parquet(p("cache", "p1_panel.parquet"), columns=["t", "mg", "k", "y", "y_cw", "y_txt"])
P = P[P.t <= 2018]
con = duckdb.connect(); con.sql("SET threads=2; SET memory_limit='3GB'")
R, C, A = [p('cache', x).replace("\\", "/") for x in ('rec.parquet', 'ipc_cw.parquet', 'kr_assign.parquet')]
docs = con.sql(f"""SELECT r.appl_no, r.mg, r.pyear, r.ksic prov, a.cw, a.k_txt FROM '{R}' r
   JOIN '{A}' a ON r.cc='KR' AND a.appl_no=r.appl_no WHERE r.cc='KR' AND r.mg<>'' AND r.pyear BETWEEN 2001 AND 2021""").df()
assert docs.pyear.max() <= 2021
af = pd.read_parquet(os.path.join(MC, r"N4_특허행태_기업폐업조기경보\cache\app_firm.parquet"), columns=["applicantCd", "appl_no", "src"])
af = af[(af.src == "registry") & af.applicantCd.astype(str).str.startswith("1")].drop_duplicates("appl_no")
rows = []
with open(os.path.join(BASE, r"공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt"), encoding="utf-8") as f:
    next(f)
    for line in f:
        q = line.rstrip("\r\n").split("¶")
        if len(q) >= 5:
            rows.append((q[0].strip(), q[3].replace("-", "").strip()))
CA = pd.DataFrame(rows, columns=["applicantCd", "jurir"])
c = sqlite3.connect("file:" + os.path.join(BASE, r"공모전데이터\외부데이터\DART\dart_company.sqlite") + "?mode=ro", uri=True)
D = pd.read_sql("select jurir_no, induty_code from company where induty_code<>''", c); c.close()
D["jurir"] = D.jurir_no.str.replace("-", "").str.strip()
D = D[D.jurir.str.len() == 13].drop_duplicates("jurir")
firm = CA[CA.jurir.str.len() == 13].merge(D[["jurir", "induty_code"]], on="jurir").drop_duplicates("applicantCd")
firm["rg"] = firm.induty_code.map(ksic_to_group)
docs = docs.merge(af[["appl_no", "applicantCd"]], on="appl_no", how="left").merge(firm[["applicantCd", "rg", "induty_code"]], on="applicantCd", how="left")
log("docs", len(docs), "firm-linked", round(float(docs.applicantCd.notna().mean()), 4), "DART industry in space", round(float(docs.rg.notna().mean()), 4))
docs["pg"] = docs.prov.map(K2G)
docs["tg"] = docs.k_txt.map(K2G)


def event_docs(ev, labcol, labkey):
    """ev: (t, mg, lab) 사건 표 → 사건 문서(미래 창, 해당 라벨)와 도착 그룹."""
    out = []
    for t, e in ev.groupby("t"):
        dd = docs[(docs.pyear >= t + 1) & (docs.pyear <= t + 3)]
        m = dd.merge(e[["mg", "lab", "eid"]], left_on=["mg", labcol], right_on=["mg", "lab"])
        past = docs[(docs.pyear >= t - 5) & (docs.pyear <= t)][["mg", "applicantCd"]].dropna().drop_duplicates()
        past["inc"] = 1
        m = m.merge(past, on=["mg", "applicantCd"], how="left")
        m["new_entrant"] = m.inc.isna() & m.applicantCd.notna()
        out.append(m)
    m = pd.concat(out)
    m["dest_g"] = m.lab.map(labkey)
    return m


res = {}
EV = {
    "PROV": (P[P.y == 1].assign(lab=lambda x: x.k)[["t", "mg", "lab"]], "prov", K2G),
    "CW": (P[P.y_cw == 1].assign(lab=lambda x: x.k.map(K2G))[["t", "mg", "lab"]].drop_duplicates(), "cw", {g: g for g in SPACE}),
    "TXT": (P[P.y_txt == 1].assign(lab=lambda x: x.k)[["t", "mg", "lab"]], "k_txt", K2G),
}
rng = np.random.default_rng(0)
store = {}
for nm, (ev, col, key) in EV.items():
    ev = ev.reset_index(drop=True); ev["eid"] = np.arange(len(ev))
    m = event_docs(ev, col, key)
    x = m[m.rg.notna()].copy()
    x["hit"] = (x.rg == x.dest_g).astype(float)
    per = x.groupby("eid").hit.mean()
    bs = [per.sample(len(per), replace=True, random_state=int(s)).mean() for s in rng.integers(0, 1e9, 1000)]
    store[nm] = per
    res[nm] = {"events": int(len(ev)), "event_docs": int(len(m)), "docs_with_DART_industry_in_space": int(len(x)),
               "events_evaluable": int(len(per)),
               "dest_match_event_mean": float(per.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
               "dest_match_doc_pooled": float(x.hit.mean()),
               "new_entrant_share_docs": float(m.new_entrant[m.applicantCd.notna()].mean())}
# 평시 기준: 같은 기간(2007–2021) 전체 문서에서 각 라벨 그룹 = 기업 그룹 비율
base = docs[docs.rg.notna() & (docs.pyear >= 2007)]
res["baseline_all_docs"] = {"PROV": float((base.pg == base.rg).mean()), "CW": float((base.cw == base.rg).mean()),
                            "TXT": float((base.tg == base.rg).mean())}
for a, b in (("PROV", "CW"), ("PROV", "TXT")):
    d = [store[a].sample(len(store[a]), replace=True, random_state=s).mean() - store[b].sample(len(store[b]), replace=True, random_state=s + 7).mean() for s in range(1000)]
    res[f"{a}-{b}"] = {"diff": float(store[a].mean() - store[b].mean()), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
json.dump(res, open(p("p1_firm_validity.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
log("end")
