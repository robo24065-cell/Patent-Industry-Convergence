# -*- coding: utf-8 -*-
r"""C11-T2 s7 (P2): 지형도(출원 2017–19 발명, 5개국 동일가중 RTA) F·H 시나리오를 연결 팔별로 독립 재계산.
C10-T3 s3_rho.py·verify_5 v5_landscape.py·alt_fix를 쓰지 않는다. duckdb는 파일 읽기·추출만, 배정 논리(최빈·표·상속)는 pandas.
 FREE-IPC 해외: 연결 KR 문서 k_free(T1 kr_txtfree) 최빈(동률 코드 작은 것) → 없으면 KR(출원 ≤2019) k_free의 sg→mg→sc(각 ≥3건) 최빈표.
 TXT 해외: T3 풀 안 문서 = C11 txt_assign_{arm}, 풀 밖 = 저장 alt_assign k_txt(패밀리 경로 없음).
 KR(F 시나리오): 저장 alt k_free·k_txt(링크 무관). H: KR 제공 라벨.
읽기만: c5_R10/alt_assign, ESSENTIAL_R7/rec_conc·conc_ksic_map, T1 kr_txtfree, _ideas_R8 fam_link, C10-T3 links.parquet, C11 cache/txt_assign_*.
쓰기: res/s7_landscape.json, res/s7_landscape.csv"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, duckdb
from scipy.stats import spearmanr
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"; T3 = MC + "/T3_기술셀부상예측"
W = BASE + "/분석/검증루프_20261012/C11/C11-T2"; C10 = BASE + "/분석/검증루프_20261012/C10/C10-T3"
ALT = MC + "/_shared_report/fusion_advantage/c5_R10/alt_assign/*.parquet"
RC = T3 + "/ESSENTIAL_R7/rec_conc.parquet"; KF = MC + "/T1_산업대기실/TXT_R9/cache/kr_txtfree.parquet"
FAM = MC + "/_ideas_R8/cache/fam_link.parquet"; LINKS = C10 + "/cache/links.parquet"
CCS = ["KR", "US", "JP", "CN", "EP"]
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
con = duckdb.connect(); con.execute(f"set memory_limit='3GB'; set threads=2; set enable_progress_bar=false; set temp_directory='{W}/cache/_tmp'")
km = pd.read_csv(T3 + "/ESSENTIAL_R7/conc_ksic_map.csv", dtype=str)[["ksic", "code"]]
K50 = sorted(km.ksic)
# ---- 문서(창: 발명, 출원 2017–19) ----
D = con.sql(f"""select a.cc, a.appl_no, a.ksic_prov, a.code, a.k_txt, a.k_free, a.m_free, r.sg from read_parquet('{ALT}') a
    left join read_parquet('{RC}') r using(cc, appl_no) where a.is_inv and a.appl_year between 2017 and 2019""").df()
log("창 문서", D.groupby("cc").size().to_dict())
kfree = pd.read_parquet(KF).dropna(subset=["k_free"]).set_index("appl_no").k_free
# ---- FREE 표: KR 출원 ≤2019 (alt_KR 행 = meta∩rec_conc) ----
KRt = con.sql(f"""select a.appl_no, r.sg from read_parquet('{ALT}') a left join read_parquet('{RC}') r using(cc, appl_no)
    where a.cc='KR' and a.appl_year<=2019""").df()
KRt["k"] = KRt.appl_no.map(kfree)
KRt = KRt.dropna(subset=["k"])

def mode_table(df, key):
    g = df[df[key].notna() & (df[key] != "")].groupby([key, "k"]).size().rename("n").reset_index()
    tot = g.groupby(key).n.transform("sum"); g = g[tot >= 3]
    g = g.assign(kk=g.k.astype(int)).sort_values([key, "n", "kk"], ascending=[True, False, True])
    return g.drop_duplicates(key).set_index(key).k

KRt["mg"] = KRt.sg.str.split("/").str[0]; KRt["sc"] = KRt.sg.str[:4]
TAB = {lvl: mode_table(KRt, lvl) for lvl in ("sg", "mg", "sc")}
log("FREE 표", {k: len(v) for k, v in TAB.items()})
F = D[D.cc != "KR"].copy(); F["mg"] = F.sg.str.split("/").str[0]; F["sc"] = F.sg.str[:4]
tabval = F.sg.map(TAB["sg"])
tabval = tabval.fillna(F.mg.map(TAB["mg"])).fillna(F.sc.map(TAB["sc"]))
F["free_tab"] = tabval
# 재구축 점검: 저장 m_free가 ipc_* 인 문서는 표 값이 저장값과 같아야
chk = F[F.m_free.str.startswith("ipc_")]; R = {"free_table_rebuild_eq_stored": float((chk.free_tab == chk.k_free).mean()), "n_chk": int(len(chk)),
     "free_none_rebuild": float(F.loc[F.m_free == "none", "free_tab"].isna().mean())}
log("표 재구축", R)
key = F.cc + "|" + F.appl_no

def links(arm):
    if arm == "r8": q = f"select cc, appl_no, kr_appl_no from read_parquet('{FAM}')"
    else: q = f"select cc, appl_no, kr_appl_no from read_parquet('{LINKS}') where link='{arm.upper()}'"
    L = con.sql(q).df(); L["k"] = L.kr_appl_no.map(kfree); L = L.dropna(subset=["k"])
    g = L.groupby(["cc", "appl_no", "k"]).size().rename("n").reset_index()
    g = g.assign(kk=g.k.astype(int)).sort_values(["cc", "appl_no", "n", "kk"], ascending=[True, True, False, True]).drop_duplicates(["cc", "appl_no"])
    return pd.Series(g.k.values, index=(g.cc + "|" + g.appl_no).values)

FREE = {}
for arm in ("r8", "reg", "pub", "pure"):
    fam = pd.Series(dtype=object) if arm == "pure" else links(arm)
    v = key.map(fam); FREE[arm] = v.fillna(F.free_tab).values
R["free_r8_eq_stored"] = float((pd.Series(FREE["r8"]).fillna("NA").values == F.k_free.fillna("NA").values).mean())
log("FREE r8 = 저장", R["free_r8_eq_stored"])
# ---- TXT 해외 ----
TXT = {}
for arm in ("r8", "reg", "pub", "pure"):
    ta = con.sql(f"""select t.cc||'|'||t.appl_no as kkey, t.ksic_txt from read_parquet('{W}/cache/txt_assign_{arm}.parquet') t
        join (select cc, appl_no from read_parquet('{ALT}') where cc<>'KR' and is_inv and appl_year between 2017 and 2019) a using(cc, appl_no)""").df().set_index("kkey").ksic_txt
    TXT[arm] = key.map(ta).fillna(F.k_txt).values   # 풀 밖 = 저장 k_txt
    R[f"txt_pool_share_{arm}"] = float(key.isin(ta.index).mean())
R["txt_r8_eq_stored"] = float((pd.Series(TXT["r8"]).fillna("NA").values == F.k_txt.fillna("NA").values).mean())
log("TXT r8 = 저장", R["txt_r8_eq_stored"])
# ---- RTA·지형도 ----
KR = D[D.cc == "KR"]

def table(kr_lab, for_lab):
    n = pd.DataFrame(0.0, index=K50, columns=CCS)
    n["KR"] = kr_lab.value_counts().reindex(K50).fillna(0)
    for cc in CCS[1:]:
        m = (F.cc == cc).values
        n[cc] = pd.Series(for_lab[m]).dropna().value_counts().reindex(K50).fillna(0).values
    return n

def rta(n):
    s = n / n.sum(axis=0)
    return s.div(s.mean(axis=1), axis=0)

prov = rta(table(KR.ksic_prov, F.ksic_prov.values))
lead_p = prov.idxmax(axis=1); krl_p = set(lead_p.index[lead_p == "KR"])
R["prov"] = dict(kr_rta_gt1=int((prov.KR > 1).sum()), kr_leader=len(krl_p))
rows = []
def summ(name, rt):
    ld = rt.idxmax(axis=1); krl = set(ld.index[ld == "KR"]); rho = spearmanr(rt.KR, prov.KR).statistic
    rows.append(dict(name=name, kr_rta_gt1=int((rt.KR > 1).sum()), kr_leader=len(krl), kept=len(krl & krl_p), rho_KR=round(float(rho), 3), rho_KR_raw=float(rho),
                     ties_leader=int((rt.eq(rt.max(axis=1), axis=0).sum(axis=1) > 1).sum())))
# cweq F (링크 무관): 코드 동일가중
w = km.assign(w=1 / km.groupby("code").ksic.transform("size"))
def cweq_counts(df):
    m = df[["code"]].merge(w, on="code"); return m.groupby("ksic").w.sum().reindex(K50).fillna(0)
ncw = pd.DataFrame({cc: cweq_counts(D[D.cc == cc]) for cc in CCS})
summ("cweq_F", rta(ncw))
for arm in ("r8", "reg", "pub", "pure"):
    summ(f"free_F_{arm}", rta(table(KR.k_free.dropna(), FREE[arm])))
    summ(f"free_H_{arm}", rta(table(KR.ksic_prov, FREE[arm])))
    summ(f"txt_F_{arm}", rta(table(KR.k_txt.dropna(), TXT[arm])))
    summ(f"txt_H_{arm}", rta(table(KR.ksic_prov, TXT[arm])))
T = pd.DataFrame(rows); T.to_csv(W + "/res/s7_landscape.csv", index=False, encoding="utf-8-sig"); print(T.to_string())
# 168행 범위(F 시나리오: cweq, txt, free) — raw에서 소수 둘째 자리 1회 반올림
rng_ = {}
for arm in ("r8", "reg", "pub", "pure"):
    v = T.set_index("name").rho_KR_raw.loc[["cweq_F", f"txt_F_{arm}", f"free_F_{arm}"]]
    lead = T.set_index("name").kr_leader.loc[["cweq_F", f"txt_F_{arm}", f"free_F_{arm}"]]
    rng_[arm] = dict(rho_min_raw=float(v.min()), rho_max_raw=float(v.max()), text=f"{v.min():.2f}–{v.max():.2f}",
                     text_double_round=f"{round(round(v.min(), 3), 2):.2f}–{round(round(v.max(), 3), 2):.2f}", leader_range=f"{lead.min()}–{lead.max()}")
R["range_168"] = rng_; R["rows"] = rows
json.dump(R, open(W + "/res/s7_landscape.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(rng_, ensure_ascii=False, indent=1)); log("end")
