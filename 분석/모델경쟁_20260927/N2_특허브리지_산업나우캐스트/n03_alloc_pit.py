# -*- coding: utf-8 -*-
"""N2 누수 보완: 시점판(point-in-time) 특허 KSIC 배분가중. T년 말(=원점 t의 t−1년 말) 기준.
분자: 제공범위 등록특허 중 T년 말에 기업 i가 '보유'하고 '생존'한 KSIC k 건수.
  - 보유: 등록원부 owner 이력. T 이전 권리이전이 없으면 출원인(코드 기반 연결). 있으면 T 이전 마지막 이전 등록의 권리자(이름→기업 매핑),
          '일부이전'이면 직전 권리자도 유지. (현재 권리자 스냅샷 미사용)
  - 생존: 설정등록 ≤ T, T까지의 마지막 소멸/말소 사건 뒤 회복등록이 없으면 소멸, 존속기간만료일 > T.
분모: max(분자 합, 분자 합 + 제공범위 밖 추정 생존 보유특허).
  - 제공범위 밖: business_rights.right_list(권리자란) 특허(10) 중 제공범위에 없는 것. 스냅샷 편향을 줄이려고
    '출원인=현 권리자'(자사 출원 유지분)만, 설정등록 ≤ T, 존속기간만료 > T 인 건수 × 기업의 제공범위 내 T 시점 생존율(소멸 이력 기반).
출력: cache/alloc_pit.parquet (corp_code, t0=T, ksic, n_k, n_in, n_out_est, w_pit, w_pit_share)
      cache/alloc_snap.parquet (파일럿 스냅샷판: 출원인 기준 생존 + right_list 누적 분모, 비교용)"""
import os, sqlite3, re, json, time
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "cache")
REG = os.path.join(BASE, r"수집파이프라인\registry_full.sqlite")
BR = os.path.join(BASE, r"수집파이프라인\business_rights.sqlite")
import sys
TMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 2023      # 미래 모드(final_run_r2.py --mode future)만 2024
SUF = "" if TMAX == 2023 else f"_{TMAX}"
TS = list(range(2012, TMAX + 1))   # T = t−1 → t = 2013..2024 (T=2023 은 최종시험용 가중; 특허 가중은 결과변수 아님)
t_start = time.time()

def yr(s):
    return pd.to_numeric(s.astype(str).str[:4], errors="coerce")

def norm(s):
    s = str(s)
    s = re.sub(r"주식회사|\(주\)|㈜|유한회사|\(유\)|유한책임회사|co\.?,?\s*ltd\.?|inc\.?|corporation|corp\.?", "", s, flags=re.I)
    return re.sub(r"[\s\.,\-\(\)·&]", "", s).lower()

pf = pd.read_parquet(os.path.join(C, "pat_firm.parquet"))
dart = pd.read_parquet(os.path.join(C, "dart_firms.parquet"))
pat = pf.drop_duplicates("rgst_no")[["rgst_no", "ksic", "rgstDate", "cndrtExptnDate"]].copy()
pat["ry"] = yr(pat.rgstDate); pat["exy"] = yr(pat.cndrtExptnDate)
app_link = pf[pf.corp_code.notna()][["rgst_no", "corp_code"]].drop_duplicates()

# 이름 → corp_code (유일 매핑만)
nm = pd.concat([pf[pf.corp_code.notna()][["applicantName", "corp_code"]].rename(columns={"applicantName": "name"}),
                dart[["corp_name", "corp_code"]].rename(columns={"corp_name": "name"})])
nm["k"] = nm.name.map(norm)
nm = nm[nm.k.str.len() >= 2].drop_duplicates(["k", "corp_code"])
uniq = nm.groupby("k").corp_code.nunique()
name2corp = nm[nm.k.isin(uniq[uniq == 1].index)].drop_duplicates("k").set_index("k").corp_code

con = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
own = pd.read_sql("select rgst_no, seq, ownerName, ownerRgstCsDate d, ownerRgstCsName n, ownerRgstCsReason r from owner", con)
rt = pd.read_sql("select rgst_no, rgstCsDate d, rgstCsName n from right where rgstCsName in ('소멸등록','권리의 말소등록','말소등록','회복등록')", con)
con.close()
own["y"] = pd.to_numeric(own.d.astype(str).str[:8], errors="coerce")
own["corp"] = own.ownerName.map(norm).map(name2corp)
tr = own[own.seq > 0].copy()                        # 이전 등록(seq0 = 설정등록 시 권리자)
tr["partial"] = tr.r.fillna("").str.contains("일부")
rt["y"] = pd.to_numeric(rt.d.astype(str).str[:8], errors="coerce")
rt["kill"] = (rt.n != "회복등록").astype(int)
print("name map size", len(name2corp), "transfer rows", len(tr), "mapped transfer rows", tr.corp.notna().mean().round(3), flush=True)

# 제공범위 밖 (right_list, 권리자란, 자사 출원 유지분)
bcon = sqlite3.connect(f"file:{BR}?mode=ro", uri=True)
rl = pd.read_sql("select replace(corp_rgstno,'-','') jurir, rgst_no, rgst_date, expire_date, "
                 "json_extract(raw,'$.applicantInfo') ai, json_extract(raw,'$.rightHolderInfo') rh from right_list "
                 "where substr(rgst_no,1,2)='10' and corp_rgstno is not null", bcon)
bcon.close()
rl = rl.merge(dart[["jurir_no", "corp_code"]], left_on="jurir", right_on="jurir_no")
inscope = set(pat.rgst_no)
rl["in"] = rl.rgst_no.isin(inscope)
rl["own"] = rl.ai.fillna("") == rl.rh.fillna("")
rl["ry"] = yr(rl.rgst_date); rl["exy"] = yr(rl.expire_date)
out = rl[(~rl["in"]) & rl["own"]].drop_duplicates(["corp_code", "rgst_no"])
print("right_list firms", rl.corp_code.nunique(), "out-of-scope own rows", len(out), "share own among out", round(rl[~rl["in"]]["own"].mean(), 3), flush=True)

rows, snap_rows = [], []
for T in TS:
    cut = (T + 1) * 10000   # YYYYMMDD < (T+1)0101
    p = pat[(pat.ry <= T)].copy()
    # 생존
    e = rt[rt.y < cut].sort_values("y").drop_duplicates("rgst_no", keep="last")
    dead = set(e[e.kill == 1].rgst_no)
    p["alive"] = (~p.rgst_no.isin(dead)) & (p.exy.isna() | (p.exy > T))
    # 보유
    t_ = tr[tr.y < cut]
    last = t_.groupby("rgst_no").y.max().rename("ly")
    t_ = t_.merge(last, on="rgst_no")
    cur = t_[t_.y == t_.ly]
    moved = set(cur.rgst_no)
    hold_orig = app_link[app_link.rgst_no.isin(p.rgst_no) & ~app_link.rgst_no.isin(moved)]
    hold_new = cur[cur.corp.notna()][["rgst_no", "corp"]].rename(columns={"corp": "corp_code"})
    # 일부이전: 직전 보유자 유지 (직전 이전 기록이 있으면 그 권리자, 없으면 출원인)
    part = cur[cur.partial].rgst_no.unique()
    prev_t = t_[(t_.y < t_.ly) & t_.rgst_no.isin(part)]
    if len(prev_t):
        pl = prev_t.groupby("rgst_no").y.max().rename("py")
        prev_t = prev_t.merge(pl, on="rgst_no"); prev_t = prev_t[prev_t.y == prev_t.py]
    keep_prev = prev_t[prev_t.corp.notna()][["rgst_no", "corp"]].rename(columns={"corp": "corp_code"}) if len(prev_t) else pd.DataFrame(columns=["rgst_no", "corp_code"])
    part_orig = app_link[app_link.rgst_no.isin(set(part) - set(prev_t.rgst_no if len(prev_t) else []))]
    H = pd.concat([hold_orig, hold_new, keep_prev, part_orig]).drop_duplicates()
    H = H.merge(p[["rgst_no", "ksic", "alive", "ry"]], on="rgst_no")
    Ha = H[H.alive]
    nk = Ha.groupby(["corp_code", "ksic"]).size().rename("n_k").reset_index()
    Hr = Ha[Ha.ry >= T - 4]                      # 최근 5년 등록분(현재 제품 구성 반영 가설)
    nkr = Hr.groupby(["corp_code", "ksic"]).size().rename("n_k_rec").reset_index()
    nk = nk.merge(nkr, on=["corp_code", "ksic"], how="left").fillna({"n_k_rec": 0})
    nk["n_rec"] = nk.groupby("corp_code").n_k_rec.transform("sum")
    for win in (3, 8):                           # 창 길이 민감도
        r_ = Ha[Ha.ry >= T - win + 1].groupby(["corp_code", "ksic"]).size().rename(f"n_k_rec{win}").reset_index()
        nk = nk.merge(r_, on=["corp_code", "ksic"], how="left").fillna({f"n_k_rec{win}": 0})
        nk[f"n_rec{win}"] = nk.groupby("corp_code")[f"n_k_rec{win}"].transform("sum")
    nin = Ha.groupby("corp_code").size().rename("n_in")
    # 기업 생존율(출원인 기준, 제공범위, 존속기간 내 등록 중 생존)
    a0 = app_link.merge(p[["rgst_no", "alive", "exy"]], on="rgst_no")
    a0 = a0[a0.exy.isna() | (a0.exy > T)]
    sr = a0.groupby("corp_code").alive.agg(["mean", "size"])
    glob = a0.alive.mean()
    surv = np.where(sr["size"] >= 5, sr["mean"], glob); surv = pd.Series(surv, index=sr.index)
    o = out[(out.ry <= T) & (out.exy.isna() | (out.exy > T))].groupby("corp_code").size().rename("n_out_raw")
    F = pd.concat([nin, o], axis=1).fillna({"n_out_raw": 0})
    F = F[F.n_in.notna()]
    F["surv"] = surv.reindex(F.index).fillna(glob)
    F["n_out_est"] = F.n_out_raw * F.surv
    nk = nk.merge(F[["n_in", "n_out_est"]].reset_index(), on="corp_code")
    nk["w_pit"] = nk.n_k / np.maximum(nk.n_in + nk.n_out_est, nk.n_in)
    nk["w_pit_share"] = nk.n_k / nk.n_in
    scale = nk.n_in / np.maximum(nk.n_in + nk.n_out_est, nk.n_in)
    nk["w_pit_rec"] = np.where(nk.n_rec > 0, nk.n_k_rec / nk.n_rec.where(nk.n_rec > 0, 1) * scale, nk.w_pit)
    for win in (3, 8):
        nr = nk[f"n_rec{win}"]
        nk[f"w_pit_rec{win}"] = np.where(nr > 0, nk[f"n_k_rec{win}"] / nr.where(nr > 0, 1) * scale, nk.w_pit)
    nk["t0"] = T
    rows.append(nk)
    # 스냅샷판(파일럿 재현): 출원인 기준 생존 스톡 + right_list 누적 등록(권리자란 전체, 소멸 포함) 분모
    s0 = app_link.merge(p[["rgst_no", "ksic", "alive"]], on="rgst_no")
    s0 = s0[s0.alive]
    sk = s0.groupby(["corp_code", "ksic"]).size().rename("n_k").reset_index()
    sn = s0.groupby("corp_code").size().rename("n_alive")
    Nf = rl[rl.ry <= T].drop_duplicates(["corp_code", "rgst_no"]).groupby("corp_code").size().rename("N_full")
    sk = sk.merge(sn.reset_index(), on="corp_code").merge(Nf.reset_index(), on="corp_code", how="left")
    sk["w_snap"] = sk.n_k / np.maximum(sk.N_full.fillna(0), sk.n_alive)
    sk["w_snap_share"] = sk.n_k / sk.n_alive
    sk["t0"] = T
    snap_rows.append(sk[["corp_code", "t0", "ksic", "n_k", "n_alive", "N_full", "w_snap", "w_snap_share"]])
    print(T, "held firms", nin.size, "held-alive patents", int(nin.sum()), "moved-in rows", len(hold_new), "orig-link firms(alive)", s0.corp_code.nunique(),
          round(time.time() - t_start), "s", flush=True)
A = pd.concat(rows, ignore_index=True)
A[["corp_code", "t0", "ksic", "n_k", "n_k_rec", "n_rec", "n_k_rec3", "n_k_rec8", "n_in", "n_out_est", "w_pit", "w_pit_share", "w_pit_rec", "w_pit_rec3", "w_pit_rec8"]].to_parquet(os.path.join(C, f"alloc_pit{SUF}.parquet"), index=False)
S = pd.concat(snap_rows, ignore_index=True); S.to_parquet(os.path.join(C, f"alloc_snap{SUF}.parquet"), index=False)
g = A.groupby(["corp_code", "t0"]).w_pit.sum()
print("sum w_pit per firm quantiles", g.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
g2 = S.groupby(["corp_code", "t0"]).w_snap.sum()
print("sum w_snap per firm quantiles", g2.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
