# -*- coding: utf-8 -*-
r"""T4 R7 필수성 검증 하네스: 동결 공식 챔피언 R2-E03(분류 0.2·위험 0.4·로지스틱 0.2·HGB 0.2, 배깅 3, 시드 고정)을
개발 원점 O12·O14·O16·O18에서 피처만 바꿔 재학습. 동결 프레임(R2/runs/<O>/*.parquet)은 읽기만, 예측·결과는 ESSENTIAL_R7/runs/<VAR>/ 에만.
  python e7_t4_run.py exp <VAR> [--par 2]     VAR: CH(재현 점검) S_KSIC R_KSIC R_PROV
  python e7_t4_run.py job <VAR> <O> <comp> <k|->   (내부, 9분 이내)
  python e7_t4_run.py score <VAR>
기준선(base)은 공식 그대로(동결 예측 파일 복사) — skill 분모 불변."""
import os, sys, json, time, shutil, subprocess, copy
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); T4 = os.path.dirname(HERE); R2 = os.path.join(T4, "R2")
sys.path.insert(0, R2)
os.environ["T4R2_MODE"] = "dev"
import numpy as np, pandas as pd, pyarrow.parquet as pq
import r2lib as R

SPEC = json.load(open(os.path.join(R2, "specs", "E03.json"), encoding="utf-8"))["spec"]
KSB = ["port_ksic_n", "port_share_ksic", "aks_rate", "aks_rate1y", "aks_n", "aks_alive",
       "mac_prod_yoy", "mac_ship_yoy", "mac_inv_yoy", "mac_prod_rel", "mac_prod_3y"]
FAM = ["fam_n", "fam_cc", "fam_us", "fam_ep", "fam_jp", "fam_cn", "fam_wo"]
XC = ["ksic"] + KSB
DESC = {"CH": "재현 점검(변경 없음)",
        "S_KSIC": "대체: ksic 범주·출원인 KSIC 포트폴리오 2개·출원인×KSIC 이웃 불납률 4개·KSIC 경기지표 5개를 공식 IPC-KSIC 연계표 배정으로 재계산",
        "R_KSIC": "제거: 위 KSIC 의존 피처 11개 삭제 + ksic 범주 상수화",
        "R_PROV": "제거: KSIC 의존 11개 + ksic 범주 상수화 + 제공 5개국 패밀리 7개"}
VAR = None


def var_spec(v):
    s = copy.deepcopy(SPEC)
    drop = KSB if v == "R_KSIC" else (KSB + FAM if v == "R_PROV" else [])
    for c in s["components"].values():
        c["num"] = [x for x in c["num"] if x not in drop]
        if "log1p" in c: c["log1p"] = [x for x in c["log1p"] if x not in drop]
    return s


def src(o): return os.path.join(R2, "runs", o)


def my_odir(o):
    p = os.path.join(HERE, "runs", VAR, o); os.makedirs(os.path.join(p, "pred"), exist_ok=True)
    return p


def my_read(o, frame, cols):
    fn = os.path.join(src(o), frame)
    have = set(pq.ParquetFile(fn).schema.names)
    d = pq.read_table(fn, columns=[c for c in cols if c in have]).to_pandas()
    if VAR == "S_KSIC":
        need = [c for c in XC if c in d.columns]
        if need:
            x = pd.read_parquet(os.path.join(HERE, "xcols", o, frame), columns=need)
            assert len(x) == len(d)
            for c in need:
                d[c] = x[c].values.astype(d[c].dtype) if c != "ksic" else x[c].values
    elif VAR in ("R_KSIC", "R_PROV") and "ksic" in d.columns:
        d["ksic"] = "NA"
    return d


def my_combine(o, spec, comp):
    key = R.comp_key(spec, comp)
    fn = os.path.join(my_odir(o), "pred", f"{key}.npy")
    if os.path.exists(fn): return np.load(fn)
    nb = spec["components"][comp].get("n_bag", 1)
    z = np.mean([np.load(os.path.join(my_odir(o), "pred", f"{key}_b{k}.npy")) for k in range(nb)], 0)
    p = 1 / (1 + np.exp(-z))
    if comp.startswith("haz"):
        row = pq.read_table(os.path.join(src(o), "ev_haz.parquet"), columns=["row"]).to_pandas().row.values
        n = pq.ParquetFile(os.path.join(src(o), "ev.parquet")).metadata.num_rows
        p = R.HZ.combine(pd.DataFrame({"row": row}), p, n)
    np.save(fn, p.astype("float64"))
    return p


_EVM = {}
def my_ev_meta(o):
    if o not in _EVM:
        _EVM[o] = pd.read_parquet(os.path.join(src(o), "ev.parquet"), columns=["pid", "L", "ksic", "acd", "atype", "y"])
    return _EVM[o]


def patch(v):
    global VAR
    VAR = v
    R.odir = my_odir; R._read_frame = my_read; R.combine_xgb = my_combine; R.ev_meta = my_ev_meta


def copy_base(o):
    import hashlib
    bkey = "base_" + hashlib.md5(json.dumps(SPEC["baseline"], sort_keys=True).encode()).hexdigest()[:10]
    dst = os.path.join(my_odir(o), "pred", f"{bkey}.npy")
    if not os.path.exists(dst):
        shutil.copyfile(os.path.join(src(o), "pred", f"{bkey}.npy"), dst)


def run_jobs(v, jobs, par):
    q = list(jobs); run = []
    while q or run:
        while q and len(run) < par:
            o, comp, k = q.pop(0)
            lg = open(os.path.join(HERE, "logs", f"job_{v}_{o}_{comp}_{k}.log"), "w", encoding="utf-8")
            p = subprocess.Popen([sys.executable, "-W", "ignore", os.path.abspath(__file__), "job", v, o, comp, str(k) if k is not None else "-"],
                                 stdout=lg, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1"))
            run.append((p, (o, comp, k), lg, time.time())); print(f"[start] {o} {comp} {k} pid={p.pid}", flush=True)
        time.sleep(5)
        for r in list(run):
            p, j, lg, t0 = r
            if p.poll() is not None:
                lg.close(); run.remove(r); print(f"[done] {j} rc={p.returncode} {time.time() - t0:.0f}s", flush=True)
                if p.returncode != 0:
                    for rr in run: rr[0].kill()
                    sys.exit(f"작업 실패 {j}")


def score(v):
    s = var_spec(v)
    res = {}
    for o, _, _ in R.ORIGINS:
        copy_base(o)
        res[o] = R.score_origin(o, s)
    ref = json.load(open(os.path.join(R2, "runs", "res_E03.json"), encoding="utf-8"))
    out = dict(variant=v, desc=DESC[v], by_origin=res, mean_auc=float(np.mean([r["auc"] for r in res.values()])),
               mean_skill=float(np.mean([r["skill"] for r in res.values()])),
               champ_E03=dict(by_origin={o: dict(auc=ref[o]["auc"], skill=ref[o]["skill"]) for o in ref},
                              mean_auc=float(np.mean([ref[o]["auc"] for o in ref])), mean_skill=float(np.mean([ref[o]["skill"] for o in ref]))))
    out["d_skill_vs_champ"] = out["mean_skill"] - out["champ_E03"]["mean_skill"]
    out["n_origins_worse"] = int(sum(res[o]["skill"] < ref[o]["skill"] for o in res))
    json.dump(out, open(os.path.join(HERE, f"res_{v}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ["variant", "mean_auc", "mean_skill", "d_skill_vs_champ", "n_origins_worse"]}, ensure_ascii=False), flush=True)
    print({o: (round(r["auc"], 4), round(r["skill"], 4), r["comps"]) for o, r in res.items()}, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    patch(a[1])
    if a[0] == "job":
        s = var_spec(a[1]); o, comp, k = a[2], a[3], a[4]
        if k != "-": R.fit_predict_xgb_bag(o, s, comp, int(k))
        else: R.fit_predict_sk(o, s, comp)
        print("peak", R.peak_mem_gb(), flush=True)
    elif a[0] == "exp":
        par = int(a[a.index("--par") + 1]) if "--par" in a else 2
        only = a[a.index("--only") + 1].split(",") if "--only" in a else None
        s = var_spec(a[1])
        jobs = [j for j in R.jobs_for(s) if j[1] != "base"]
        if only: jobs = [j for j in jobs if f"{j[0]}:{j[1]}" in only]
        print(a[1], DESC[a[1]], "jobs", len(jobs), flush=True)
        run_jobs(a[1], jobs, par)
        if not only: score(a[1])
    elif a[0] == "score":
        score(a[1])
