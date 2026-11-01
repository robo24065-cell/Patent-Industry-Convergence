# -*- coding: utf-8 -*-
r"""C19-T2 학습 작업 1개(프로세스 1개, 540초 제한). 동결 스펙 R2-E03, r2lib 사본(C13-T2 scripts/t4mirror, sha256 원본 동일)을 그대로 부른다.
바꾸는 것: (1) 출력 경로 odir → C19-T2/cache/runs/{var} (2) 프레임은 공식 runs_final/F22 에서 읽기만 하고, 읽은 직후 fam_n 열만 cache/famn/{var}__{frame}.npy 로 교체.
모형 저장 안 함(SAVE=False). base 는 fam 을 쓰지 않으므로 P0 에서만 학습(frame_base·ev 를 run 폴더로 복사해 읽음).
사용: python s2_job.py <var> cls|haz <k>  |  python s2_job.py <var> hgb|logit|base"""
import os, sys, json, time, shutil
sys.dont_write_bytecode = True
os.environ["T4R2_MODE"] = "final"; os.environ.pop("T4R2_SAVE", None)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from lib19 import *
sys.path.insert(0, J(MIRROR, "R2"))
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import r2lib as R

var = sys.argv[1]; comp = sys.argv[2]; k = int(sys.argv[3]) if len(sys.argv) > 3 else None
BD = J(C, "runs", var); os.makedirs(J(BD, "pred"), exist_ok=True)
R.odir = lambda o: BD
R.SAVE = False
_orig = R._read_frame
LOG = {}


def _read_frame(o, frame, cols):
    fn = J(F22, frame)
    have = set(pq.ParquetFile(fn).schema.names)
    need = [c for c in cols if c in have]
    d = pq.read_table(fn, columns=need).to_pandas()
    if "fam_n" in d.columns:
        v = np.load(J(C, "famn", f"{var}__{frame[:-8]}.npy"))
        assert len(v) == len(d), (frame, len(v), len(d))
        LOG[frame] = dict(rows=len(d), changed_vs_stored=int((v != d.fam_n.values.astype("float32")).sum()))
        d["fam_n"] = v.astype(d.fam_n.dtype)
    return d


R._read_frame = _read_frame
FS = json.load(open(J(MIRROR, "R2", "final_spec_r2.json"), encoding="utf-8")); SPEC = FS["spec"]
t = time.time()
if comp in ("cls", "haz"):
    assert k in (0, 1, 2)
    R.fit_predict_xgb_bag("F22", SPEC, comp, k)
elif comp in ("hgb", "logit"):
    R.fit_predict_sk("F22", SPEC, comp)
else:
    assert comp == "base" and var == "P0"
    for f in ("frame_base.parquet", "ev.parquet"):
        if not os.path.exists(J(BD, f)):
            shutil.copyfile(J(F22, f), J(BD, f))
    R.fit_predict_sk("F22", SPEC, "base")
out("job done", var, comp, k, round(time.time() - t), "peak", R.peak_mem_gb(), json.dumps(LOG))
