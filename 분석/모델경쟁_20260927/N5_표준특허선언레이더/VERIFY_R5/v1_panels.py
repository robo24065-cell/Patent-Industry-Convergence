# -*- coding: utf-8 -*-
"""R5 레드팀 1단계: n5lib 동결 코드로 기초표·패널을 캐시 없이(새 캐시 폴더) 다시 만든다. 원본 캐시는 읽지 않는다."""
import sys, os, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import n5lib
n5lib.CACHE = os.path.join(HERE, "cache")   # 원본 cache 대신 빈 폴더 → load_kr/load_family/load_ext 원자료에서 재계산
import pandas as pd
t = time.time()
k = n5lib.load_kr("26429"); print("kr", len(k), round(time.time()-t), flush=True)
ex = n5lib.load_family(k, "26429"); print("fam", len(ex), round(time.time()-t), flush=True)
E = n5lib.load_ext("26429"); print("ext", round(time.time()-t), flush=True)
out = []
for t0 in range(2012, 2021):
    out.append(n5lib.build_panel(t0, "26429", "dev", k, ex, E)); print("panel", t0, len(out[-1]), int(out[-1].y.sum()), round(time.time()-t), flush=True)
pd.concat(out, ignore_index=True).to_parquet(os.path.join(HERE, "cache", "panel_train.parquet"), index=False)
te = n5lib.build_panel(2022, "26429", "final", k, ex, E)
te.to_parquet(os.path.join(HERE, "cache", "panel_final.parquet"), index=False)
print("final", len(te), int(te.y.sum()), round(time.time()-t), flush=True)
