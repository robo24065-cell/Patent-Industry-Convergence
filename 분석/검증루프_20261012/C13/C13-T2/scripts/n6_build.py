# -*- coding: utf-8 -*-
"""N6 P0-a: 공식 final()과 같은 build("final")로 패널을 다시 만든다(트랙에 final 패널 캐시가 없음). n6_features.build는 final 모드에서 파일을 쓰지 않는다.
트랙 cache/proj_ptsim.parquet은 final 이후(21:45) future 모드가 다시 썼으므로 P0에서 동결 예측과 대조해 영향 여부를 판정한다. → cache/n6/panel_final.parquet"""
import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "n6"))
from c13util import *
import n6_features as F
D = F.build("final")
D.to_parquet(os.path.join(tdir("n6"), "panel_final.parquet"), index=False)
tlog("saved", D.shape, D.groupby("t0").y.agg(["size", "sum"]).to_dict())
