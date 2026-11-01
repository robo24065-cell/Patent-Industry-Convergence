# -*- coding: utf-8 -*-
"""T4 시드 묶음 학습 작업 1개(프로세스 1개, 9분 이내). 동결 스펙 R2-E03, r2lib 복사본(scripts/t4mirror, sha256 원본 동일)을 그대로 부르고
출력 경로(odir)만 이 폴더 cache/t4/b{b}/ 로 돌린다(모형 저장 안 함). 학습 프레임은 cache/t4/frames(공식 runs_final/F22 사본, sha256 동일)를 하드링크.
사용: python t4_job.py <b> cls|haz <k>   또는   python t4_job.py <b> hgb|logit|base
 - XGB 배깅 구성원 k: r2lib.fit_predict_xgb_bag 그대로(seed=k, k>0이면 열 순열 rng(k)). 묶음 b의 k = 3b, 3b+1, 3b+2.
 - HGB: random_state를 b로 바꾼다(공식 0). sklearn 클래스만 감싸 random_state를 덮어쓴다(그 밖의 인자 불변)."""
import os, sys, json, time
sys.dont_write_bytecode = True
os.environ["T4R2_MODE"] = "final"; os.environ.pop("T4R2_SAVE", None)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from c13util import *
sys.path.insert(0, os.path.join(HERE, "t4mirror", "R2"))
import numpy as np
import r2lib as R

b = int(sys.argv[1]); comp = sys.argv[2]; k = int(sys.argv[3]) if len(sys.argv) > 3 else None
FR = os.path.join(CACHE, "t4", "frames")
BD = os.path.join(CACHE, "t4", f"b{b}")
os.makedirs(os.path.join(BD, "pred"), exist_ok=True)
for f in os.listdir(FR):
    if not os.path.exists(os.path.join(BD, f)):
        os.link(os.path.join(FR, f), os.path.join(BD, f))
R.odir = lambda o: BD
R.SAVE = False
FS = json.load(open(os.path.join(HERE, "t4mirror", "R2", "final_spec_r2.json"), encoding="utf-8")); SPEC = FS["spec"]
t = time.time()
if comp in ("cls", "haz"):
    assert k in (3 * b, 3 * b + 1, 3 * b + 2)
    fn = R.fit_predict_xgb_bag("F22", SPEC, comp, k)
elif comp == "hgb":
    import sklearn.ensemble as SE
    _H = SE.HistGradientBoostingClassifier

    def HB(*a, **kw):          # 진짜 HGB 인스턴스를 돌려주는 생성 함수(get_params 서명 규칙 회피)
        kw["random_state"] = b
        return _H(*a, **kw)
    SE.HistGradientBoostingClassifier = HB
    p = R.fit_predict_sk("F22", SPEC, "hgb")
else:
    assert comp in ("logit", "base")
    p = R.fit_predict_sk("F22", SPEC, comp)
tlog("job done", b, comp, k, round(time.time() - t), "peak", R.peak_mem_gb())
