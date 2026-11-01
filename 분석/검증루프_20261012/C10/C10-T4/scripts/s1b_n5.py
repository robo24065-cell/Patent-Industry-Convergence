import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
N5 = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\N5_표준특허선언레이더"
LP = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012"
d = pd.read_parquet(os.path.join(N5, "final_preds_r4.parquet")); print(d.columns.tolist(), d.shape)
fr = json.load(open(os.path.join(N5, "final_result_r4.json"), encoding="utf-8"))
print(json.dumps(fr, ensure_ascii=False)[:1500])
n5 = json.load(open(os.path.join(LP, r"C04\C04-T2\n5_rescore.json"), encoding="utf-8"))
def find(o, key, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key: print("C04", path + "/" + k, v)
            find(v, key, path + "/" + k)
find(n5, "diff_vs_official")
ycol = [c for c in d.columns if c in ("y", "label", "pos", "y_strict")]
print("ycol", ycol)
out = {}
y = d[ycol[0]].values
for c in d.columns:
    if c != ycol[0] and np.issubdtype(d[c].dtype, np.number) and d[c].nunique() > 2:
        try: out[c] = roc_auc_score(y, d[c].values)
        except Exception as e: pass
print(out)
json.dump(dict(cols=d.columns.tolist(), aucs=out), open(os.path.join(LP, r"C10\C10-T4\cache\s1b_n5.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
