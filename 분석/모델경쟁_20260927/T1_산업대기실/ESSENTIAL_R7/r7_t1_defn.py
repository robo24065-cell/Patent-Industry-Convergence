# -*- coding: utf-8 -*-
"""T1 정의 시험: '법인 × 산업 첫 진입'을 제공 KSIC 라벨 대신 연계표로 정의하면 같은 사건인가?
범위안 등록특허(cache/ins_firm: 원출원인 코드 × 특허 × 제공 KSIC × 메인 IPC), 개발 원점 c=2008..2012, 진입창 (c, c+4] (<= 2016, 봉인 무관).
(a) 제공: (법인, 5자리 KSIC) 첫 진입 / (b) 제공 라벨을 소분류 3자리로 올린 첫 진입 / (c) 연계표(free) 소분류 3자리 첫 진입(25개 안만).
(b)와 (c)를 같은 해상도(3자리)에서 비교: 자카드·정밀도·재현율. (a)는 연계표로는 정의 자체가 불가(5자리 없음).
산출: defn_t1.json"""
import os, sys, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent; T1 = HERE.parent
sys.path.insert(0, str(T1.parent/'N9_정부RnD과제_특허성과예측'/'ESSENTIAL_R7'))
import numpy as np, pandas as pd
from r7_concord import assign, G50

ins = pd.read_parquet(T1/'cache'/'ins_firm.parquet', columns=['rgst_no', 'applicant_cd', 'ksic', 'ay', 'mipc'])
ins = ins[ins.ay >= 2000].copy()
g, _ = assign(ins.mipc.values, 'free'); ins['cg'] = g; ins['pg'] = ins.ksic.astype(str).str[:3]
res = dict(n_rows=int(len(ins)), conc_in_scope_share=float(np.isin(g, G50).mean()), agree3_free=float((ins.cg == ins.pg).mean()))
out = {}
for c in range(2008, 2013):
    def entries(col, mask=None):
        d = ins if mask is None else ins[mask]
        first = d.groupby(['applicant_cd', col]).ay.min()
        return set(first[(first > c) & (first <= c + 4)].index)
    e5 = entries('ksic'); e3p = entries('pg'); e3c = entries('cg', np.isin(ins.cg.values, G50))
    # 제공 3자리와 연계표 3자리: 같은 법인 모집단(원점까지 범위안 1건 이상 + 이후)에서 비교
    tp = len(e3p & e3c)
    out[c] = dict(n_entry5_provided=len(e5), n_entry3_provided=len(e3p), n_entry3_concord=len(e3c),
                  jaccard3=tp / max(1, len(e3p | e3c)), precision3_concord=tp / max(1, len(e3c)), recall3_concord=tp / max(1, len(e3p)))
res['by_origin'] = out
res['mean_jaccard3'] = float(np.mean([v['jaccard3'] for v in out.values()]))
res['mean_precision3'] = float(np.mean([v['precision3_concord'] for v in out.values()]))
res['mean_recall3'] = float(np.mean([v['recall3_concord'] for v in out.values()]))
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open(HERE/'defn_t1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
