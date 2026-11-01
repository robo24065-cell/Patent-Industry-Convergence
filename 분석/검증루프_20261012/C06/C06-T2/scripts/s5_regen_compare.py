# C06-T2 s5: 샌드박스 재생성(regen/) CSV·PNG와 제출 figs의 CSV·PNG 대조
import os, sys, json, hashlib, numpy as np, pandas as pd
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8')
REP = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012\figs"
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C06\C06-T2"
R = os.path.join(OUT, 'regen'); res = {}
names = ['fig01_framework', 'fig02_scorecard_9tracks', 'fig03_necessity_firm_industry_match', 'fig04_within_code_divergence',
         'fig05_rta_map_5x50', 'fig06_quadrant_tech_export', 'fig07_t3_future_breakout_by_ksic', 'fig08_t1_waiting_room_index',
         'fig09_n9_topk_vs_baseline', 'fig10_validation_design']
for n in names:
    a = pd.read_csv(os.path.join(REP, n + '.csv'), encoding='utf-8-sig'); b = pd.read_csv(os.path.join(R, n + '.csv'), encoding='utf-8-sig')
    r = {'csv_bytes_equal': open(os.path.join(REP, n + '.csv'), 'rb').read() == open(os.path.join(R, n + '.csv'), 'rb').read(),
         'csv_shape': [list(a.shape), list(b.shape)]}
    try:
        num = a.select_dtypes('number').columns
        r['csv_max_abs_num_diff'] = float((a[num] - b[num]).abs().max().max()) if len(num) else 0.0
        r['csv_text_equal'] = a.drop(columns=num).fillna('').astype(str).equals(b.drop(columns=num).fillna('').astype(str))
    except Exception as e:
        r['csv_err'] = str(e)
    ia = np.asarray(Image.open(os.path.join(REP, n + '.png')).convert('RGB')).astype(int)
    ib = np.asarray(Image.open(os.path.join(R, n + '.png')).convert('RGB')).astype(int)
    r['png_size'] = [list(ia.shape), list(ib.shape)]
    if ia.shape == ib.shape:
        d = np.abs(ia - ib).max(axis=2); r['png_pixels_diff'] = int((d > 0).sum()); r['png_pixels_diff_gt32'] = int((d > 32).sum())
    res[n] = r
json.dump(res, open(os.path.join(OUT, 'cache', 's5_regen_compare.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for n, r in res.items(): print(n, r)
