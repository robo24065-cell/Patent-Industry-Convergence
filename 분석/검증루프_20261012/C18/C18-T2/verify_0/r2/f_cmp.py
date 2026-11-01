# C18-T2 verify_0 r2: 독립 변형(myV1E, myV1Es) 결과를 원 발견 V1E 저장값과 대조
import os, json, pandas as pd
R = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(os.path.dirname(os.path.dirname(R)), 'cache')
out = {}
for V, base in [('find_V1E', os.path.join(C, 'map_V1E')), ('myV1E', os.path.join(R, 'map_myV1E')), ('myV1Es', os.path.join(R, 'map_myV1Es')), ('P0', os.path.join(C, 'map_P0'))]:
    t = pd.read_csv(os.path.join(base, 't1_variance_decomposition.csv'), encoding='utf-8-sig')
    t = t.set_index(t.columns[0])
    m3 = json.load(open(os.path.join(base, 'm3c_contemporaneous.json'), encoding='utf-8'))
    m4 = json.load(open(os.path.join(base, 'm4_summary.json'), encoding='utf-8'))
    out[V] = {'KRorigin_chg_share': float(t.iloc[8, 2]), 'KRorigin_chg_p': float(t.iloc[8, 4]), 'g_KR': float(t.iloc[1, 2]), 'g_5': float(t.iloc[0, 2]),
              'purged': {k: m3[k] for k in m3 if k.startswith('origin_purged')}, 'raw_EP': m3['raw_EP'], 'raw_US': m3['raw_US'],
              'm4_0817': m4['rho_patRTAchange_expRCAchange_0817'], 'row8_label': t.index[8]}
json.dump(out, open(os.path.join(R, 'f_cmp.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for k, v in out.items(): print(k, json.dumps(v, ensure_ascii=False))
# 전 파일 동일성 myV1E vs find_V1E
for fn in ['t1_variance_decomposition.csv', 'm3c_contemporaneous.json', 'm4_summary.json', 'm2_checks.json', 't1_meta.json', 'm2_country_summary.csv']:
    a = open(os.path.join(C, 'map_V1E', fn), 'rb').read(); b = open(os.path.join(R, 'map_myV1E', fn), 'rb').read()
    print('identical', fn, a == b)
