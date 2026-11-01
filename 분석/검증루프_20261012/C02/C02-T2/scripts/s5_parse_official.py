# C02-T2 s5: 내려받은 KOSIS(특허청 지식재산통계) 원문 파싱 → 연도별 특허/실용 출원·등록 (읽기만, 원문 수정 없음)
import json, os, sys, glob
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
D = sys.argv[1]; OUT = sys.argv[2]
def load(fn):
    b = open(fn, 'rb').read()
    try: t = b.decode('utf-8')
    except UnicodeDecodeError: t = b.decode('cp949', errors='replace')
    return json.loads(t)
rows = []
for f in ['data_TX_13801_A004', 'data_TX_13801_A006', 'data_TX_13801_A046', 'data_TX_13801_A040_1', 'data_TX_13801_A080_2', 'data_TX_13801_A006_M']:
    d = load(os.path.join(D, f + '.json'))
    if isinstance(d, dict): print(f, d); continue
    df = pd.DataFrame(d); df['src'] = f; rows.append(df)
    cols = [c for c in df.columns if c.endswith('_NM') and c.startswith('C')]
    print('##', f, len(df), df.TBL_NM.iloc[0], df.ITM_NM.unique()[:6], {c: list(df[c].unique()[:12]) for c in cols}, df.PRD_DE.min(), df.PRD_DE.max(), df.LST_CHN_DE.max())
a = pd.concat(rows, ignore_index=True)
a.to_csv(os.path.join(OUT, 's5_kosis_long.csv'), index=False, encoding='utf-8-sig')
