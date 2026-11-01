# 기술 철수·휴면 신호: KR 전체 출원(kr_patents, 미등록 포함 1,087,511건) 중 L 이전에 공개(또는 등록)되어 알 수 있었던 출원만 사용.
# 권리자(현재 권리자 이름)의 최근 출원 활동, 이 특허의 IPC4에서의 최근 출원 유무 → cache/tech_L{Y}.parquet
import sys, time, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from common import *
t0 = time.time()
K = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\공모전데이터\작업데이터\kr_patents.parquet"
k = pd.read_parquet(K, columns=['출원번호', '출원일', '공개일자', '등록일자', '출원인명', '메인IPC'])
k['appl'] = pd.to_datetime(k['출원일'].astype(str), format='%Y%m%d', errors='coerce')
k['known'] = pd.concat([pd.to_datetime(k['공개일자'], format='%Y%m%d', errors='coerce'),
                        pd.to_datetime(k['등록일자'], format='%Y%m%d', errors='coerce')], axis=1).min(axis=1)
k = k.dropna(subset=['known'])
k['nm'] = k['출원인명'].fillna('').str.split('|').str[0].map(norm)
k['ipc4'] = k['메인IPC'].fillna('').str[:4]
k = k[['출원번호', 'appl', 'known', 'nm', 'ipc4']]
print('kr apps with known date', len(k), round(time.time() - t0, 1), flush=True)
for Y in range(2003, LMAX + 1):
    L = pd.Timestamp(Y, 1, 1); B3 = pd.Timestamp(Y - 3, 1, 1); B6 = pd.Timestamp(Y - 6, 1, 1)
    s = pd.read_parquet(f'cache/panel_L{Y}.parquet', columns=['rgst_no', 'cur', 'ipc4'])
    kk = k[k.known < L]
    r3 = kk[kk.appl >= B3]; r6 = kk[(kk.appl >= B6) & (kk.appl < B3)]
    s['k_app3'] = s.cur.map(r3.groupby('nm').size()).fillna(0)
    s['k_app_prev'] = s.cur.map(r6.groupby('nm').size()).fillna(0)
    s['k_mom'] = np.log((s.k_app3 + 1) / (s.k_app_prev + 1))
    idx = pd.MultiIndex.from_arrays([s.cur, s.ipc4])
    s['k_ipc_app3'] = idx.map(r3.groupby(['nm', 'ipc4']).size()).fillna(0).values
    s['k_ipc_app_prev'] = idx.map(r6.groupby(['nm', 'ipc4']).size()).fillna(0).values
    s['k_ipc_share'] = s.k_ipc_app3 / s.k_app3.replace(0, np.nan)
    s['tech_exit'] = ((s.k_ipc_app_prev > 0) & (s.k_ipc_app3 == 0)).astype(int)
    last = kk.groupby('nm').appl.max()
    s['k_dormant'] = ((L - s.cur.map(last)).dt.days / 365.25).fillna(99)
    s.drop(columns=['cur', 'ipc4']).assign(L=Y).to_parquet(f'cache/tech_L{Y}.parquet', index=False)
    print(Y, len(s), 'tech_exit rate', round(s.tech_exit.mean(), 3), round(time.time() - t0, 1), flush=True)
