# [2단계 감사 수정 F1·F2] 권리자 출원활동 피처 재구성 → cache/act_L{Y}.parquet
# 감사에서 발견한 문제:
#  (1) registry.applicant / owner 의 이름은 2026 수집 시점의 '현재 명칭'(예: 2008년 등록분도 '포스코홀딩스', '에이치디현대인프라코어')이다.
#      kr_patents.출원인명은 출원 당시 명칭(예: '포스코', '두산인프라코어')이라 s24(k_*)는 권리자명과 이름공간이 어긋나고,
#      '나중에 개명한 권리자'가 L 시점에 휴면(k_dormant=99)처럼 보이는 미래정보 통로가 된다.
#  (2) s21 own_app3/own_app_prev/own_years 는 registry(=결국 등록된 출원)만 세므로 'L 이후 등록 여부'라는 미래 결과로 조건화된다
#      (L 이후 출원을 포기하는 권리자 = 철수 권리자의 출원 수가 과소 계산).
# 수정: kr_patents 전체 출원(등록·미등록, L 이전 공개/등록분만)을 쓰고, 출원인명을 registry 현재명칭 이름공간으로 정렬한다.
#  - 등록된 출원: 출원번호로 registry 첫 출원인(현재명칭)을 직접 사용
#  - 미등록 출원: '출원당시명→현재명칭' 별칭표(등록 출원 쌍에서 최빈 대응, 점유율≥0.6·3건 이상)로 변환, 없으면 원래 이름
#  별칭표는 이름 정렬(동일 실체 식별)에만 쓰며 사건·결과 정보는 담지 않는다.
import sys, time, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from common import *
t0 = time.time()
def P(*a): print(*a, round(time.time() - t0, 1), flush=True)
K = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\공모전데이터\작업데이터\kr_patents.parquet"
k = pd.read_parquet(K, columns=['출원번호', '출원일', '공개일자', '등록일자', '출원인명', '메인IPC']).drop_duplicates('출원번호')
k['appl'] = pd.to_datetime(k['출원일'].astype(str), format='%Y%m%d', errors='coerce')
k['known'] = pd.concat([pd.to_datetime(k['공개일자'], format='%Y%m%d', errors='coerce'),
                        pd.to_datetime(k['등록일자'], format='%Y%m%d', errors='coerce')], axis=1).min(axis=1)
k['k_nm'] = k['출원인명'].fillna('').map(norm); k['ipc4'] = k['메인IPC'].fillna('').str[:4]
k['applNo'] = k['출원번호'].astype(str)
pat = pd.read_parquet('cache/pat.parquet', columns=['rgst_no', 'applNo'])
app = pd.read_parquet('cache/app.parquet', columns=['rgst_no', 'seq', 'applicantName'])
reg1 = app.sort_values(['rgst_no', 'seq']).groupby('rgst_no').applicantName.first().map(norm)
pat['reg_nm'] = pat.rgst_no.map(reg1); pat['applNo'] = pat.applNo.astype(str)
k = k.merge(pat[['applNo', 'reg_nm']], on='applNo', how='left')
pairs = k.dropna(subset=['reg_nm']); pairs = pairs[pairs.k_nm != '']
cnt = pairs.groupby(['k_nm', 'reg_nm']).size().reset_index(name='c')
tot = cnt.groupby('k_nm').c.transform('sum'); cnt['sh'] = cnt.c / tot
al = cnt[(cnt.sh >= 0.6) & (cnt.c >= 3) & (cnt.k_nm != cnt.reg_nm)].set_index('k_nm').reg_nm
k['nm'] = k.reg_nm.fillna(k.k_nm.map(al)).fillna(k.k_nm)
P('kr apps', len(k), 'registered', int(k.reg_nm.notna().sum()), 'alias entries', len(al),
  'unregistered renamed via alias', int((k.reg_nm.isna() & k.k_nm.isin(al.index)).sum()))
k = k.dropna(subset=['known', 'appl'])
k = k[['appl', 'known', 'nm', 'ipc4']]
for Y in range(2003, LMAX + 1):
    L = pd.Timestamp(Y, 1, 1); B3 = pd.Timestamp(Y - 3, 1, 1); B6 = pd.Timestamp(Y - 6, 1, 1)
    s = pd.read_parquet(f'cache/panel_L{Y}.parquet', columns=['rgst_no', 'cur', 'ipc4'])
    kk = k[k.known < L]
    r3 = kk[kk.appl >= B3]; r6 = kk[(kk.appl >= B6) & (kk.appl < B3)]
    s['a_app3'] = s.cur.map(r3.groupby('nm').size()).fillna(0)
    s['a_app_prev'] = s.cur.map(r6.groupby('nm').size()).fillna(0)
    s['a_mom'] = np.log((s.a_app3 + 1) / (s.a_app_prev + 1))
    idx = pd.MultiIndex.from_arrays([s.cur, s.ipc4])
    s['a_ipc_app3'] = idx.map(r3.groupby(['nm', 'ipc4']).size()).fillna(0).values
    s['a_ipc_app_prev'] = idx.map(r6.groupby(['nm', 'ipc4']).size()).fillna(0).values
    s['a_ipc_share'] = s.a_ipc_app3 / s.a_app3.replace(0, np.nan)
    s['a_tech_exit'] = ((s.a_ipc_app_prev > 0) & (s.a_ipc_app3 == 0)).astype(int)
    s['a_dormant'] = ((L - s.cur.map(kk.groupby('nm').appl.max())).dt.days / 365.25).fillna(99)
    s['a_years'] = ((L - s.cur.map(kk.groupby('nm').appl.min())).dt.days / 365.25).fillna(-1)
    s.drop(columns=['cur', 'ipc4']).assign(L=Y).to_parquet(f'cache/act_L{Y}.parquet', index=False)
    P(Y, len(s), 'a_app3>0', round((s.a_app3 > 0).mean(), 3), 'dormant99', round((s.a_dormant >= 99).mean(), 3))
