# s4: bias_checks.csv 작성(s3 결과 + 명단 전체 분포 + NPS 단위 규모)
import sys, os, json; sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from lib6 import *
r = pd.read_csv(os.path.join(C, 's3_bias_rows.csv'))
s0 = json.load(open(os.path.join(C, 's0_venture_check.json'), encoding='utf-8'))
s1 = json.load(open(os.path.join(C, 's1_A_log.json'), encoding='utf-8'))
s2 = json.load(open(os.path.join(C, 's2_B_log.json'), encoding='utf-8'))
add = [
 ('C06-A', '명단 전체', '행 / 정확 중복 / 정규화 상호 중복 행 / 상호+주소 중복 행', s0['download_rows'], s0['dup_exact_rows'], f"{s0['dup_name_rows']} / {s0['dup_name_addr_rows']}"),
 ('C06-A', '명단 전체', '신규 / 재확인', s0['new_reconf'].get('신규'), s0['new_reconf'].get('재확인'), ''),
 ('C06-A', '명단 전체', '유형(혁신성장/벤처투자/연구개발/예비)', json.dumps(s0['type'], ensure_ascii=False), '', ''),
 ('C06-A', '명단 전체', '유효기간 일수(최소–최대)', s0['duration_days_quantiles']['0'], s0['duration_days_quantiles']['1'], '모두 3년'),
 ('C06-A', '명단 전체', '법인 표기 비율', s0['corp_form_share'], '', ''),
 ('C06-A', '명단 전체', '명단상 유효 수 2024-12-31 / 2025-12-31 / 2026-04-30', s0['valid_on_2024-12-31'], s0['valid_on_2025-12-31'], s0['valid_on_2026-04-30']),
 ('C06-A', '연결 정밀도', 'NPS 시도 일치 / 출원인주소 시도 일치', s1['P2c']['agree_nps'], s1['P2c']['agree_pat_addr'], f"쌍 {s1['P2c']['pairs']}, 모호 상호 비율 {s1['P2c']['amb_names_share_pairs']}"),
]
for k in ['201512', '202012', '202312', '202412']:
    v = s2[k]; add.append(('C06-B', f'S0 {k}', '연결 단위(E0≥5) / NPS 법인 기업키(E0≥5) / 다사업장 비율', v['units'], v['nps_corp_keys_E0ge5'], v.get('multi_wkpl_share_S0', '(잠금 S0: 크기만)')))
r2 = pd.concat([r, pd.DataFrame(add, columns=['후보', '비교', '지표', '값1', '값2', '비고'])], ignore_index=True)
r2.to_csv(os.path.join(W, 'bias_checks.csv'), index=False, encoding='utf-8-sig')
print(r2.to_string())
