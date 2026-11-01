# -*- coding: utf-8 -*-
r"""C18-T2 s3f: family 토큰 번호 형식 이중 표기(US7324782 / US07324782, EP3955501 / EP03955501)의 T4 노출 정량.
 T4 동결 스펙 fam_n = 추정 공개연도 ≤ L연도−1 인 패밀리 토큰 수(feat2.fam_dated, s2_fam_tokens 원문 토큰 단위). fam_cc·fam_us 등은 국가 수·이진이라 영향 없음.
 (1) 재현: 원문 토큰으로 계산한 fam_n 이 최종시험 프레임(R2/runs_final/F22/ev.parquet, L=2022-01-01)의 저장값과 같은지
 (2) 정규화 토큰(국가 접두 + 숫자 앞 0 제거) 기준 fam_n_norm 과의 차이: 바뀌는 시험 단위 비율, 비, 순위상관, 토큰 연도별 이중 표기 비중(시점 일관성)
 읽기만. → cache/s3f_t4_famn.json"""
import os, sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
import numpy as np, pandas as pd
from scipy.stats import spearmanr
T4 = MC + r'\T4_특허포기예측'
pt = pd.read_parquet(T4 + r'\cache\patent.parquet', columns=['rgst_no', 'rgstDate', 'applNo'])
pt = pt[pt.rgstDate.notna() & (pt.rgstDate.astype(str).str.strip() != '') & (pt.rgst_no.str[:2] == '10')].reset_index(drop=True)
pt['pid'] = np.arange(len(pt))
tk = pd.read_parquet(T4 + r'\cache\s2_fam_tokens.parquet')
tk['norm'] = tk.tok.str[:2] + tk.tok.str[2:].str.lstrip('0')
m = pd.Series(pt.pid.values, index=pt.applNo.values); m = m[~m.index.duplicated()]
tk['pid'] = tk.applNo.map(m); tk = tk[tk.pid.notna()]; tk['pid'] = tk.pid.astype(int)
R = {'tokens_mapped': int(len(tk))}
fp = pd.read_parquet(T4 + r'\cache\s2_fam_pid.parquet')
R['tokens_mapped_vs_s2_fam_pid'] = [int(len(tk)), int(len(fp))]
ev = pd.read_parquet(T4 + r'\R2\runs_final\F22\ev.parquet', columns=['pid', 'fam_n', 'fam_us', 'fam_ep'])
s = tk[tk.yr <= 2021]
raw = s.groupby('pid').size(); norm = s.drop_duplicates(['pid', 'norm']).groupby('pid').size()
ev['fam_n_raw'] = ev.pid.map(raw).fillna(0).values; ev['fam_n_norm'] = ev.pid.map(norm).fillna(0).values
R['repro_max_abs_raw_vs_stored'] = float(np.abs(ev.fam_n_raw - ev.fam_n).max())
R['test_units'] = int(len(ev)); R['test_units_fam_pos'] = int((ev.fam_n > 0).sum())
ch = ev.fam_n_raw != ev.fam_n_norm
R['test_units_changed'] = int(ch.sum()); R['test_units_changed_share'] = float(ch.mean()); R['changed_share_among_fam_pos'] = float(ch[ev.fam_n > 0].mean())
R['ratio_raw_over_norm_among_changed'] = [float(x) for x in (ev.fam_n_raw[ch] / ev.fam_n_norm[ch]).describe()[['mean', '50%', 'max']]]
R['spearman_raw_norm_all'] = float(spearmanr(ev.fam_n_raw, ev.fam_n_norm).statistic)
R['spearman_raw_norm_fam_pos'] = float(spearmanr(ev.fam_n_raw[ev.fam_n > 0], ev.fam_n_norm[ev.fam_n > 0]).statistic)
# 토큰 연도별 이중 표기 비중(한 기록 안 같은 정규화 키의 두 번째 이상 토큰 비율)
s2 = tk.copy(); s2['dup'] = s2.duplicated(['pid', 'norm'])
R['dup_token_share_by_yr'] = {int(k): round(float(v), 4) for k, v in s2.groupby('yr').dup.mean().items() if 2000 <= k <= 2024}
R['dup_token_share_by_cc'] = {k: round(float(v), 4) for k, v in s2.groupby('cc').dup.mean().sort_values(ascending=False).head(8).items()}
jdump(R, 's3f_t4_famn.json'); out(json.dumps(R, ensure_ascii=False))
