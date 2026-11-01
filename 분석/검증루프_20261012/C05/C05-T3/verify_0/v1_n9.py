# -*- coding: utf-8 -*-
"""C05-T3 verify_0: N9 레이블 교차 재구성 — s1과 다른 코드·다른 원천본.
- 제공 KR: meta parquet 대신 공모전데이터\작업데이터\kr_patents.parquet(별도 추출본, 한글 열)
- 조인: duckdb 대신 pandas dict/set 연산
- 반박 시도: (i) ProjectYear=2022 요구 (ii) 출원연도를 NTIS Year로(출원 레코드만) (iii) 등록 경로 제거 (iv) 출원연도 창을 [t, t+2]로 넓힘(정의 민감도, 참고)
  (v) 성과 IprType 특허만 (vi) 2023-12 출원 미공개 보정 상한: 2023-12 결손 비율만큼 양성 추가 시 최악 시나리오(양성 무작위 추가)에서 AUC 범위
"""
import pandas as pd, numpy as np, json, os, sys, io
from sklearn.metrics import roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
B = os.path.join(ROOT, "분석"); SH = os.path.join(B, r"모델경쟁_20260927\_shared")
N9 = os.path.join(B, r"모델경쟁_20260927\N9_정부RnD과제_특허성과예측")
V = os.path.join(B, r"검증루프_20261012\C05\C05-T3\verify_0")
o = {}
kp = pd.read_parquet(os.path.join(ROOT, r"공모전데이터\작업데이터\kr_patents.parquet"))
cols = list(kp.columns)
o['kr_patents_cols'] = cols
an = [c for c in cols if c == '출원번호'][0]; rn = [c for c in cols if c == '등록번호'][0]; ay = [c for c in cols if c == '출원연도'][0]
kp = kp[[an, rn, ay]].copy(); kp.columns = ['appl_no', 'rgst_no', 'ay']
kp['appl_no'] = kp.appl_no.astype(str).str.replace(r'\D', '', regex=True)
kp['rgst_no'] = kp.rgst_no.astype(str).str.replace(r'\D', '', regex=True)
kp['ay'] = pd.to_numeric(kp.ay, errors='coerce')
o['kr_patents_n'] = int(len(kp)); o['kr_patents_uniq_appl'] = int(kp.appl_no.nunique())
o['kr_patents_rgst_len_dist'] = kp.rgst_no.str.len().value_counts().head(5).to_dict()
a2y = dict(zip(kp.appl_no, kp.ay))
# 등록번호 형식 차이 대비: 13자리 및 앞 10자리(10-XXXXXXX) 둘 다 시도
r2a = {}
for a, r in zip(kp.appl_no, kp.rgst_no):
    if r and r != 'nan' and r != 'None':
        r2a[r] = a
        if len(r) == 10:
            r2a[r + '0000'] = a
rp = pd.read_parquet(os.path.join(SH, "ext", "ntis_rpatent_raw.parquet"),
                     columns=['ResultID', 'ProjectID', 'ProjectYear', 'Year', 'RegistType', 'IprType', 'num_digits', 'is_kr'])
rp = rp[rp.is_kr & (rp.num_digits.str.len() == 13)]
rp['appl_from_appl'] = rp.num_digits.map(lambda x: x if x in a2y else None)
rp['appl_from_rgst'] = rp.num_digits.map(r2a.get)
rp['route'] = np.where(rp.appl_from_appl.notna(), 'appl', np.where(rp.appl_from_rgst.notna(), 'rgst', None))
rp['appl'] = rp.appl_from_appl.fillna(rp.appl_from_rgst)
lk = rp[rp.appl.notna()].copy()
lk['ay'] = lk.appl.map(a2y)
o['links_rows'] = int(len(lk)); o['links_patents'] = int(lk.appl.nunique())
units = pd.read_parquet(os.path.join(SH, "ext", "ntis_project_raw.parquet"), columns=['pjt_no', 'year', 'secret_flag'])
units = units[(pd.to_numeric(units.year) == 2022) & (~units.secret_flag.fillna(False).astype(bool))].pjt_no.unique()
pr = pd.read_parquet(os.path.join(N9, "final_preds_r6.parquet"))
o['units_equal_preds'] = bool(set(units) == set(pr.pjt_no))


def ylab(mask):
    s = set(lk[mask].ProjectID)
    return pr.pjt_no.isin(s).astype(int).values


def sc(y):
    a = roc_auc_score(y, pr.p_champ); b = roc_auc_score(y, pr.p_M0_L2)
    k = int(np.ceil(0.01 * len(y))); top = np.argsort(-pr.p_champ.values, kind='stable')[:k]
    return {'pos': int(y.sum()), 'auc': a, 'auc_M0L2': b, 'skill': (a - b) / (1 - b), 'top1': float(y[top].mean()),
            'agree_with_official_y': float((y == pr.y.values).mean())}


base = lk.ay.between(2022, 2023)
o['main'] = sc(ylab(base))
o['refute_i_projectyear2022'] = sc(ylab(base & (pd.to_numeric(lk.ProjectYear, errors='coerce') == 2022)))
o['refute_ii_ntis_year_appl_records'] = sc(ylab((lk.RegistType.str.contains('출원')) & pd.to_numeric(lk.Year, errors='coerce').between(2022, 2023)))
o['refute_iii_no_rgst_route'] = sc(ylab(base & (lk.route == 'appl')))
o['refute_iv_window_t_t2'] = sc(ylab(lk.ay.between(2022, 2024)))
o['refute_v_ipr_patent_only'] = sc(ylab(base & lk.IprType.str.contains('특허')))
# (vi) 2023-12 미공개 보정 최악/최선: 음성 중 무작위가 아니라 점수 하위(최악)·상위(최선)에 양성 추가
y0 = ylab(base)
dec_missing_share = 0.056  # s3: 2023-12 출원 비율 1.497 대 2016–22 평균 1.586 → 약 5.6% 결손(2022 대비 9.1%)
dec_share_window = float(lk[base & (lk.appl.str[3:7] == '2023')].shape[0]) / max(1, lk[base].shape[0])
n_add = int(np.ceil(y0.sum() * 0.091 * (1 / 24) * 1.6))  # 보수적 상한: 결과창 출원의 약 1/24×1.6(12월 가중)×9.1%
neg = np.where(y0 == 0)[0]; order = neg[np.argsort(pr.p_champ.values[neg])]
yw = y0.copy(); yw[order[:n_add]] = 1
yb = y0.copy(); yb[order[-n_add:]] = 1
o['refute_vi_dec2023_missing'] = {'n_add_upper': n_add, 'worst': sc(yw), 'best': sc(yb)}
o['official_json'] = {k: json.load(open(os.path.join(N9, 'final_result_r6.json'), encoding='utf-8'))[k] for k in ['pos', 'auc', 'auc_M0L2', 'skill', 'top1']}
json.dump(o, open(os.path.join(V, 'v1_n9_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in o.items() if k != 'kr_patents_cols'}, ensure_ascii=False, indent=1, default=str))
