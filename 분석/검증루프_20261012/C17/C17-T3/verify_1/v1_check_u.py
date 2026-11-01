# verify_1: 공식 최종 예측 파일로 IPC 2021.01 부재 메인그룹 셀 수를 독립 재계산
import os, re, zipfile, sys, json
import pandas as pd
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
D = os.path.join(ROOT, "검증루프_20261012", "C05", "C05-T1", "download")
R3 = os.path.join(ROOT, "모델경쟁_20260927", "T3_기술셀부상예측", "R3")
def mgset(v):
    z = zipfile.ZipFile(os.path.join(D, f'EN_ipc_title_list_{v}.zip')); M=set(); S=set()
    for n in z.namelist():
        if not n.endswith('.txt'): continue
        for line in z.read(n).decode('utf-8','replace').splitlines():
            c = line.split('\t')[0].strip()
            m = re.match(r'^([A-H]\d\d[A-Z])(\d{4})(\d{6})$', c)
            if m and m.group(3)=='000000': M.add(m.group(1)+' '+str(int(m.group(2))))
            if re.match(r'^[A-H]\d\d[A-Z]$', c): S.add(c)
    return M,S
M21,S21 = mgset('20210101')
later=set(); 
for v in ['20220101','20230101','20240101','20250101','20260101']:
    later |= mgset(v)[0]
P = pd.read_parquet(os.path.join(R3,'preds','FINAL_R3_t0_2021.parquet'))
print(P.columns.tolist(), len(P))
cellcol = 'cell' if 'cell' in P.columns else P.columns[0]
P['mg'] = P[cellcol].str.split('|').str[1]
P['sub'] = P.mg.str[:4]
notin = ~P.mg.isin(M21)
print('final cells', len(P), 'not in 2021.01:', notin.sum(), 'and in later:', (notin & P.mg.isin(later)).sum(), 'in neither:', (notin & ~P.mg.isin(later)).sum())
ycol = [c for c in P.columns if c in ('yb','y_true','label')]
print('ycols', ycol)
if ycol:
    y=ycol[0]; print('pos total', P[y].sum(), 'pos in U', P.loc[notin & P.mg.isin(later), y].sum())
U = P[notin & P.mg.isin(later)]
print('new subclass cells', (~U['sub'].isin(S21)).sum(), 'new mg in old subclass', U['sub'].isin(S21).sum())
print(U['sub'].value_counts().to_dict())
u0 = pd.read_csv(os.path.join(ROOT,"검증루프_20261012","C17","C17-T3","cache","u_cells_final.csv"), encoding='utf-8-sig')
s1=set(U[cellcol]); s0=set(u0.loc[u0.U, 'cell'])
print('U match with C17 u_cells:', s1==s0, len(s1), len(s0), 'T48', u0.T48.sum(), 'T48 subset of U', set(u0.loc[u0.T48,'cell'])<=s1)
print('neither list', sorted(P.loc[notin & ~P.mg.isin(later), cellcol])[:20])
