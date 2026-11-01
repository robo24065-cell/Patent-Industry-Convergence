# 산업 '최근 프런티어'·'최근 진입자 첫 특허' 중심벡터와 법인 범위밖 제목의 유사도 (기준시점 연말까지 등록된 범위안 특허만)
import pandas as pd, numpy as np, time
from lib import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
t0 = time.time()
D = load(); oos = D['oos']; K = D['K']; kidx = D['kidx']
kr = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['rgst_no', 'title', 'ksic', 'rgst_date', 'ay'])
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=5, max_features=300000, sublinear_tf=True, dtype=np.float32)
vec.fit(kr[kr.rgst_date <= '20091231'].title.fillna(''))
Xo = normalize(vec.transform(oos.title.fillna(''))); print('oos vec', Xo.shape, round(time.time() - t0), flush=True)
ins = D['ins']
for cut in (2002, 2003, 2004, 2009):
    known = kr[(kr.rgst_date <= f'{cut}1231')]
    rec = known[(known.ay > cut - 5) & (known.ay <= cut)]
    Xr = normalize(vec.transform(rec.title.fillna('')))
    C1 = np.zeros((50, Xr.shape[1]), np.float32)
    for k, idx in rec.groupby('ksic').indices.items(): C1[kidx[k]] = np.asarray(Xr[idx].mean(0)).ravel()
    # 최근 진입자 첫 특허: 기준시점까지 알려진(등록) 범위안에서 법인×KSIC 첫 출원이 (cut-5,cut]
    ik = ins[ins.rgst_date <= f'{cut}1231'].sort_values('appl_date')
    first = ik.drop_duplicates(['applicant_cd', 'ksic'])
    first = first[(first.ay > cut - 5) & (first.ay <= cut)]
    Xe = normalize(vec.transform(first.title.fillna('')))
    C2 = np.zeros((50, Xe.shape[1]), np.float32)
    for k, idx in first.groupby('ksic').indices.items(): C2[kidx[k]] = np.asarray(Xe[idx].mean(0)).ravel()
    C1 = normalize(C1); C2 = normalize(C2)
    sel = (oos.ay <= cut).values
    S1 = (Xo[sel] @ C1.T).astype(np.float32); S2 = (Xo[sel] @ C2.T).astype(np.float32)
    np.save(CACHE/f'sim_rec_c{cut}.npy', S1.astype(np.float16)); np.save(CACHE/f'sim_ent_c{cut}.npy', S2.astype(np.float16))
    np.save(CACHE/f'sim_rows_c{cut}.npy', np.nonzero(sel)[0])
    print(cut, 'recent n', len(rec), 'entrant-first n', len(first), round(time.time() - t0), flush=True)
