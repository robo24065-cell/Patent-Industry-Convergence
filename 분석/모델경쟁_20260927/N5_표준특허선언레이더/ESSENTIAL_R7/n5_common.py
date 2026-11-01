# -*- coding: utf-8 -*-
"""R7 N5 필수성 검증 공용 모듈. 동결 n5lib.py 는 import 만 한다(N5/cache 의 26429 캐시를 읽기만; 새 캐시는 ESSENTIAL_R7/cache 에만).
모집단 정의
  orig  : 제공 KSIC 26429 라벨 KR 출원 (동결 단위)
  conc  : 공식 IPC-KSIC 연계표(2013, 9차) argmax 배정 코드 C2640(통신 및 방송 장비 제조업; 26429 의 10→9차 대응 26429 의 상위 코드) KR 출원
  concm : 연계표 '관대한 다중 배정'(점유율 ≥0.10 코드 전부)에 C2640 이 들어간 KR 출원
  (연계표 배정 표는 T3_기술셀부상예측/ESSENTIAL_R7/rec_conc*.parquet — 같은 R7 검증에서 만든 산출물을 읽기만)
"""
import os, sys, re, sqlite3
import numpy as np, pandas as pd, duckdb
HERE = os.path.dirname(os.path.abspath(__file__)); N5 = os.path.dirname(HERE)
sys.path.insert(0, N5); sys.dont_write_bytecode = True
import n5lib as N
MC = N.MC; EXT = N.EXT; BASE = N.BASE
T3E = os.path.join(MC, 'T3_기술셀부상예측', 'ESSENTIAL_R7')
CACHE = os.path.join(HERE, 'cache'); os.makedirs(CACHE, exist_ok=True)
CODE = 'C2640'
META = os.path.join(MC, '_shared', 'meta', '*.parquet').replace('\\', '/')


def rc_path(pop):
    return os.path.join(T3E, 'rec_conc.parquet' if pop == 'conc' else 'rec_conc_multi.parquet').replace('\\', '/')


def kr_codes(pop):
    """KR 출원별 연계표 배정 여부(code == C2640)."""
    return set(duckdb.sql(f"select distinct appl_no from read_parquet('{rc_path(pop)}') where cc='KR' and code='{CODE}'").df().appl_no.astype(str))


def load_kr_pop(pop):
    """n5lib.load_kr 와 같은 변환을 연계표 모집단에 적용(필터만 다름). 캐시: ESSENTIAL_R7/cache/kr_{pop}.parquet"""
    p = os.path.join(CACHE, f'kr_{pop}.parquet')
    if os.path.exists(p):
        return pd.read_parquet(p)
    L = pd.read_parquet(os.path.join(EXT, 'trade_etsi_sep_link.parquet'))
    L['refl'] = L.first_refl.dt.strftime('%Y%m%d').astype(float)
    Lkr = L[L.cc == 'KR'].groupby('appl_no').refl.min()
    k = pd.read_parquet(os.path.join(BASE, r'공모전데이터\작업데이터\kr_patents.parquet'),
                        columns=['출원번호', '출원연도', '등록번호', '공개번호', '공개일자', '등록일자', '출원인명', '메인IPC', '세세분류코드', '발명명칭', '우선권국가'])
    k.columns = ['appl_no', 'ay', 'rg', 'pb', 'pub', 'rgd', 'names', 'ipc', 'ksic', 'title', 'prio_cc']
    k['appl_no'] = k.appl_no.astype(str)
    k = k[k.appl_no.isin(kr_codes(pop))].copy()
    for c in ['pub', 'rgd']:
        k[c] = pd.to_numeric(k[c].astype(str).str.replace(r'\D', '', regex=True).str[:8], errors='coerce')
    k['known'] = k[['pub', 'rgd']].min(axis=1)
    k['app'] = k.names.fillna('').str.split('|').str[0].map(N.norm)
    k['mg'] = k.ipc.fillna('').str.split('/').str[0].str.replace(' ', '')
    k['sg'] = k.ipc.fillna('').str.replace(' ', '')
    k['refl'] = k.appl_no.map(Lkr)
    k['pct'] = (k.appl_no.str[6] == '7').astype(float)
    pc = k.prio_cc.fillna('')
    k['fprio'] = pc.map(lambda s: float(any(x and x != 'KR' for x in s.split('|'))))
    c = sqlite3.connect(f"file:{os.path.join(BASE, '수집파이프라인', 'registry_full.sqlite')}?mode=ro", uri=True)
    cl = pd.read_sql("select applNo appl_no, claimCount from patent", c); c.close()
    cl = cl[cl.appl_no.isin(set(k.appl_no))]
    k = k.merge(cl.drop_duplicates('appl_no').assign(claimCount=lambda d: pd.to_numeric(d.claimCount, errors='coerce')), on='appl_no', how='left')
    k = k.drop(columns=['prio_cc']); k['ksic'] = k.ksic.astype(str)
    k.to_parquet(p, index=False)
    return k


def load_family_pop(k, fpop, tag):
    """n5lib.load_family 와 같은 토큰 규칙. 해외(US·EP·JP) 레코드 필터만 fpop 으로: 'orig' = 제공 ksic 26429, 'conc'/'concm' = 연계표 C2640."""
    p = os.path.join(CACHE, f'fam_{tag}.parquet')
    if os.path.exists(p):
        return pd.read_parquet(p)
    L = pd.read_parquet(os.path.join(EXT, 'trade_etsi_sep_link.parquet'))
    L['refl'] = L.first_refl.dt.strftime('%Y%m%d').astype(float)
    Lfo = L[L.cc.isin(['US', 'EP', 'JP'])].copy()
    Lfo['fkey'] = Lfo.cc + Lfo.appl_no.astype(str).str.split('.').str[0]
    Lfo = Lfo.groupby('fkey').refl.min()
    tok = {}
    for c, f in [('appl_no', lambda s: 'KR' + s), ('rg', lambda s: 'KR' + s.str[:9]), ('pb', lambda s: 'KR' + s.str[2:])]:
        s = k[c].fillna('').astype(str).str.replace(r'\D', '', regex=True); ok = s.str.len() >= 9
        tok.update(dict(zip(f(s[ok]), k.appl_no[ok])))
    if fpop == 'orig':
        fo = duckdb.sql(f"select distinct cc, appl_no, pub_date, family from read_parquet('{META}') where cc in ('US','EP','JP') and ksic='26429' and family like '%KR%'").df()
    else:
        fo = duckdb.sql(f"""select distinct m.cc, m.appl_no, m.pub_date, m.family from read_parquet('{META}') m
                           join (select distinct cc, appl_no from read_parquet('{rc_path(fpop)}') where code='{CODE}' and cc in ('US','EP','JP')) r
                           on m.cc = r.cc and m.appl_no = r.appl_no where m.family like '%KR%'""").df()
    fo['fkey'] = fo.cc + fo.appl_no.astype(str); fo['fknown'] = pd.to_numeric(fo.pub_date, errors='coerce'); fo['frefl'] = fo.fkey.map(Lfo)
    fo['tok'] = fo.family.fillna('').str.split('|')
    ex = fo[['fkey', 'cc', 'fknown', 'frefl', 'tok']].explode('tok')
    ex = ex[ex.tok.str.startswith('KR', na=False)]; ex['kr'] = ex.tok.map(tok)
    ex = ex.dropna(subset=['kr']).drop_duplicates(['fkey', 'kr']).drop(columns=['tok'])
    ex.to_parquet(p, index=False)
    return ex


def load_ext_pop(k, E0, tag):
    """단위 수준 외부데이터(RoS·NTIS)를 출원번호로 다시 붙인다(동결본은 ksic=26429 필터; KR 출원은 한 KSIC 에만 있으므로 출원번호 필터와 같음)."""
    p = os.path.join(CACHE, f'ext_{tag}.pkl')
    if os.path.exists(p):
        return pd.read_pickle(p)
    con = duckdb.connect(); con.register('u', pd.DataFrame({'appl_no': sorted(set(k.appl_no))}))
    ros = con.sql(f"""select f.appl_no, p.issue_year, p.n_cit from read_parquet('{os.path.join(EXT, 'r3ros_family_us_pairs.parquet').replace(chr(92), '/')}') f
        join read_parquet('{os.path.join(EXT, 'r3ros_patent_sci.parquet').replace(chr(92), '/')}') p on f.us_pn = p.us_pn
        join u on u.appl_no = f.appl_no where f.cc='KR'""").df()
    nt = pd.read_parquet(os.path.join(EXT, 'ntis_patent_link.parquet'), columns=['cc', 'appl_no', 'ksic', 'Year', 'perform_grp'])
    nt = nt[(nt.cc == 'KR') & nt.appl_no.astype(str).isin(set(k.appl_no))].copy(); nt['Year'] = pd.to_numeric(nt.Year, errors='coerce')
    E = dict(E0); E['ros'] = ros; E['ntis'] = nt[['appl_no', 'Year', 'perform_grp']]
    pd.to_pickle(E, p)
    return E


def hybrid_panel(t0, ku, ks, ex, E, mode='dev'):
    """챔피언 피처만, 단위는 ku(목표 모집단), 모집단 통계(선언률·제목 풀·어휘)는 ks(통계 모집단)에서. n5lib.build_panel 과 같은 식.
    ku is ks 이면 동결 build_panel 의 챔피언 피처와 같아야 한다(검증: n5_panels.py)."""
    if mode == 'dev': assert t0 + 2 < N.F_YEAR
    end = t0 * 10000 + 1231
    kn = ks[ks.known <= end]; dec = kn.refl <= end; prior = dec.mean()
    ga = kn.assign(d=dec).groupby('app').agg(n=('d', 'size'), d=('d', 'sum'))
    gm = kn.assign(d=dec).groupby('mg').agg(n=('d', 'size'), d=('d', 'sum'))
    gs = kn.assign(d=dec).groupby('sg').agg(n=('d', 'size'), d=('d', 'sum'))
    rec = kn[(kn.refl > (t0 - 2) * 10000 + 1231) & (kn.refl <= end)]
    ku_ = ku[ku.known <= end]
    U = ku_[ku_.ay.between(t0 - 5, t0) & ~(ku_.refl <= end)].copy()
    U['app_age'] = t0 - U.ay; U['reg_t0'] = (U.rgd <= end).astype(float); U['claims_t0'] = U.claimCount.where(U.rgd <= end)
    U['app_rate'] = N.sm_rate(U.app.map(ga.d).fillna(0), U.app.map(ga.n).fillna(0), prior)
    U['app_logn'] = np.log1p(U.app.map(ga.n).fillna(0))
    U['mg_rate'] = N.sm_rate(U.mg.map(gm.d).fillna(0), U.mg.map(gm.n).fillna(0), prior)
    U['mg_logn'] = np.log1p(U.mg.map(gm.n).fillna(0))
    e = ex[ex.fknown <= end]
    for cc in ['US', 'EP', 'JP']:
        U['fam_' + cc] = U.appl_no.isin(set(e.loc[e.cc == cc, 'kr'])).astype(float)
    U['fam_n'] = U.appl_no.map(e.groupby('kr').fkey.nunique()).fillna(0)
    r = E['ros'][E['ros'].issue_year <= t0]
    rg = r.groupby('appl_no').agg(n_us=('n_cit', 'size'), n_cit=('n_cit', 'sum'), n_sci=('n_cit', lambda x: (x > 0).sum()))
    U['ros_us_issued'] = np.log1p(U.appl_no.map(rg.n_us).fillna(0)); U['ros_ncit'] = np.log1p(U.appl_no.map(rg.n_cit).fillna(0))
    U['ros_sci'] = (U.appl_no.map(rg.n_sci).fillna(0) > 0).astype(float)
    nt = E['ntis'][E['ntis'].Year <= t0]
    U['ntis_gov'] = U.appl_no.isin(set(nt.appl_no)).astype(float)
    U['ntis_gri'] = U.appl_no.isin(set(nt.loc[nt.perform_grp.isin(['출연연', '국공립연']), 'appl_no'])).astype(float)
    U['ntis_firm'] = U.appl_no.isin(set(nt.loc[nt.perform_grp == '기업', 'appl_no'])).astype(float)
    U['ntis_univ'] = U.appl_no.isin(set(nt.loc[nt.perform_grp == '대학', 'appl_no'])).astype(float)
    U['app_rec_rate'] = N.sm_rate(U.app.map(rec.groupby('app').size()).fillna(0), U.app.map(ga.n).fillna(0), len(rec) / max(1, len(kn)))
    U['sg_rate'] = N.sm_rate(U.sg.map(gs.d).fillna(0), U.sg.map(gs.n).fillna(0), U.mg_rate)
    U['sg_logn'] = np.log1p(U.sg.map(gs.n).fillna(0))
    from sklearn.feature_extraction.text import TfidfVectorizer
    pool = kn[(kn.refl > (t0 - 3) * 10000 + 1231) & (kn.refl <= end)]
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3), min_df=3, sublinear_tf=True, dtype=np.float32)
    vec.fit(kn.title.fillna(''))
    P = vec.transform(pool.title.fillna('')); X = vec.transform(U.title.fillna(''))
    smax = np.zeros(X.shape[0], np.float32); stop = np.zeros(X.shape[0], np.float32)
    if P.shape[0] > 0:
        kk = min(10, P.shape[0])
        for i in range(0, X.shape[0], 2000):
            S = (X[i:i + 2000] @ P.T).toarray(); smax[i:i + 2000] = S.max(1); stop[i:i + 2000] = -np.sort(-S, axis=1)[:, :kk].mean(1)
    U['title_sim_max'] = smax; U['title_sim_top10'] = stop
    U['y'] = ((U.refl > end) & (U.refl <= (t0 + 2) * 10000 + 1231)).astype(int) if mode == 'dev' else -1
    U['t0'] = t0
    keep = ['appl_no', 'app', 'mg', 'ay', 'ksic', 't0', 'y'] + list(dict.fromkeys(N.FEATS['M0'] + N.FEATS['M1'] + N.FEATS['ros'] + N.FEATS['ntis'] + ['app_rec_rate', 'sg_rate', 'sg_logn', 'title_sim_max', 'title_sim_top10', 'pct', 'fprio']))
    return U[keep].reset_index(drop=True)
