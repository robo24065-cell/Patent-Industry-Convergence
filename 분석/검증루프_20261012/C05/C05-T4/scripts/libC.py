# C05-T4 공용 함수. C04-T4 libA.py(읽기만)를 바탕으로 다시 쓴 것: form=='1' 필터(E-C04T4-D1), 잠금 S1 로드 금지, 로드 기록.
import os, re, json, calendar, hashlib
import numpy as np, pandas as pd
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
MC = os.path.join(B, '분석', '모델경쟁_20260927')
EXT = os.path.join(MC, '_shared', 'ext')
W = os.path.join(B, '분석', '검증루프_20261012', 'C05', 'C05-T4')
C = os.path.join(W, 'cache')
DEV = ['201512', '201612', '201712', '201812', '201912', '202012', '202112', '202212', '202312']
EVAL = ['201812', '201912', '202012', '202112', '202212', '202312']
LOCK = '202412'
REF = {'201512': '20151130', '201612': '20161130', '201712': '20171130', '201812': '20181130', '201912': '20191130',
       '202012': '20201231', '202112': '20211231', '202212': '20221231', '202312': '20231231', '202412': '20241231'}
PREV = {'201612': '201512', '201712': '201612', '201812': '201712', '201912': '201812', '202012': '201912',
        '202112': '202012', '202212': '202112', '202312': '202212', '202412': '202312'}
LOCKED_S1 = {'202512', '202608'}
DEV_PUB_MAX = '20240630'
CORP_TC = {str(x) for x in range(81, 89)}
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
LOADLOG = os.path.join(C, 'nps_load_log.txt')


def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)


def add_months(ymd, m):
    y, mo, d = int(ymd[:4]), int(ymd[4:6]), int(ymd[6:])
    mo += m
    y += (mo - 1) // 12; mo = (mo - 1) % 12 + 1
    d = min(d, calendar.monthrange(y, mo)[1])
    return f'{y:04d}{mo:02d}{d:02d}'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def load_ca():
    rows = []
    with open(os.path.join(B, '공모전데이터', '외부데이터', 'Corporate_20260720', 'CORP_APPLICANT.txt'), encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\r\n').split('¶')
            if len(p) >= 5:
                rows.append((p[0].strip(), p[1].strip(), p[3].strip(), p[4].strip()))
    ca = pd.DataFrame(rows, columns=['applicantCd', 'aname', 'corpno', 'bizno'])
    ca['bizno'] = ca.bizno.str.replace(r'\D', '', regex=True)
    ca['nname'] = ca.aname.map(norm)
    ca['has_biz'] = ca.bizno.str.len() == 10
    ca['biz6'] = np.where(ca.has_biz, ca.bizno.str[:6], None)
    ca['corp_tc'] = ca.has_biz & ca.bizno.str[3:5].isin(CORP_TC)
    return ca


def name_map(ca):
    g = ca[ca.nname.str.len() >= 2].groupby('nname')
    st = g.agg(n=('applicantCd', 'size'), nb=('has_biz', 'sum'), nb6=('biz6', 'nunique'), b6=('biz6', 'first'), ctc=('corp_tc', 'max'))
    ok = st[(st.nb == st.n) & (st.nb6 == 1) & st.ctc]
    return ok.b6.to_dict(), st


def load_docs():
    k = pd.read_parquet(os.path.join(B, '공모전데이터', '작업데이터', 'kr_patents.parquet'),
                        columns=['출원번호', '출원연도', '공개일자', '등록일자', '출원인명', '세세분류코드', '우선권국가'])
    k.columns = ['appl_no', 'ay', 'pub', 'rg', 'aname', 'ksic', 'prio']
    k['pub'] = k.pub.fillna('').astype(str).str.strip(); k['rg'] = k.rg.fillna('').astype(str).str.strip()
    k.loc[k.pub.str.len() != 8, 'pub'] = None; k.loc[k.rg.str.len() != 8, 'rg'] = None
    k['vis'] = k.pub
    m = k.rg.notna() & (k.vis.isna() | (k.rg.fillna('99999999') < k.vis.fillna('99999999')))
    k.loc[m, 'vis'] = k.loc[m, 'rg']
    k['nname'] = k.aname.map(norm)
    k['ksic'] = k.ksic.astype(str)
    p = k.prio.fillna('')
    k['dom_prio'] = (p == '') | p.str.split('|').map(lambda L: all(x == 'KR' for x in L if x))
    k['kor_corp'] = k.aname.fillna('').str.contains(r'주식회사|\(주\)|㈜|유한회사').values & k.dom_prio.values
    return k[['appl_no', 'ay', 'pub', 'rg', 'vis', 'nname', 'ksic', 'dom_prio', 'kor_corp']]


def load_nps(s, cols=('name', 'biz6', 'stat', 'form', 'ldong', 'indcd', 'adpt_dt', 'n_emp', 'n_new', 'n_loss'), tag=''):
    assert s not in LOCKED_S1, 'locked snapshot must not be loaded'
    assert s in REF, s
    cols = list(dict.fromkeys(list(cols) + ['form', 'stat', 'biz6', 'name']))
    d = pd.read_parquet(os.path.join(EXT, f'firm_nps_wkpl_{s}.parquet'), columns=cols)
    with open(LOADLOG, 'a', encoding='utf-8') as f:
        f.write(f'{tag}\t{s}\n')
    nform = d.form.astype(str).value_counts().to_dict()
    d = d[(d.stat.astype(str) == '1') & (d.form.astype(str) == '1') & d.biz6.astype(str).str[3:5].isin(CORP_TC)].copy()
    d['key'] = d.biz6 + '|' + d.name.map(norm)
    return d.reset_index(drop=True), nform


def firm_table(d, ref):
    d = d.sort_values('n_emp', ascending=False)
    d['ay0'] = pd.to_numeric(d.adpt_dt.astype(str).str[:4], errors='coerce')
    g = d.groupby('key', sort=False)
    f = g.agg(E=('n_emp', 'sum'), n_wkpl=('n_emp', 'size'), n_new=('n_new', 'sum'), n_loss=('n_loss', 'sum'),
              indcd=('indcd', 'first'), ldong=('ldong', 'first'), ay0=('ay0', 'min'))
    ic = f.indcd.fillna('').astype(str).str.strip()
    ic = ic.where(ic.str.fullmatch(r'\d{1,6}') & ~ic.isin(['', '000000']), None)
    f['ind6'] = ic.map(lambda x: x.zfill(6) if isinstance(x, str) else 'NA')
    f['ind4'] = f.ind6.map(lambda x: x[:4] if x != 'NA' else 'NA')
    f['ind2'] = f.ind6.map(lambda x: x[:2] if x != 'NA' else 'NA')
    ld = f.ldong.fillna('').astype(str).str.strip()
    f['sido'] = ld.str[:2].where(ld.str.len() >= 5, 'NA')
    f['sgg'] = ld.str[:5].where(ld.str.len() >= 5, 'NA')
    f['age'] = (int(ref[:4]) - f.ay0).clip(0, 80)
    return f.drop(columns=['indcd', 'ldong'])


def load_ntis():
    nt = pd.read_parquet(os.path.join(EXT, 'ntis_project_raw.parquet'), columns=['year', 'corp_rgst_no', 'research_agency'])
    nt['bz'] = nt.corp_rgst_no.fillna('').str.replace(r'\D', '', regex=True)
    nt = nt[nt.bz.str.len() == 10].copy()
    nt['year'] = pd.to_numeric(nt.year, errors='coerce')
    nt['key6'] = nt.bz.str[:6] + '|' + nt.research_agency.map(norm)
    nt['is_tc'] = nt.bz.str[3:5].isin(CORP_TC)
    return nt


def auc(y, s, w=None):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    if w is None:
        w = np.ones(len(y))
    o = np.argsort(s, kind='mergesort'); s, y, w = s[o], y[o], w[o]
    # 같은 점수 묶음 처리
    _, start = np.unique(s, return_index=True)
    w1 = np.where(y, w, 0.0); w0 = np.where(~y, w, 0.0)
    c0 = np.cumsum(w0)
    grp = np.repeat(np.arange(len(start)), np.diff(np.append(start, len(s))))
    g0 = np.bincount(grp, weights=w0); g1 = np.bincount(grp, weights=w1)
    below = np.concatenate([[0.0], np.cumsum(g0)[:-1]])
    num = np.sum(g1 * (below + 0.5 * g0))
    W1, W0 = w1.sum(), w0.sum()
    return float(num / (W1 * W0)) if W1 > 0 and W0 > 0 else np.nan


def ap_top(y, s, w=None, k=1000):
    """가중 평균정밀도(PR-AUC)와 상위 k(가중 누적) 정밀도. 점수 내림차순, 동점은 안정 정렬."""
    y = np.asarray(y).astype(float); s = np.asarray(s, float)
    if w is None:
        w = np.ones(len(y))
    o = np.argsort(-s, kind='mergesort'); y, w = y[o], w[o]
    tp = np.cumsum(w * y); n = np.cumsum(w)
    P = tp[-1]
    prec = tp / n
    ap = float(np.sum(prec * w * y) / P) if P > 0 else np.nan
    i = np.searchsorted(n, k, side='left'); i = min(i, len(n) - 1)
    top = float(tp[i] / n[i])
    return ap, top
