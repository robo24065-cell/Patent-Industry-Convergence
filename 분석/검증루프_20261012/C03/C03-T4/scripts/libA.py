# C03-A 공용: 개발 원점 사건 구성(2025 이전 자료만)
import numpy as np, pandas as pd
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
W = B + r'\분석\검증루프_20261012\C03\C03-T4'
C = W + r'\cache'
C2 = B + r'\분석\검증루프_20261012\C02\C02-T4\cache'
DEV = list(range(2012, 2024))


def load(meth=('PUB',)):
    docs = pd.read_parquet(C2 + r'\docs.parquet')
    docs = docs[docs.cd.str[0] == '1'][['cd', 'appl_no', 'ksic', 'vis']].copy()
    docs['ksic'] = docs.ksic.astype(str)
    L = pd.read_parquet(C + r'\A_links_lt2025.parquet')
    assert L.pub_date.max() < '20250101'
    L = L[L.meth.isin(meth)]
    L = L[['cc', 'appl_no', 'pub_date', 'kr_appl']].drop_duplicates().rename(columns={'appl_no': 'fno'})
    return docs, L


def build(t0, docs, L, doc_lab=None):
    """doc_lab: docs와 같은 index의 라벨 Series(None이면 제공 라벨). 반환: stats, positives DataFrame(cd,k), risk set size."""
    T = f'{t0}1231'; ws = f'{t0 + 1}0101'; we = f'{t0 + 1}0630'
    lab = docs.ksic if doc_lab is None else doc_lab
    dd = docs.assign(k=lab)
    vis_T = dd[dd.vis <= T]
    cnt = vis_T.groupby('cd').appl_no.nunique()
    Fset = set(cnt[cnt >= 3].index)
    dom = vis_T[vis_T.cd.isin(Fset)].dropna(subset=['k'])[['cd', 'k']].drop_duplicates()
    # 해외 공개 ↔ KR 문서(기업, 라벨)
    j = L.merge(dd[['cd', 'appl_no', 'k', 'vis']].dropna(subset=['k']), left_on='kr_appl', right_on='appl_no')
    j = j[j.cd.isin(Fset)]
    pre = j[(j.pub_date <= T) & (j.vis <= T)][['cd', 'k']].drop_duplicates()
    risk = dom.merge(pre, on=['cd', 'k'], how='left', indicator=True)
    risk = risk[risk._merge == 'left_only'][['cd', 'k']]
    win = j[(j.pub_date >= ws) & (j.pub_date <= we) & (j.vis <= we)]
    ev = win.merge(risk, on=['cd', 'k'])
    pos = ev.sort_values('pub_date').groupby(['cd', 'k']).agg(pub=('pub_date', 'first'), n_foreign=('fno', 'nunique'),
                                                               cc_first=('cc', 'first')).reset_index()
    pos['t0'] = t0
    st = dict(t0=t0, eligible_firms=len(Fset), dom_pairs=len(dom), already_abroad_pairs=len(pre), risk_pairs=len(risk),
              win_links_eligible=len(win), win_foreign_docs=int(win.cc.str.cat(win.fno).nunique()) if len(win) else 0,
              pos=len(pos), pos_firms=int(pos.cd.nunique()), base_rate=round(len(pos) / max(len(risk), 1), 5),
              firms_with_any_foreign_in_window=int(win.cd.nunique()))
    return st, pos, risk
