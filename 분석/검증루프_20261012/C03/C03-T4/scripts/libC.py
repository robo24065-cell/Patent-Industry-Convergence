# C03-C 공용: 기업 × KSIC 몫 상승(출원공개일 기준). 개발 자료(공개 < 2025-01-01)만 적재.
import numpy as np, pandas as pd
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
W = B + r'\분석\검증루프_20261012\C03\C03-T4'
C = W + r'\cache'
C2 = B + r'\분석\검증루프_20261012\C02\C02-T4\cache'
DEV = list(range(2012, 2024))
KSIC50 = ['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924',
          '25932', '26111', '26112', '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323',
          '26422', '26429', '26519', '26529', '27199', '27211', '27212', '27301', '28111', '28122', '28511', '28902',
          '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280', '30393', '31202', '31312',
          '33301', '33309']


def load():
    kp = pd.read_parquet(B + r'\공모전데이터\작업데이터\kr_patents.parquet', columns=['출원번호', '공개일자', '세세분류코드'])
    kp.columns = ['appl_no', 'p', 'ksic']
    kp = kp[kp.p.fillna('').str.len() == 8]
    kp = kp[kp.p < '20250101'].copy()
    assert kp.p.max() < '20250101'
    kp['ksic'] = kp.ksic.astype(str)
    docs = pd.read_parquet(C2 + r'\docs.parquet', columns=['cd', 'appl_no'])
    docs = docs[docs.cd.str[0] == '1'].drop_duplicates()
    d = docs.merge(kp, on='appl_no')
    return d  # cd, appl_no, p, ksic(제공)


def build(t0, d, lab=None, thr=0.20):
    """lab: d와 같은 index 라벨 Series(None=제공). 반환 stats, pos(cd,k), risk 크기"""
    k = d.ksic if lab is None else lab
    x = d.assign(k=k).dropna(subset=['k'])
    bs, be = f'{t0-2}0101', f'{t0}1231'; ws, we = f'{t0+1}0101', f'{t0+1}0630'
    b = x[(x.p >= bs) & (x.p <= be)]; w = x[(x.p >= ws) & (x.p <= we)]
    nb = b.groupby('cd').appl_no.nunique(); nw = w.groupby('cd').appl_no.nunique()
    F = set(nb[nb >= 3].index) & set(nw[nw >= 3].index)
    b = b[b.cd.isin(F)]; w = w[w.cd.isin(F)]
    sb = (b.groupby(['cd', 'k']).appl_no.nunique() / b.cd.map(nb).groupby([b.cd, b.k]).first()).rename('sb')
    sw = (w.groupby(['cd', 'k']).appl_no.nunique() / w.cd.map(nw).groupby([w.cd, w.k]).first()).rename('sw')
    r = sb.to_frame().join(sw, how='left').fillna({'sw': 0.0})
    r['y'] = (r.sw - r.sb) >= thr - 1e-12
    pos = r[r.y].reset_index()[['cd', 'k', 'sb', 'sw']]; pos['t0'] = t0
    st = dict(t0=t0, eligible_firms=len(F), risk_pairs=len(r), pos=int(r.y.sum()), pos_firms=int(pos.cd.nunique()),
              base_rate=round(float(r.y.mean()), 4) if len(r) else None)
    return st, pos, r
