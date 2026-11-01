# C-D 공용 함수(개발 원점 전용). 입력은 s1 cache(2025 이전 owner만).
import numpy as np, pandas as pd
W = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T4'
C = W + r'\cache'
MC = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927'
KSIC50 = ['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924',
          '25932', '26111', '26112', '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323',
          '26422', '26429', '26519', '26529', '27199', '27211', '27212', '27301', '28111', '28122', '28511', '28902',
          '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280', '30393', '31202', '31312',
          '33301', '33309']
DEV = list(range(2012, 2023))
STOP2 = {'한국', '대한', '코리', '서울', '부산', '대구', '인천', '광주', '대전', '울산', '경기', '강원', '충청', '충북', '충남', '전라', '전북', '전남',
         '경상', '경북', '경남', '제주', '세종', '국제', '글로', '동양', '동아', '에스', '케이', '제이', '엠에', '아이', '디에', '비에', '씨에', '엘에',
         '티에', '유니', '인터', '테크', '하이', '뉴테', '그린', '미래', '세계', '국민', '중앙', '산학', '학교', '재단', '사단', '대학', '연구', '에이',
         '이에', '오에', '알에', '피에', '지에', '엔에', '에프', '더블', '스마', '바이', '나노', '메디', '엔지', '시스', '디지', '일렉', '솔루', '파워',
         '대성', '동일', '한일', '신한', '성원', '우리', '제일', '삼일', '일진', '태양', '신성', '대원', '동원', '세원', '진성', '영진', '대진'}


def load():
    own = pd.read_parquet(C + r'\own_lt2025.parquet')
    pat = pd.read_parquet(C + r'\pat.parquet', columns=['rgst_no', 'ksic', 'appl_no', 'rgstDate'])
    app = pd.read_parquet(C + r'\app_reg.parquet', columns=['rgst_no', 'cd', 'nn', 'vis'])
    docs = pd.read_parquet(C + r'\docs.parquet')
    own = own.merge(pat[['rgst_no', 'ksic', 'appl_no', 'rgstDate']], on='rgst_no', how='left')
    # 일괄: 인정 사유 행, 같은 날 같은 정규화 매입자 ≥10
    a = own[own.acc]
    g = a.groupby(['nn', 'd']).size()
    own['bulk'] = False
    idx = own.acc
    own.loc[idx, 'bulk'] = own.loc[idx].set_index(['nn', 'd']).index.map(g).values >= 10
    orig = app.groupby('rgst_no').cd.agg(set).to_dict()
    return own, pat, app, docs, orig


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n


def seller_info(own):
    """인정 사유 행마다 직전 권리자(seq 작은 행 중 최신 날짜의 모든 이름)와 집단 내부 여부, 판매자 유형."""
    o = own.sort_values(['rgst_no', 'seq'])
    grp = {}
    for r, s, d, nn, corp, inst in zip(o.rgst_no.values, o.seq.values, o.d.values, o.nn.values, o.corp.values, o.inst.values):
        grp.setdefault(r, []).append((s, d, nn, corp, inst))
    acc = o[o.acc]
    res = []
    for ix, r, s, d, nn in zip(acc.index, acc.rgst_no.values, acc.seq.values, acc.d.values, acc.nn.values):
        prev = [x for x in grp[r] if x[0] < s and x[1] <= d]
        if not prev:
            res.append((ix, False, 'unknown', False)); continue
        md = max(x[1] for x in prev)
        ps = [x for x in prev if x[1] == md]
        same = False
        for x in ps:
            if not x[2] or not nn: continue
            if x[2] == nn: same = True; break
            L = lcp(x[2], nn)
            if L >= 4 or (L >= 2 and x[2][:2] not in STOP2): same = True; break
        if any(x[4] for x in ps): st = 'inst'
        elif any(x[3] for x in ps): st = 'corp'
        else: st = 'indiv_or_other'
        res.append((ix, same, st, any(x[2] == nn for x in ps)))
    R = pd.DataFrame(res, columns=['ix', 'group', 'seller_type', 'same_name']).set_index('ix')
    return R


def alias_at(app, T):
    a = app[(app.vis <= T) & (app.nn.str.len() >= 2)][['nn', 'cd']].drop_duplicates()
    k = a.groupby('nn').cd.nunique()
    uni = a[a.nn.map(k) == 1]
    return dict(zip(uni.nn, uni.cd)), set(k[k > 1].index)


def build(t0, own, docs, app, orig, doc_lab, pat_lab, keep_rows=False):
    """doc_lab: docs와 같은 순서의 라벨 Series(결측=미배정), pat_lab: rgst_no→라벨 dict.
    반환: 통계 dict, 양성 DataFrame(cd, k, 첫 이전 행 속성), blocked 집합, 적격 법인 집합."""
    T = f'{t0}1231'; ws = f'{t0 + 1}0101'; we = f'{t0 + 2}0731'
    n2c, amb = alias_at(app, T)
    dv = docs.vis <= T
    dd = docs[dv & (docs.cd.str[0] == '1')]
    cnt = dd.groupby('cd').appl_no.nunique()
    F = set(cnt[cnt >= 3].index)
    lab_d = doc_lab[dv & (docs.cd.str[0] == '1')]
    ownp = pd.DataFrame({'cd': dd.cd.values, 'k': lab_d.values}).dropna()
    ownp = ownp[ownp.cd.isin(F)].drop_duplicates()
    pr = own[(own.seq > 0) & (own.d <= T)]
    pcd = pr.nn.map(n2c)
    prp = pd.DataFrame({'cd': pcd.values, 'k': pr.rgst_no.map(pat_lab).values}).dropna()
    prp = prp[prp.cd.isin(F)].drop_duplicates()
    blocked = set(zip(ownp.cd, ownp.k)) | set(zip(prp.cd, prp.k))
    w = own[own.acc & (own.d >= ws) & (own.d <= we)].copy()
    w['cd'] = w.nn.map(n2c)
    w['map'] = np.where(w.cd.notna(), 'unique', np.where(w.nn.isin(amb), 'ambiguous', 'none'))
    w['k'] = w.rgst_no.map(pat_lab)
    st = dict(t0=t0, eligible_firms=len(F), win_acc_rows=len(w), win_map_unique=int((w['map'] == 'unique').sum()),
              win_map_amb=int((w['map'] == 'ambiguous').sum()), win_map_none=int((w['map'] == 'none').sum()))
    m = w[w.cd.notna() & (w.cd.str[0] == '1')].copy()
    st['win_mapped_prefix1_rows'] = len(m)
    m['inF'] = m.cd.isin(F)
    m['is_orig'] = [c in orig.get(r, set()) for r, c in zip(m.rgst_no, m.cd)]
    m['blk'] = [(c, k) in blocked for c, k in zip(m.cd, m.k)]
    m['k_ok'] = m.k.notna()
    st['excl_not_eligible_rows'] = int((~m.inF).sum())
    st['excl_original_applicant_rows'] = int((m.inF & m.is_orig).sum())
    st['excl_already_in_k_rows'] = int((m.inF & ~m.is_orig & m.blk).sum())
    st['excl_label_missing_rows'] = int((m.inF & ~m.is_orig & ~m.blk & ~m.k_ok).sum())
    ev = m[m.inF & ~m.is_orig & ~m.blk & m.k_ok].sort_values('d')
    st['event_rows'] = len(ev)
    pos = ev.groupby(['cd', 'k']).agg(d=('d', 'first'), rgst_no=('rgst_no', 'first'), n_rows=('rgst_no', 'size'),
                                       bulk_any=('bulk', 'max'), cs=('cs', 'first'), ix=('d', lambda s: s.index[0])).reset_index()
    pos['t0'] = t0
    st['pos'] = len(pos)
    nb = sum(1 for (c, k) in blocked if c in F)
    st['risk_units'] = len(F) * 50 - nb
    out = (st, pos, blocked, F)
    if keep_rows:
        out = out + (w,)
    return out
