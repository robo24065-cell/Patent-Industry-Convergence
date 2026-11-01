# verify_1: "NPS 연결 기업 매입 진입 양성률 = 비연결의 2.83배" 반박 검증. 개발 원점(t0=2012–2022)만. 잠금 창 미사용.
import sys, json, time
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from lib_cd import *
V = W + r'\verify_1'
T0 = time.time()
res = {}
# ---- 1. 저장된 칸 표에서 독립 재계산
CE = pd.read_parquet(C + r'\dev_cells.parquet'); P0 = pd.read_parquet(C + r'\dev_pos.parquet')
rl = CE.pos_l.sum() / CE.units_l.sum(); rn = (CE.pos.sum() - CE.pos_l.sum()) / (CE.units.sum() - CE.units_l.sum())
res['cells_recalc'] = dict(pos=int(CE.pos.sum()), pos_l=int(CE.pos_l.sum()), units=int(CE.units.sum()), units_l=int(CE.units_l.sum()),
                           rate_l=rl, rate_n=rn, ratio=rl / rn, pos_nps_from_devpos=int(P0.nps.sum()), n_devpos=len(P0))
# ---- 2. 원천에서 재구성
own, pat, app, docs, orig = load()
doc_lab = docs.ksic.where(docs.ksic.isin(KSIC50)); pat_lab = dict(zip(pat.rgst_no, pat.ksic))
link = pd.read_parquet(MC + r'\_shared\ext\firm_nps_applicant_link.parquet', columns=['applicantCd', 'nps_hit']).drop_duplicates('applicantCd')
L = set(link[link.nps_hit.astype(bool)].applicantCd); BIZ = set(link.applicantCd)
panel = pd.read_parquet(MC + r'\_shared\ext\firm_nps_firm_panel.parquet', columns=['applicantCd', 'snap', 'n_emp', 'active'])
FR = []
for t0 in DEV:
    st, pos, blocked, F, w = build(t0, own, docs, app, orig, doc_lab, pat_lab, keep_rows=True)
    T = f'{t0}1231'
    n2c, amb = alias_at(app, T)
    uniq_cd = set(n2c.values())
    dc = docs[(docs.vis <= T) & docs.cd.isin(F)].groupby('cd').appl_no.nunique()
    nb = pd.Series([c for (c, k) in blocked if c in F]).value_counts()
    npos = pos.cd.value_counts()
    # 연결 시점 정합: t0 말 이전 스냅샷에 등장(2015+만 가능)
    snaps_le = [s for s in panel.snap.unique() if s <= f'{t0}-12']
    seen_le = set(panel[panel.snap.isin(snaps_le)].applicantCd) if snaps_le else set()
    seen_win = set(panel[panel.snap.isin([f'{t0+1}-12', f'{t0+2}-12'])].applicantCd)
    f = pd.DataFrame({'cd': sorted(F)})
    f['t0'] = t0
    f['L'] = f.cd.isin(L); f['biz'] = f.cd.isin(BIZ); f['uniqname'] = f.cd.isin(uniq_cd)
    f['docs'] = f.cd.map(dc).fillna(0).values
    f['units'] = 50 - f.cd.map(nb).fillna(0).values
    f['pos'] = f.cd.map(npos).fillna(0).values
    f['seen_le_t0'] = f.cd.isin(seen_le); f['seen_win'] = f.cd.isin(seen_win)
    FR.append(f)
    assert int(f.pos.sum()) == st['pos'] and int(f.units.sum()) == st['risk_units']
F = pd.concat(FR, ignore_index=True)
F.to_parquet(V + r'\firm_t0.parquet', index=False)


def ratio(d, g):
    a = d[g]; b = d[~g]
    ra = a.pos.sum() / a.units.sum(); rb = b.pos.sum() / b.units.sum()
    return dict(rate_g=float(ra), rate_not=float(rb), ratio=float(ra / rb) if rb > 0 else None, pos_g=int(a.pos.sum()), pos_not=int(b.pos.sum()),
                units_g=int(a.units.sum()), units_not=int(b.units.sum()))


def mh(d, g, strata):
    num = den = 0.0
    for _, s in d.groupby(strata):
        a = s[g]; b = s[~g]; n = s.units.sum()
        if a.units.sum() == 0 or b.units.sum() == 0: continue
        num += a.pos.sum() * b.units.sum() / n; den += b.pos.sum() * a.units.sum() / n
    return float(num / den) if den > 0 else None


res['raw_L'] = ratio(F, F.L)
res['firm_share_L'] = float(F.L.mean())
# 사업자번호 없음 vs 있음·NPS 미매칭 분해
res['unlinked_split'] = {
    'no_bizno': ratio(F[~F.L | ~F.biz], ~F[~F.L | ~F.biz].biz) if False else None}
G = F.copy(); G['grp'] = np.where(G.L, 'L', np.where(G.biz, 'biz_noNPS', 'no_biz'))
res['by_group'] = {k: dict(rate=float(s.pos.sum() / s.units.sum()), pos=int(s.pos.sum()), units=int(s.units.sum()), firms=len(s),
                           docs_median=float(s.docs.median()), uniqname_share=float(s.uniqname.mean())) for k, s in G.groupby('grp')}
# 규모(문서 수) 층화
F['dbin'] = pd.cut(F.docs, [0, 4, 9, 29, 1e9], labels=['3-4', '5-9', '10-29', '30+'])
res['docs_by_L'] = {str(k): float(v) for k, v in F.groupby('L').docs.median().items()}
res['by_docs_bin'] = {str(k): ratio(s, s.L) for k, s in F.groupby('dbin', observed=True)}
res['mh_L_strat_t0_dbin'] = mh(F, F.L, ['t0', 'dbin'])
# 상호 고유 매핑 가능 기업만
U = F[F.uniqname]
res['uniqname_share_by_L'] = {str(k): float(v) for k, v in F.groupby('L').uniqname.mean().items()}
res['raw_L_uniqname_only'] = ratio(U, U.L)
res['mh_L_uniqname_t0_dbin'] = mh(U, U.L, ['t0', 'dbin'])
res['pos_from_nonuniq_firms'] = int(F[~F.uniqname].pos.sum())
# 시점 정합 연결(t0 말 이전 스냅샷 등장), 2015–2022
E = F[F.t0 >= 2015]
res['t2015_raw_L_anytime'] = ratio(E, E.L)
res['t2015_raw_seen_le_t0'] = ratio(E, E.seen_le_t0)
res['t2015_L_but_not_seen_le_t0'] = ratio(E[~(E.L & ~E.seen_le_t0)], E[~(E.L & ~E.seen_le_t0)].L) if False else None
E2 = E.copy(); E2['g'] = np.where(E2.seen_le_t0, 'seen_le_t0', np.where(E2.L, 'L_later_only', 'never'))
res['t2015_groups'] = {k: dict(rate=float(s.pos.sum() / s.units.sum()), pos=int(s.pos.sum()), units=int(s.units.sum()), firms=len(s)) for k, s in E2.groupby('g')}
res['t2015_mh_seen_le_t0_dbin'] = mh(E, E.seen_le_t0, ['t0', 'dbin'])
res['elapsed'] = round(time.time() - T0, 1)
json.dump(res, open(V + r'\v1_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
