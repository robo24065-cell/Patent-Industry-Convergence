# T1 R3 B단계 — 국세청 휴·폐업(r3nts) 위험집합 정제: 폐업일 ≤ 원점 연말인 법인의 쌍은 '진입 불가'로 보고 점수를 최저로 둔다(채점 단계, 재학습 없음).
#   python R3/r3nts.py --src R3E11 --dst R3E19      개발 원점 5개 + 보조 원점 c=2015에 적용 → runs/<dst>/o<c>.parquet, 요약은 r3run.py --summarize
# 시점: 폐업일(end_dt)은 되돌려지지 않는 사건 날짜 → 원점 연말 이전 폐업만 사용(사후 정보 아님). 단, 2013년 이전 폐업은 좌측 절단(수집 기록 A3)이라
#       개발 원점(c≤2012)에서는 거의 0건이다. 휴업은 현재 스냅샷이라 쓰지 않는다.
import sys, json, argparse, pathlib
R3 = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(R3))
from r3 import *
EXTD = B/'분석/모델경쟁_20260927/_shared/ext'


def closed_firms(cut):
    s = pd.read_parquet(EXTD/'r3nts_applicant_status.parquet', columns=['applicantCd', 'end_dt'])
    s = s.dropna(subset=['end_dt']); s = s[s.end_dt <= pd.Timestamp(f'{cut}-12-31')]
    return set(s.applicantCd.astype(str))


def apply(src, dst, c):
    o = pd.read_parquet(R3/'runs'/src/f'o{c}.parquet'); cl = closed_firms(c)
    m = o.firm.astype(str).isin(cl).values
    o['px'] = np.where(m, -1.0, o.px.values).astype(np.float32); o['pl'] = np.where(m, -1.0, o.pl.values).astype(np.float32)
    RD = R3/'runs'/dst; RD.mkdir(parents=True, exist_ok=True); o.to_parquet(RD/f'o{c}.parquet', index=False)
    k = ~o.prior_any.values
    st = dict(origin=c, closed_firms_all=len(cl), pairs=int(k.sum()), closed_pairs=int((m & k).sum()), closed_firms_in_risk=int(o.firm[m & k].nunique()),
              pos_closed=int(o.y_u.values[m & k].sum()), rate_closed=float(o.y_u.values[m & k].mean()) if (m & k).any() else None, rate_all=float(o.y_u.values[k].mean()))
    return st, m


def refined_eval(src, c, m):
    """보조 변형(위험집합 정제): 폐업 법인 쌍을 평가에서 빼고 모형·기준선을 같은 집합에서 채점"""
    o = pd.read_parquet(R3/'runs'/src/f'o{c}.parquet'); bs = pd.read_parquet(fc_r3(c)/'baseline_scores.parquet')
    k = (~o.prior_any.values) & (~m); f = o[k]
    s = 0.9 * fr.rank(f.px.values) + 0.1 * fr.rank(f.pl.values)
    mm = mets(f.y_u.values, s, f.newc.values)
    bm = {kk: mets(f.y_u.values, bs[kk].values[k], f.newc.values) for kk in bs.columns}
    b = max(v['auc_all'] for v in bm.values()); bn = max(v['auc_new'] for v in bm.values())
    return dict(auc_all=mm['auc_all'], base_auc_all=b, skill=skill(mm['auc_all'], b), skill_new=skill(mm['auc_new'], bn), n=int(k.sum()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--src', default='R3E11'); ap.add_argument('--dst', default='R3E19'); a = ap.parse_args()
    out = {}
    for c in ORIGINS + AUX:
        st, m = apply(a.src, a.dst, c)
        if c in AUX: st['refined_risk_set'] = refined_eval(a.src, c, m)
        out[c] = st; print(json.dumps(st, ensure_ascii=False), flush=True)
    json.dump(out, open(R3/'runs'/a.dst/'nts_stats.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
