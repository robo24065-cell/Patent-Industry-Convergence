# -*- coding: utf-8 -*-
"""T1 R4 — NTIS 2단 보정층 챔피언/도전자 (NTIS Open API 활용). 사전 고정 설계(결과를 보기 전에 작성, 2026-09-29):
  챔피언 = R3E29 동결 예측(R3/runs/FROZEN_dev/o{c}.parquet, 읽기만). 점수 s = 0.9·rank(px) + 0.1·rank(pl) (R3 score_origin 과 같음).
  도전자 = 로지스틱 보정층 logit(s) + NTIS 피처(결측 → 0 + 결측 지시자, 학습 폴드 기준 z-점수).
           개발 원점 안에서 법인 해시 2-fold 교차적합(한쪽 법인으로 계수 학습 → 다른 쪽 채점). ntis_TESTS §6 권고 ①과 같은 구조.
  통제   = 같은 교차적합을 logit(s) 하나로(재보정 효과만). 도전자가 챔피언과 통제 **둘 다**에 대해 +0.003·60% 를 넘어야 채택.
  지표   = R3와 같다: AUC(y_u, prior_any 제외) → skill = (AUC − 최강 기준선 AUC)/(1 − 최강 기준선 AUC). 기준선은 R3/cache/o{c}/baselines_r3.json(읽기만).
  개발 원점 2008–2012(채택 판정), 보조 원점 2015(진입 2016–19, 판정에 쓰지 않음). 보조 원점은 두 가지로 보고:
     (a) 같은 교차적합, (b) 시간 정합 적합: c=2011 원점 전체로 계수 학습(진입창 2012–15 ≤ 2015) → c=2015 에 적용.
  🔒 c=2019 최종시험 예측·레이블(FROZEN_final, cache_final)은 읽지 않는다.
사용: python t1_layer.py R4E01 [R4E02 ...]
"""
import sys, json, pathlib
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
R4 = pathlib.Path(__file__).resolve().parent; T1 = R4.parent; R3 = T1 / 'R3'
DEV = [2008, 2009, 2010, 2011, 2012]; AUXC = 2015
for c in DEV + [AUXC]:
    assert c + 4 <= 2019
BW = 0.9
EXPS = {
    'R4E01': dict(note='채택. 통제(재보정만) 대비 +0.0064(5/5). 보조 c=2015: 교차적합 0.4105, 시간 정합(2011 적합→2015) 0.4128 vs 챔피언 0.4084', feats=['g_tgt_cls', 'g_any', 'g_ln'],
                  change='[NTIS] 과제분류→목표 KSIC 리프트(g_tgt_cls, 배분표 성과 Year ≤ c−2) + 정부 기원 특허 보유·log 건수 — 2단 보정층(법인 2-fold 교차적합). ntis_TESTS T1-N1'),
    'R4E02': dict(parent='R4E01', feats=['g_tgt_ipc'],
                  change='(R4E01 위) [NTIS] 정부 기원 특허 IPC 서브클래스 프로파일 × 목표 KSIC IPC 프로파일 코사인(정부특허 IPC 적합도) + 보유 — 2단 보정층. ntis_TESTS T1-N3'),
    'R4E03': dict(parent='R4E01', feats=['g_self', 'g_coop', 'g_share'],
                  change='(R4E01 위) [NTIS] 정부과제 성과 보유 여부·수행 유형(기업 직접 수행 / 대학·출연연 공동)·보유 비율 — 2단 보정층. ntis_TESTS T1-N2 변형'),
    'R4E04': dict(parent='R4E01', feats=['p_any', 'p_lnf', 'p_tgt_cls'],
                  change='(R4E01 위) [NTIS 과제 raw] 출원인코드→사업자번호(국세청 연결표)→정부과제 수행 이력: 수행 여부·최근 5년 정부연구비 log·수행 과제분류→목표 KSIC 리프트(과제연도 ≤ c−1; 사업자번호가 2010년부터라 c ≤ 2010 은 결측) — 2단 보정층',
                  note='무효 판정 근거 있음: 출원인→사업자번호 표(CORP_APPLICANT 2026-07 스냅샷)는 사후 목록이다. 표에 있는 법인의 진입률이 1.7배(c=2011 1.26% 대 0.74%)이고, 이름으로는 과제 수행이 확인되지만 사업자번호 연결이 안 되는 쌍의 진입률은 0.68%로 평균보다 낮다 → 증분의 상당 부분이 생존 편향. 누수 없는 이름 연결 R4E05는 +0.0004'),
    'R4E05': dict(parent='R4E01', feats=['q_any', 'q_lnf', 'q_tgt_cls'],
                  change='(R4E01 위) [NTIS 과제 raw] 수행기관명↔출원인명 정규화 일치로 정부과제 수행 이력 연결(2002년~, 모든 원점 값 있음): 수행 여부·최근 5년 정부연구비 log·과제분류→목표 KSIC 리프트(과제연도 ≤ c−1) — R4E04의 사업자번호 결측(c ≤ 2010)을 푼 변형, 2단 보정층'),
}


def rank(v): return pd.Series(v).rank(pct=True).values
def skill(a, b): return (a - b) / (1 - b)
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def load(c):
    o = pd.read_parquet(R3 / 'runs' / 'FROZEN_dev' / f'o{c}.parquet')
    o = o[~o.prior_any.values].reset_index(drop=True); o['firm'] = o.firm.astype(str)
    f = pd.read_parquet(R4 / 'cache' / f'ntis_o{c}.parquet')
    assert len(f) == len(o) and (f.firm.values == o.firm.values).all() and (f.k.values == o.k.values).all(), '행 정렬 불일치'
    p2 = R4 / 'cache' / f'ntis_name_o{c}.parquet'
    if p2.exists():
        f2 = pd.read_parquet(p2)
        assert (f2.firm.values == o.firm.values).all() and (f2.k.values == o.k.values).all()
        for x in f2.columns[2:]: f[x] = f2[x].values
    s = BW * rank(o.px.values) + (1 - BW) * rank(o.pl.values)
    b = json.load(open(R3 / 'cache' / f'o{c}' / 'baselines_r3.json', encoding='utf-8'))['strongest']
    return o, f, s, b


def design(f, feats, zc, mu_sd=None):
    cols = [zc]; names = ['zc']
    for x in feats:
        v = f[x].values.astype(float)
        if np.isnan(v).all():
            continue
        miss = np.isnan(v)
        cols.append(np.where(miss, 0.0, v)); names.append(x)
        if miss.any() and not miss.all():
            cols.append(miss.astype(float)); names.append(x + '_na')
    M = np.column_stack(cols)
    if mu_sd is None:
        mu_sd = (M.mean(0), M.std(0) + 1e-9)
    return (M - mu_sd[0]) / mu_sd[1], names, mu_sd


def crossfit(o, f, s, feats):
    zc = logit(s); y = o.y_u.values
    fold = (pd.util.hash_array(o.firm.values.astype(object)) % 2).astype(int)
    pred = np.zeros(len(o)); coefs = []
    for k in (0, 1):
        tr, te = fold != k, fold == k
        Mtr, names, ms = design(f[tr], feats, zc[tr])
        Mte, _, _ = design(f[te], feats, zc[te], ms)
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Mtr, y[tr])
        pred[te] = lr.decision_function(Mte); coefs.append(dict(zip(names, np.round(lr.coef_[0], 4))))
    return pred, coefs


def mets(o, s, b):
    y = o.y_u.values; nc = o.newc.values.astype(bool)
    a = roc_auc_score(y, s); an = roc_auc_score(y[nc], s[nc])
    return dict(auc_all=float(a), auc_new=float(an), skill=float(skill(a, b['auc_all'])), skill_new=float(skill(an, b['auc_new'])))


def all_feats(eid):
    E = EXPS[eid]; par = E.get('parent', 'R3E29')
    return (all_feats(par) if par in EXPS else []) + [x for x in E['feats']]


def run(eid):
    E = EXPS[eid]; par = E.get('parent', 'R3E29'); feats = all_feats(eid)
    res = dict(exp=eid, parent=par, feats=feats, change=E['change'], per_origin={})
    champ = json.load(open(R3 / 'dev_result_r3.json', encoding='utf-8'))['per_origin']
    for c in DEV + [AUXC]:
        o, f, s, b = load(c)
        m0 = mets(o, s, b)
        if str(c) in champ:
            assert abs(m0['skill'] - champ[str(c)]['skill']) < 1e-9, f'챔피언 재현 실패 {c}'
        pc, _ = crossfit(o, f, s, [])
        p1, coefs = crossfit(o, f, s, feats)
        r = dict(champ=m0, control=mets(o, pc, b), chal=mets(o, p1, b), coef=coefs,
                 nonnull={x: round(float(np.mean(~np.isnan(f[x].values))), 3) for x in feats},
                 pos_share={x: round(float(np.nanmean(f[x].values != 0)), 4) for x in feats})
        if c == AUXC:   # (b) 시간 정합: c=2011 전체로 적합 → 2015 적용
            o1, f1, s1, _ = load(2011)
            M1, names, ms = design(f1, feats, logit(s1))
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(M1, o1.y_u.values)
            _, names2, _ = design(f, feats, logit(s), None)
            if names2 == names:
                Mraw, _, _ = design(f, feats, logit(s), ms)      # 2011 적합의 평균·표준편차로 표준화
                r['chal_timefit'] = mets(o, lr.decision_function(Mraw), b); r['timefit_coef'] = dict(zip(names, np.round(lr.coef_[0], 4)))
            else:
                r['chal_timefit'] = None; r['timefit_note'] = f'열 불일치 {names} vs {names2}'
        res['per_origin'][str(c)] = r
        print(eid, c, 'champ', round(m0['skill'], 4), 'ctrl', round(r['control']['skill'], 4), 'chal', round(r['chal']['skill'], 4),
              'new', round(m0['skill_new'], 4), round(r['chal']['skill_new'], 4), coefs[0], flush=True)
    sk = lambda key: [res['per_origin'][str(c)][key]['skill'] for c in DEV]
    ch, ct, cl = sk('champ'), sk('control'), sk('chal')
    if par in EXPS:        # 부모가 보정층이면 부모(현 챔피언) 도전자 점수와 비교. 통제는 부모 자체
        P = json.load(open(R4 / 'runs' / f'{par}.json', encoding='utf-8'))
        ch = [P['per_origin'][str(c)]['chal']['skill'] for c in DEV]; ct = ch
    d_ch = float(np.mean(cl) - np.mean(ch)); w_ch = float(np.mean([a > b_ + 1e-9 for a, b_ in zip(cl, ch)]))
    d_ct = float(np.mean(cl) - np.mean(ct)); w_ct = float(np.mean([a > b_ + 1e-9 for a, b_ in zip(cl, ct)]))
    acc = (d_ch >= 0.003 and w_ch >= 0.6) and (d_ct >= 0.003 and w_ct >= 0.6)
    res.update(mean_skill=float(np.mean(cl)), mean_champ=float(np.mean(ch)), mean_control=float(np.mean(ct)),
               delta_vs_champ=d_ch, win_vs_champ=w_ch, delta_vs_control=d_ct, win_vs_control=w_ct, decision='ACCEPT' if acc else 'ROLLBACK')
    (R4 / 'runs').mkdir(exist_ok=True)
    json.dump(res, open(R4 / 'runs' / f'{eid}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    A = res['per_origin'][str(AUXC)]
    row = dict(exp_id=eid, parent=par, change=E['change'])
    for c in DEV: row[f'skill_o{c}'] = round(res['per_origin'][str(c)]['chal']['skill'], 4)
    row.update(mean_skill=round(res['mean_skill'], 4),
               mean_skill_new=round(float(np.mean([res['per_origin'][str(c)]['chal']['skill_new'] for c in DEV])), 4),
               champ_mean=round(res['mean_champ'], 4), control_mean=round(res['mean_control'], 4),
               delta_vs_champion=round(d_ch, 4), win_rate=w_ch, delta_vs_control=round(d_ct, 4), win_rate_vs_control=w_ct,
               aux2015_champ=round(A['champ']['skill'], 4), aux2015_crossfit=round(A['chal']['skill'], 4),
               aux2015_timefit=round(A['chal_timefit']['skill'], 4) if A.get('chal_timefit') else '',
               decision=res['decision'], note=E.get('note', ''))
    p = R4 / 'experiments.csv'
    df = pd.DataFrame([row])
    if p.exists():
        old = pd.read_csv(p, encoding='utf-8-sig'); old = old[old.exp_id != eid]; df = pd.concat([old, df], ignore_index=True)
    df.to_csv(p, index=False, encoding='utf-8-sig')
    print(json.dumps({k: v for k, v in row.items() if k != 'change'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    for e in sys.argv[1:]:
        run(e)
