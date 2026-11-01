# T1 산업 대기실 — 2라운드 동결 실행기 (R2/final_spec_r2.json)
#   python R2/final_run_r2.py --mode dev      개발 원점 2008–2012 재현(학습·채점·기준선) → R2/dev_result.json
#   python R2/final_run_r2.py --mode final    🔒 최종시험: 기준시점 c=2017(피처 = 2017-12-31까지 등록·공개된 정보) → 진입 2018–2022 (F=2018)
#                                             학습 원점 2009–12(진입창 끝 ≤2017 < F), 학습 레이블·분류기·IPW·공개 보강은 2017-12-31 이전 관측분만
#                                             → R2/final_result.json (모형·고정 기준선 지표 + 법인 군집 부트스트랩 95% CI)
#   python R2/final_run_r2.py --mode future   전체 가용 자료: c=2024(→2025–2029 잠재 진입), 학습 원점 2014–17(레이블 cap=2025-08-22) → R2/future_forecast_r2.csv
# 한 번 실행은 9분 이내로 끊긴다(종료코드 3 → 같은 명령 재실행 시 캐시에서 이어짐). 메모리: 음성 25% 추출·float32로 프로세스당 4GB 이하 목표.
# 판정(개발 단계에서 고정): 진입 = (c, c+5] 출원 범위안 특허가 '등록(원출원인 코드) 또는 공개(법인 별칭 매칭)'된 것(공개 기준, 데이터 끝 2025-08-22).
#   위험집합 = 기준시점 이하 출원 중 그 KSIC 등록 0건(1라운드) AND 그 KSIC 공개·등록 선행출원 0건(2라운드 추가).
import sys, json, argparse, pathlib
R2 = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(R2))
from r2 import *
import r2run

SPEC = json.load(open(R2/'final_spec_r2.json', encoding='utf-8'))
EC = SPEC['config']


def build(c, mode):
    """분류기·패널·기준선 패널 구축(1라운드 final_run 함수 재사용, 원점별 cap)."""
    lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR = GUARD[mode]
    cfg = cfg_of(c, mode=mode); FC = fc_of(c, mode=mode); FC.mkdir(parents=True, exist_ok=True)
    fr.T_START = T0; fr.BUDGET = BUDGET
    if all((FC/f'feat_c{t}.parquet').exists() for t in list(cfg['train_cuts']) + [c]) and (mode == 'future' or (FC/f'base_app_c{c}.parquet').exists()):
        return
    new_clf = not all((FC/f).exists() for f in ('P_base.npy', 'P_os51.npy', 'P_ins_cf.npy', 'ep_firm.parquet'))
    P = fr.clf_base(cfg, FC); P51, PO = fr.clf_os51(cfg, FC); Pins, ins_idx = fr.clf_ins(cfg, FC); ep = fr.ep_match(cfg, FC); cdf = fr.lag_cdf(cfg)
    if new_clf: sys.exit(3)   # 분류기 단계와 패널 단계를 다른 프로세스로(메모리)
    built_feat = False
    for t in cfg['train_cuts']:
        if not (FC/f'feat_c{t}.parquet').exists(): fr.cut_features(t, cfg['cap'], cfg, FC, P, P51, PO, Pins, ins_idx, ep, cdf); built_feat = True
    if not (FC/f'feat_c{c}.parquet').exists(): fr.cut_features(c, None, cfg, FC, P, P51, PO, Pins, ins_idx, ep, cdf); built_feat = True
    del P, P51, PO, Pins
    # [실행 오류 수정 2026-09-28] 원래 조건은 기준선 패널이 하나도 없으면 매번 종료해 무한 반복했다 → 이 프로세스에서 패널을 새로 만든 경우에만 종료
    if mode != 'future' and built_feat and not (FC/f'base_app_c{c}.parquet').exists():
        sys.exit(3)           # 기준선 패널은 다음 프로세스에서
    if mode != 'future':
        for t in cfg['train_cuts']: fr.base_app_panel(t, cfg['cap'], FC)
        fr.base_app_panel(c, None, FC)


def boot_ci(y, nc, firm, S, FC, R=200, chunk=10):
    """법인 군집 부트스트랩: 법인을 복원추출하고 그 법인의 모든 쌍을 함께 넣는다. 재표본마다 기준선은 변형 중 최강값."""
    fr_, inv = np.unique(firm, return_inverse=True)
    order = np.argsort(inv, kind='stable'); st = np.searchsorted(inv[order], np.arange(len(fr_))); en = np.searchsorted(inv[order], np.arange(len(fr_)), side='right')
    keys = list(S); parts = []
    for ci in range(int(np.ceil(R / chunk))):
        fp = FC/f'r2boot_R{R}_{ci}.npy'
        if fp.exists(): parts.append(np.load(fp)); continue
        need(chunk * 0.6 * len(keys) * len(y) / 1e6, f'부트스트랩 chunk {ci}')   # 재표본 1회·점수 1종당 AUC 2개 ≈ 0.6s/백만행
        rng = np.random.RandomState(2000 + ci); arr = np.zeros((min(chunk, R - ci * chunk), len(keys), 2))
        for r in range(arr.shape[0]):
            idx = np.concatenate([order[st[i]:en[i]] for i in rng.randint(0, len(fr_), len(fr_))]); yy = y[idx]; nn = nc[idx]
            for j, kk in enumerate(keys):
                ss = S[kk][idx]; arr[r, j] = [roc_auc_score(yy, ss), roc_auc_score(yy[nn], ss[nn])]
        np.save(fp, arr); parts.append(arr)
    A = np.concatenate(parts); q = lambda v: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))]
    out = {kk: {'auc_all': q(A[:, j, 0]), 'auc_new': q(A[:, j, 1])} for j, kk in enumerate(keys)}
    bmax = A[:, 1:, :].max(1)
    out['skill_vs_strongest'] = {'auc_all': q((A[:, 0, 0] - bmax[:, 0]) / (1 - bmax[:, 0])), 'auc_new': q((A[:, 0, 1] - bmax[:, 1]) / (1 - bmax[:, 1]))}
    out['diff_model_minus_strongest'] = {'auc_all': q(A[:, 0, 0] - bmax[:, 0]), 'auc_new': q(A[:, 0, 1] - bmax[:, 1])}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', required=True, choices=['dev', 'final', 'future']); ap.add_argument('--boot', type=int, default=200)
    a = ap.parse_args(); mode = a.mode
    if mode == 'dev':
        RD = R2/'runs'/'FROZEN_dev'; rows = {}
        for c in ORIGINS:   # 메모리: 한 프로세스는 한 원점만 학습(학습 후 종료코드 3 → 재실행)
            if not (RD/f'o{c}.parquet').exists():
                r2run.run_origin('FROZEN', c, run_root=RD, ec=EC, mode='dev', lowmem=True)
                if any(not (RD/f'o{c2}.parquet').exists() for c2 in ORIGINS): sys.exit(3)
        for c in ORIGINS:
            m, _ = r2run.score_origin('FROZEN', c, ec=EC, run_root=RD); b = baselines(c)['strongest']
            m['baseline_strongest_auc_all'] = b['auc_all']; m['baseline_strongest_auc_new'] = b['auc_new']
            m['skill'] = skill(m['auc_all'], b['auc_all']); m['skill_new'] = skill(m['auc_new'], b['auc_new']); rows[c] = m
        res = dict(mode='dev', spec=SPEC['exp_id'], per_origin=rows, mean_skill=float(np.mean([rows[c]['skill'] for c in ORIGINS])),
                   mean_skill_new=float(np.mean([rows[c]['skill_new'] for c in ORIGINS])))
        json.dump(res, open(R2/'dev_result.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
        print(json.dumps({k: v for k, v in res.items() if k != 'per_origin'}, ensure_ascii=False)); return
    c = 2017 if mode == 'final' else 2024
    RD = R2/'runs'/('FROZEN_final' if mode == 'final' else 'FROZEN_future')
    if not (fc_of(c, mode=mode)/'prep_te.parquet').exists():
        build(c, mode); prep(c, mode=mode); sys.exit(3)          # 구축 단계는 별도 프로세스(메모리)
    if not (RD/f'o{c}.parquet').exists():
        r2run.run_origin('FROZEN', c, run_root=RD, ec=EC, mode=mode, lowmem=True); sys.exit(3)   # 학습 단계도 별도 프로세스
    o = pd.read_parquet(RD/f'o{c}.parquet'); bw = EC.get('blend_w', 0.9)
    if mode == 'future':
        te = o[~o.prior_any.values].copy(); te['score'] = bw * fr.rank(te.px.values) + (1 - bw) * fr.rank(te.pl.values)
        K = lib.load()['K']; te['ksic'] = [K[i] for i in te.k.values]
        fm = pd.read_parquet(CACHE/'firms.parquet').set_index('applicant_cd').applicant_name; te['firm_name'] = te.firm.map(fm)
        try:
            js = json.load(open(B/'공모전데이터/외부데이터/KSIC해설/ksic50_definitions.json', encoding='utf-8'))
            nm = {str(x.get('code', x.get('ksic', ''))): (x.get('codenmKr') or x.get('name', '')) for x in js}   # [출력 수정] 해설서 키는 codenmKr if isinstance(js, list) else {str(k2): (v.get('name') if isinstance(v, dict) else str(v)) for k2, v in js.items()}
        except Exception: nm = {}
        g = te.groupby('ksic').agg(at_risk=('px', 'size'), waiting_room_index=('px', 'sum'), mean_prob=('px', 'mean'))
        g['index_newcomer'] = te[te.newc].groupby('ksic').px.sum().reindex(g.index).fillna(0)
        g['index_share'] = g.waiting_room_index / g.waiting_room_index.sum(); g['ksic_name'] = [nm.get(k2, '') for k2 in g.index]
        g['top5_candidates'] = te.sort_values('score', ascending=False).groupby('ksic').head(5).groupby('ksic').firm_name.apply(lambda x: ' | '.join(map(str, x)))
        g['window'] = '2025-2029'; g['info_cutoff'] = '2024-12-31 등록·공개분'
        g.sort_values('waiting_room_index', ascending=False).reset_index().to_csv(R2/'future_forecast_r2.csv', index=False, encoding='utf-8-sig')
        print('saved future_forecast_r2.csv'); return
    # ---- 🔒 최종시험 채점(1회) ----
    lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR = GUARD['final']
    FC = fc_of(c, mode='final')
    te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'cut', 'y_reg', 'newc', 'prior_any'])
    assert (te.firm.values == o.firm.values).all() and (te.k.values == o.k.values).all()
    te['y_pub'] = pub_lab(te, END); te['y_u'] = np.maximum(te.y_reg.values, te.y_pub.values)
    keep = ~te.prior_any.values; t = te[keep]; oo = o[keep]
    s = bw * fr.rank(oo.px.values) + (1 - bw) * fr.rank(oo.pl.values)
    mm = mets(t.y_u.values, s, t.newc.values); mm['auc_all_reglabel'] = float(roc_auc_score(t.y_reg.values, s))
    baselines(c, final=True); bs = pd.read_parquet(FC/'baseline_scores.parquet').loc[keep].reset_index(drop=True)
    bm = {kk: mets(t.y_u.values, bs[kk].values, t.newc.values) for kk in bs.columns}
    strongest = {m: max(v[m] for v in bm.values()) for m in ('auc_all', 'auc_new', 'pr_all', 'pr_new')}
    res = dict(mode='final', spec=SPEC['exp_id'], origin=c, window='2018-2022', F=2018,
               n=dict(rows=int(len(t)), firms=int(t.firm.nunique()), pos=int(t.y_u.sum()), new_rows=int(t.newc.sum()), new_pos=int(t.y_u.values[t.newc.values].sum()),
                      pos_reg=int(t.y_reg.sum()), removed_prior_pairs=int((~keep).sum())),
               model=mm, baselines=bm, baseline_strongest=strongest,
               skill=dict(auc_all=skill(mm['auc_all'], strongest['auc_all']), auc_new=skill(mm['auc_new'], strongest['auc_new'])))
    if a.boot > 0:
        S = {'model': s}; S.update({kk: bs[kk].values for kk in bs.columns})
        res['bootstrap_firm_cluster'] = boot_ci(t.y_u.values, t.newc.values.astype(bool), t.firm.values, S, FC, R=a.boot); res['bootstrap_R'] = a.boot
    json.dump(res, open(R2/'final_result.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(json.dumps(res['skill'])); print('saved final_result.json')


if __name__ == '__main__':
    main()
