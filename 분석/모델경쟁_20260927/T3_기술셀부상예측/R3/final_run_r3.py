"""T3 3라운드 동결 모형 실행기 (final_spec_r3.json). 공개연도 기준 변형.

사용 (어디서든):
  python final_run_r3.py --mode dev      # 개발 원점 2010..2018 재현 → dev_result_r3.json (캐시 cache_pub, 공개연도 ≤2021, 🔒 F'=2022 미접촉)
                                         #   --fresh: 성분 예측 캐시를 쓰지 않고 처음부터 다시 학습
  python final_run_r3.py --mode final    # 🔒 평가자 전용 1회 (PROTOCOL_R3 §3). cache_pub_final(공개연도 ≤2024) 빌드
                                         #   → 학습 t0 2005..2018 (레이블 결과창 공개연도 ≤2021 < F'=2022)
                                         #   → t0=2021 채점(KR 공개연도 2022–24 대 2019–21 성장 상위 10%) + IPC 메인그룹 군집 부트스트랩 95% CI → final_result_r3.json
  python final_run_r3.py --mode selftest # 코드 점검(개발 캐시만): final·future 코드 경로를 가짜 원점(평가 2017·2018, 학습 ≤2014 / 예측 2018, 학습 ≤2015)으로 실행.
                                         #   외부 피처를 _selftest 접미어로 새로 빌드해 A·B단계 산출물과 값이 같은지 대조
  python final_run_r3.py --mode future   # final 뒤에만. 학습 t0 2005..2021 → t0=2024 예측(공개연도 2025–27) → future_forecast_r3.csv
빌드 단계는 산출물이 있으면 건너뛴다(9분 제한에 걸리면 같은 명령을 다시 실행).
피처는 모두 t0년 말까지 공개된 자료만(셀 집계 = 공개연도 ≤ t0, KOSIS 생산지수 t0-1년 12월~t0년 11월, 외부 B단계 규칙은 build_ext_r3b.py 머리말).
"""
import os, sys, json, argparse, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); T3 = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--mode', required=True, choices=['dev', 'final', 'future', 'selftest'])
ap.add_argument('--n_boot', type=int, default=1000)
ap.add_argument('--fresh', action='store_true')
args = ap.parse_args()
SPEC = json.load(open(os.path.join(HERE, 'final_spec_r3.json'), encoding='utf-8'))

env = dict(T3R3_VAR='pub', PYTHONIOENCODING='utf-8')
if args.mode == 'selftest':
    env.update(T3R3_CACHE='cache_pub', T3R3_MAXYEAR='2021', T3R3_LABEL_MAX='2021', T3R3_EXT_SUFFIX='_selftest', T3R3_EXTB_SUFFIX='_selftest', T3R3_TRAIN_MAX_T0='2014')
    os.environ.pop('T3R3_FINAL_UNLOCK', None)
    build_env = dict(T3_CACHE='R3/cache_pub', T3_MAXYEAR='2021', T3_T0MAX='2018'); T0MAX_FEAT = 2018
elif args.mode == 'dev':
    env.update(T3R3_CACHE='cache_pub', T3R3_MAXYEAR='2021', T3R3_LABEL_MAX='2021', T3R3_EXT_SUFFIX='', T3R3_EXTB_SUFFIX='')
    os.environ.pop('T3R3_FINAL_UNLOCK', None); os.environ.pop('T3R3_TRAIN_MAX_T0', None)
    build_env = dict(T3_CACHE='R3/cache_pub', T3_MAXYEAR='2021', T3_T0MAX='2018'); T0MAX_FEAT = 2018
else:
    # 🔒 final/future 에서만 공개연도 2022 이후 자료 해제
    env.update(T3R3_CACHE='cache_pub_final', T3R3_MAXYEAR='2024', T3R3_LABEL_MAX='2024', T3R3_FINAL_UNLOCK='1')
    build_env = dict(T3_CACHE='R3/cache_pub_final', T3_MAXYEAR='2024', T3_T0MAX='2024', T3R3_FINAL_UNLOCK='1')
    if args.mode == 'final':
        env.update(T3R3_TRAIN_MAX_T0=str(SPEC['final']['train_max_t0']), T3R3_EXT_SUFFIX='_final', T3R3_EXTB_SUFFIX='_final')
        T0MAX_FEAT = SPEC['final']['test_t0']          # 외부 피처는 t0 ≤2021 까지만 만든다
    else:
        if not os.path.exists(os.path.join(HERE, 'final_result_r3.json')): raise SystemExit('future 는 final 실행 뒤에만')
        env.update(T3R3_TRAIN_MAX_T0=str(SPEC['future']['last_train_t0']), T3R3_EXT_SUFFIX='_future', T3R3_EXTB_SUFFIX='_future')
        T0MAX_FEAT = SPEC['future']['t0']
os.environ.update(env)
CACHE = os.path.join(HERE, env['T3R3_CACHE']); os.makedirs(CACHE, exist_ok=True)

def run(cmd, cwd):
    t = time.time(); print('[build]', ' '.join(cmd), flush=True)
    r = subprocess.run([sys.executable] + cmd, cwd=cwd, env={**os.environ, **build_env})
    if r.returncode != 0: raise SystemExit(f'{cmd} 실패')
    print(f'[build] 완료 {time.time()-t:.0f}s', flush=True)
# 1) 공개연도 셀 패널: base(R3 빌더) → panel·feat2(1라운드 빌더, 수정 없음)
for step, cwd, out in [(os.path.join('R3', 'build_base_pub.py'), T3, 'base_rows.parquet'), ('build_panel.py', T3, 'panel_ccy.parquet'), ('build_feat2.py', T3, 'feat2.parquet')]:
    if not os.path.exists(os.path.join(CACHE, out)): run([step], cwd)

sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import r3lib as L
import r3core as C
from sklearn.metrics import roc_auc_score
assert L.CACHE.endswith(env['T3R3_CACHE'])
CFG = SPEC['cfg']; TAG = {'dev': 'dev_fresh' if args.fresh else 'dev', 'final': 'final', 'future': 'future', 'selftest': 'selftest'}[args.mode]
SUF = env['T3R3_EXT_SUFFIX']
T0S_FEAT = list(range(2005, T0MAX_FEAT + 1))

# 2) 외부 피처: KOSIS 광공업생산지수 성장률 3피처(build_ext3.py §1과 같은 규칙) → ext_ksic_t0_pub{SUF}.parquet
def build_kosis_pi(out):
    DATA = os.path.join(HERE, '..', '..', '..', '..', '공모전데이터')
    pi = pd.read_csv(os.path.join(DATA, '외부데이터', '경기지표', 'kosis_광공업생산지수_월별.csv'), dtype={'산업코드': str}); pi['값'] = pd.to_numeric(pi.값, errors='coerce')
    pi['y'] = pi.연월 // 100; pi['m'] = pi.연월 % 100
    pi = pi[pi.연월 <= T0MAX_FEAT * 100 + 11]      # 마지막 원점 t0년 11월까지(12월 지수는 이듬해 1월 공표)
    pi['wy'] = np.where(pi.m == 12, pi.y + 1, pi.y)
    cnt = pi.groupby(['산업코드', '항목코드', 'wy']).값.count(); PA = pi.groupby(['산업코드', '항목코드', 'wy']).값.mean()[cnt == 12]
    def pv(code, it, y):
        try: return PA.loc[(code, it, y)]
        except KeyError: return np.nan
    lg = lambda a_, b_: np.log(a_ / b_) if (a_ == a_ and b_ == b_ and a_ > 0 and b_ > 0) else np.nan
    rows = []
    for ks in sorted(set(L.load()['ksic'])):
        c3 = 'C' + ks[:3]; c2 = 'C' + ks[:2]; code = c3 if (c3, 'T10', 2010) in PA.index else c2
        for t0 in T0S_FEAT:
            p = lambda it, y: pv(code, it, y)
            rows.append(dict(ksic=ks, t0=t0,
                             pi_prod_g2=lg((p('T10', t0) + p('T10', t0 - 1)) / 2, (p('T10', t0 - 2) + p('T10', t0 - 3)) / 2),
                             pi_prod_g1=lg(p('T10', t0), p('T10', t0 - 1)),
                             pi_ship_g2=lg((p('T11', t0) + p('T11', t0 - 1)) / 2, (p('T11', t0 - 2) + p('T11', t0 - 3)) / 2)))
    pd.DataFrame(rows).to_parquet(out)
ext_file = os.path.join(HERE, f'ext_ksic_t0_pub{SUF}.parquet')
if args.mode != 'dev': build_kosis_pi(ext_file)          # dev 는 A단계 산출물(ext_ksic_t0_pub.parquet)을 그대로 쓴다
# 3) B단계 외부 피처(RoS·OpenAlex 셀·KOSIS 품목편 등) → extB_*_pub{SUF}.parquet
import build_ext_r3b as XB
if args.mode != 'dev' or not os.path.exists(os.path.join(HERE, 'extB_cell_pub.parquet')):
    XB.build(L, T0S_FEAT, SUF)

def prep(D):
    cf = pd.read_parquet(os.path.join(HERE, f'extB_cell_pub{SUF}.parquet')); kf = pd.read_parquet(os.path.join(HERE, f'extB_ksic_pub{SUF}.parquet'))
    n0 = len(D); D = D.merge(cf, on=['cell', 't0'], how='left').merge(kf, on=['ksic', 't0'], how='left'); assert len(D) == n0
    D['oa_mg_rel'] = D.oa_mg_g - D.oa_all_g if 'oa_all_g' in D else np.nan
    D['ix_sci_fx'] = D.ros_sci_rel * D.fx_mom; D['ix_sci_kr'] = D.ros_sci_rel * D.kr_mom; D['ix_oa_gap'] = D.oa_mg_g - D.kr_mom
    return D
_gw = C.get_wide
def gw(Dx, mt):
    w = _gw(Dx, mt)
    if 'ros_sci' not in w: w = prep(w); C._WIDE[mt] = w
    return w
C.get_wide = gw
need = sorted({f for c in CFG['components'] for f in c['features']})
if 'oa_mg_rel' in need: need.append('oa_all_g')   # 파생 피처의 원천도 있어야 함

def cluster_boot(blocks, n_boot, seed=0):
    """blocks: [(mg, yb, p_model, p_base)] 원점별. 군집 = IPC 메인그룹(같은 메인그룹의 다른 KSIC 셀·다른 원점 함께 뽑힘)."""
    rng = np.random.default_rng(seed); ums = np.unique(np.concatenate([b[0] for b in blocks]))
    gi = [pd.Series(np.arange(len(b[0]))).groupby(b[0]).apply(lambda s: s.values).to_dict() for b in blocks]
    out = []
    for _ in range(n_boot):
        s = rng.choice(ums, len(ums)); am, ab = [], []
        for b, g in zip(blocks, gi):
            ii = np.concatenate([g[u] for u in s if u in g]); y = b[1][ii]
            if y.min() == y.max(): continue
            am.append(roc_auc_score(y, b[2][ii])); ab.append(roc_auc_score(y, b[3][ii]))
        a, a0 = np.mean(am), np.mean(ab)
        out.append((a, a0, (a - a0) / (1 - a0), a - a0, np.mean([(x - y0) / (1 - y0) for x, y0 in zip(am, ab)])))
    o = np.array(out); q = lambda j: [float(v) for v in np.percentile(o[:, j], [2.5, 97.5])]
    return dict(auc=q(0), auc_baseline=q(1), skill=q(2), auc_diff=q(3), skill_mean_of_origins=q(4), n_boot=n_boot,
                cluster='IPC 메인그룹', n_clusters=int(len(ums)))

def evaluate(D, origins, out_json, pred_file, boot):
    res, blocks, preds = {}, [], []
    for v in origins:
        te = D[D.t0 == v]; tr = C.train_rows(D, v, CFG)
        assert tr.t0.max() + 3 <= v and tr.t0.max() + 3 < L.F_YEAR, '학습 레이블은 결과창이 F\'(2022) 이전에 끝난 원점만'
        p = C.score(CFG, D, v, TAG); p0 = C.baseline_pred(D, v)
        m = C.metrics(te.y.values, te.yb.values, p); b = C.metrics(te.y.values, te.yb.values, p0)
        res[v] = dict(model=m, baseline=b, skill=(m['auc'] - b['auc']) / (1 - b['auc']), train_t0=[int(tr.t0.min()), int(tr.t0.max())], n_train=int(len(tr)))
        print(f"t0={v}: 모형 AUC {m['auc']:.4f} PR-AUC {m['prauc']:.3f} P@10 {m['p_at10']:.3f} | 기준선 AUC {b['auc']:.4f} PR-AUC {b['prauc']:.3f} P@10 {b['p_at10']:.3f} | skill {res[v]['skill']:.4f}", flush=True)
        blocks.append((te.mg.values, te.yb.values, p, p0))
        preds.append(pd.DataFrame(dict(cell=te.cell.values, ksic=te.ksic.values, mg=te.mg.values, t0=v, kr_W=te.kr_W.values, p=p, p_base=p0, y=te.y.values, yb=te.yb.values)))
    au = np.mean([r['model']['auc'] for r in res.values()]); a0 = np.mean([r['baseline']['auc'] for r in res.values()])
    summ = dict(auc=float(au), auc_baseline=float(a0), skill_of_mean_auc=float((au - a0) / (1 - a0)),
                mean_skill=float(np.mean([r['skill'] for r in res.values()])),
                prauc=float(np.mean([r['model']['prauc'] for r in res.values()])), prauc_baseline=float(np.mean([r['baseline']['prauc'] for r in res.values()])),
                p_at10=float(np.mean([r['model']['p_at10'] for r in res.values()])), p_at10_baseline=float(np.mean([r['baseline']['p_at10'] for r in res.values()])),
                rho=float(np.mean([r['model']['rho'] for r in res.values()])), rho_baseline=float(np.mean([r['baseline']['rho'] for r in res.values()])))
    out = dict(mode=args.mode, model=SPEC['model_name'], variant='공개연도(pub)', origins=origins, summary=summ,
               by_origin={str(k): v for k, v in res.items()}, created=time.strftime('%Y-%m-%d %H:%M:%S'))
    if boot: out['ci95'] = cluster_boot(blocks, args.n_boot)
    json.dump(out, open(os.path.join(HERE, out_json), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    os.makedirs(os.path.join(HERE, 'preds'), exist_ok=True)
    pd.concat(preds, ignore_index=True).to_parquet(os.path.join(HERE, 'preds', pred_file))
    print(json.dumps(dict(summary=summ, ci95=out.get('ci95')), ensure_ascii=False, indent=1))
    return out

def run_future(ft, lt, prefix):
    D = prep(C.get_D(t0s=list(range(C.FIRST_T0, lt + 1)) + [ft]))
    te = D[D.t0 == ft].copy(); te['score'] = C.score(CFG, D, ft, TAG)
    te['rank_pct'] = te.score.rank(pct=True, ascending=False); te['pred_breakout_top10'] = (te.rank_pct <= 0.10).astype(int)
    te['rank_in_ksic'] = te.groupby('ksic').score.rank(ascending=False, method='first').astype(int)
    cols = [c for c in ['ksic', 'mg', 'kr_W', 'tot_W', 'kr_mom', 'US_mom', 'CN_mom', 'JP_mom', 'EP_mom', 'pi_prod_g2', 'ros_sci', 'oa_mg_g',
                        'kr_share_rel_ksic', 'score', 'rank_pct', 'pred_breakout_top10', 'rank_in_ksic'] if c in te]
    te = te.sort_values('score', ascending=False); te[cols].to_csv(os.path.join(HERE, prefix + '.csv'), index=False, encoding='utf-8-sig')
    te.groupby('ksic').agg(n_cells=('score', 'size'), n_pred_breakout=('pred_breakout_top10', 'sum')) \
      .assign(pred_rate=lambda d: d.n_pred_breakout / d.n_cells).sort_values('pred_rate', ascending=False) \
      .to_csv(os.path.join(HERE, prefix + '_ksic.csv'), encoding='utf-8-sig')
    json.dump(dict(mode=args.mode, t0=ft, window=[ft + 1, ft + 3], train_t0=[C.FIRST_T0, lt], n_cells=int(len(te)), created=time.strftime('%Y-%m-%d %H:%M:%S'),
                   note='공개연도 기준. KR·US·EP·CN 공개연도 2024 완결(2025 부분) → 예측 원점 2024'),
              open(os.path.join(HERE, prefix + '_meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('저장:', prefix + '.csv')

t = time.time()
if args.mode == 'dev':
    D = prep(C.get_D()); miss = [f for f in need if f not in D]; assert not miss, miss
    out = evaluate(D, SPEC['dev']['origins'], 'dev_result_r3_fresh.json' if args.fresh else 'dev_result_r3.json', 'dev_preds_r3.parquet', boot=True)
    ref = SPEC['dev']['mean_skill']
    print(f"재현 점검: 평균 skill {out['summary']['mean_skill']:.4f} vs 동결 기록 {ref:.4f} → {'일치' if abs(out['summary']['mean_skill'] - ref) < 5e-4 else '불일치!'}")
elif args.mode == 'final':
    ft = SPEC['final']['test_t0']; tmax = SPEC['final']['train_max_t0']
    assert ft == 2021 and tmax + 3 < L.F_YEAR
    D = prep(C.get_D(t0s=list(range(C.FIRST_T0, tmax + 1)) + [ft])); miss = [f for f in need if f not in D]; assert not miss, miss
    assert D[D.t0 < ft].t0.max() == tmax
    evaluate(D, [ft], 'final_result_r3.json', 'FINAL_R3_t0_2021.parquet', boot=True)
elif args.mode == 'selftest':
    # 외부 피처 재빌드 대조(같은 규칙이면 A·B단계 산출물과 값이 같아야 함)
    a0 = pd.read_parquet(os.path.join(HERE, 'ext_ksic_t0_pub.parquet'))[['ksic', 't0', 'pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2']]
    a1 = pd.read_parquet(ext_file).merge(a0, on=['ksic', 't0'], suffixes=('', '_A'))
    b0 = pd.read_parquet(os.path.join(HERE, 'extB_cell_pub.parquet')); b1 = pd.read_parquet(os.path.join(HERE, 'extB_cell_pub_selftest.parquet'))
    chk = dict(kosis_pi_maxdiff=float(max(np.nanmax(np.abs(a1[c] - a1[c + '_A'])) for c in ['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2'])),
               extB_equal=bool(b0.equals(b1)))
    print('외부 피처 재빌드 대조:', chk, flush=True)
    D = prep(C.get_D(t0s=list(range(C.FIRST_T0, 2015)) + [2017, 2018])); miss = [f for f in need if f not in D]; assert not miss, miss
    o = evaluate(D, [2017, 2018], 'selftest_final_r3.json', 'selftest_final_r3.parquet', boot=True)
    o['rebuild_check'] = chk; json.dump(o, open(os.path.join(HERE, 'selftest_final_r3.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    C._D.clear(); C._WIDE.clear(); os.environ['T3R3_TRAIN_MAX_T0'] = '2015'; C.TRAIN_MAX_T0 = 2015
    run_future(2018, 2015, 'selftest_future_r3')
else:
    run_future(SPEC['future']['t0'], SPEC['future']['last_train_t0'], 'future_forecast_r3')
print(f'총 {time.time()-t:.0f}s')
