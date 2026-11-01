# -*- coding: utf-8 -*-
"""T1 R7 제공 특허-산업 융합데이터 필수성 시험 — 동결 챔피언 R3E29 설정 그대로(음성 25% 추출·XGB 2시드·로지스틱 10% 순위 혼합·IPW·공개 레이블 가중), 피처만 바꿈.
동결 실행기 R3/final_run_r3.py의 train_origin()을 import해 그대로 쓰고, 산출물은 ESSENTIAL_R7/runs/<변형>/ 에만 쓴다(RD 인자).
사용: python r7_t1_run.py <변형> <원점>   (종료코드 3 = 단계 종료, 같은 명령 재실행)
      python r7_t1_run.py score            → results_dev.csv
계산량(기록): 동결 설정의 음성 25% 추출(RandomState 1234+c)이 이미 계산량을 줄이므로 원점 축소 없이 개발 원점 5개(2008–2012) 전부 실행(변형·원점당 약 4–6분, 2–5개 프로세스 병렬).
FULL은 동결 dev 산출물(R3/runs/FROZEN_dev, 읽기만)을 그대로 채점."""
import sys, os, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent; T1 = HERE.parent; R3 = T1/'R3'
sys.path.insert(0, str(R3)); sys.path.insert(0, str(HERE))
os.chdir(str(T1))

B_TXT = ['t_mean', 't_hi', 't_lsum', 't_max', 't_n20', 't_top3', 't_rank', 't_ent', 't_rw', 't_rec3', 't_mom', 't_lrec_n', 't_grow', 't_nbr',
         'ot_mean', 'ot_max', 'ot_top3', 'ot_hi', 'ot_rw', 'ot_nbr', 'oin50', 'tsib3', 'tsib2', 'ti_mean', 'ti_max', 'ti_n20', 'ti_lsum',
         't_rrw', 't_rrec2', 't_rrec2_max', 'ds_lsum', 'ds_max', 'ds_n20', 'ds_rec3', 'ds_n', 'um_lsum', 'um_max', 'um_n20', 'um_rec3', 'um_n']
B_HIST = ['lin', 'inshare', 'dens', 'kpop', 'sib4', 'sib3', 'sib2', 'k_grow', 'k_newent', 'fl_mix', 'fl_max', 'dv_new3', 'dv_new_p3', 'dv_held',
          'io_down', 'io_up', 'io_sell', 'io_max', 'io_same', 'p_n', 'p_k', 'p_share', 'pa_k', 'pa_n', 'pa_rec2', 'pa_k_rec2', 'pa_pend_n', 'pa_share',
          'pa_k_pend', 'peer_k', 'peer_k_rank', 'ent_fit', 'ent_fit_rank', 'r_in2', 'now_in3']
B_IPC = ['ipc_mix', 'ipc_max', 'ipc_fit', 'ipc_fit_rank', 'ipcx_fit', 'ipcx_fit_rank', 'ipcm_fit', 'ipcm_fit_rank']
B_5C = ['ep_k', 'ep_n', 'ep_share', 'ep_k_rec', 'ep_krprio']
B_LAB = ['k_prior', 'k_prior_new']
KEEP_ALWAYS = ['lsize', 'lout', 'r_n1', 'r_n2', 'gap_last', 'n_last1', 'now1', 'now3', 'now_grow', 'vis_frac',
               'pr_ppi_g3', 'pr_ppi_g1', 'pr_exp_g3', 'pr_imp_g3', 'pr_gap', 'k', 'f_any']
VARS = {'R_TXT': B_TXT, 'R_HIST': B_HIST, 'R_IPC': B_IPC, 'R_5C': B_5C, 'R_LAB': B_LAB,
        'R_ALL': B_TXT + B_HIST + B_IPC + B_5C + B_LAB}
SUBS = {'S_TXT': 'alt_txt', 'S_TXTIPC': 'alt_txt', 'S_IPC': None}   # S_TXT: 연계표 라벨로 다시 학습한 제목 분류기 피처(r7_t1_alt.py 산출)
# S_IPC: 목표 KSIC의 IPC 프로파일을 연계표 건수로(r7_t1_ipcsub.py), S_TXTIPC: 둘 다. 피처 목록은 동결 그대로


def check_bundles(cols):
    allb = B_TXT + B_HIST + B_IPC + B_5C + B_LAB + KEEP_ALWAYS
    assert sorted(allb) == sorted(cols), (set(cols) - set(allb), set(allb) - set(cols))


def run(var, c):
    import final_run_r3 as F
    ec = json.loads(json.dumps(F.EC)); check_bundles(ec['cols'])
    RD = HERE/'runs'/var
    if var in VARS:
        ec['cols'] = [x for x in ec['cols'] if x not in VARS[var]]
        FCp = F.fc_r3(c)
    else:
        if SUBS[var] is None:
            FCp = F.fc_r3(c)
        else:
            FCp = HERE/SUBS[var]/f'o{c}'
            assert (FCp/'prep_tr.parquet').exists() and (FCp/'prep_te.parquet').exists(), 'r7_t1_alt.py 먼저'
        if 'IPC' in var:
            import r7_t1_ipcsub; r7_t1_ipcsub.patch()
    ec['ext'] = list(ec['ext'])
    F.train_origin(c, FCp, 'prep_tr.parquet', RD, ec)


def score():
    import numpy as np, pandas as pd
    import r3
    from r3 import mets, skill, baselines
    from final_run_r3 import blend
    rows = []
    for var_dir in sorted([R3/'runs'/'FROZEN_dev'] + [p for p in (HERE/'runs').iterdir() if p.is_dir()]):
        var = 'FULL' if var_dir.name == 'FROZEN_dev' else var_dir.name
        for c in r3.ORIGINS:
            f = var_dir/f'o{c}.parquet'
            if not f.exists():
                continue
            o = pd.read_parquet(f); o = o[~o.prior_any.values]; s = blend(o)
            m = mets(o.y_u.values, s, o.newc.values); b = baselines(c)['strongest']
            rows.append(dict(variant=var, origin=c, auc=m['auc_all'], auc_new=m['auc_new'], base_auc=b['auc_all'],
                             skill=skill(m['auc_all'], b['auc_all']), skill_new=skill(m['auc_new'], b['auc_new']), pr_all=m.get('pr_all'),
                             p_at_1000=float(o.y_u.values[np.argsort(-s)[:1000]].mean()),
                             capture_top1pct=float(o.y_u.values[np.argsort(-s)[:max(1, len(s) // 100)]].sum() / max(1, o.y_u.sum()))))
    R = pd.DataFrame(rows); R.to_csv(HERE/'results_dev.csv', index=False, encoding='utf-8-sig'); print(R.round(4).to_string(index=False))


if __name__ == '__main__':
    if sys.argv[1] == 'score':
        score()
    else:
        run(sys.argv[1], int(sys.argv[2]))
