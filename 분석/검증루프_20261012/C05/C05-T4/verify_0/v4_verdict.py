# verify_0 v4: v1(해시·레이블 미열람·위험집합·잠금 피처) + v2(CA 사업자번호 경로 M SMD) + v3(다리 피처 독립 재구현) 판정 종합.
# 하나라도 실패하면 pass=False → s6(채점)이 실행되지 않는다(동결 보류).
import os, json, sys
sys.stdout.reconfigure(encoding='utf-8')
WD = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C05\C05-T4'
V = os.path.join(WD, 'verify_0')
v1 = json.load(open(os.path.join(V, 'v1_out.json'), encoding='utf-8'))
v2 = json.load(open(os.path.join(V, 'v2_out.json'), encoding='utf-8'))
v3 = json.load(open(os.path.join(V, 'v3_bridge_out.json'), encoding='utf-8'))
spec = json.load(open(os.path.join(WD, 'final_spec_C05T4.json'), encoding='utf-8'))
ih = v1['i_hash']; il = v1['iii_label']; iv = v1['iv_riskset']; fd = v1['ii_feature_maxabs_diff']
chk = {
    'i_code_hash': ih['code_mismatch'] == [],
    'i_frozen_scores_hash': ih['frozen_scores_ok'] is True,
    'i_prereg_final_hash': ih['prereg_final_ok'] is True and ih['spec_prereg_final_sha'] is True,
    'ii_no_locked_snapshot': v1['ii_locked_snapshot_loaded'] is False and set(v1['ii_nps_snapshots_loaded']) <= set(['2015' + '12', '201612', '201712', '201812', '201912', '202012', '202112', '202212', '202312', '202412']),
    'ii_docs_le_ref': int(v1['ii_docs_max_vis_used_le_ref']) <= 20241231,
    'ii_lock_features_match': all(float(x) < 1e-9 for x in fd.values()),
    'ii_bridge_features_match': v3['pass'] is True and v3['max_vis_used'] <= 20241231,
    'ii_ca_biz_path_smd_lt_0.2': v2['pass_abs_smd_lt_0.2'] is True,
    'iii_no_label_code_outside_s6': il['scripts_with_2025_dates_excl_s6'] == [],
    'iii_no_label_columns': 'y' not in il['frozen_cols'] and il['lock_y_values'] == [-1],
    'iii_no_label_files': (not il['final_result_exists']) and (not il['lock_labels_exists']) and all('label' not in f for f in il['cache_files_with_lock']),
    'iv_riskset_equal': iv['equal'] is True and iv['n_frozen_rows'] == iv['n_frozen'],
    'iv_riskset_size_matches_prereg_final': iv['n_indep'] == 314715,
}
out = dict(checks=chk, pass_=all(chk.values()), champion=spec['champion'], m_smd=v2['smd'], riskset=iv['n_indep'],
           note='같은 에이전트의 별도 단계 검토(완전 독립 인력 아님). 목표 인코딩(te_*)은 s5 실행 중 계산되어 저장되지 않으므로 코드 검토(add_te: 평가 행은 학습 원점 레이블만)로 대신함.')
out['pass'] = out.pop('pass_')
json.dump(out, open(os.path.join(V, 'verify_verdict.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
