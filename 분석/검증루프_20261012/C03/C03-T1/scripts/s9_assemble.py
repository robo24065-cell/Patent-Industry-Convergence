# C03-T1 s9: rescore_sensitivity.csv 조립(s3 점추정 + s4 부트스트랩 차이 + s6 이름 최악 가정 + P2 판정)
import json, pathlib, sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
W = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C03\C03-T1"); WC = W/'cache'
s3 = json.load(open(WC/'s3_out.json', encoding='utf-8')); s4 = json.load(open(WC/'s4_out.json', encoding='utf-8')); s6 = json.load(open(WC/'s6_out.json', encoding='utf-8'))
TH = dict(auc_all=0.00697/4, skill_all=0.03483/4, auc_new=0.01974/4, skill_new=0.06565/4)
F = s3['final']; off = F['official']; rows = []
def verdict(rate, d):
    small = all(abs(d[k]) < TH[k] for k in TH)
    if rate >= 0.05: return '수정 필요'
    if rate >= 0.02 or not small: return '단서 필요'
    return '영향 없음'
rows.append(dict(scenario='공식(재구성)', pos=off['pos'], pos_change_rate=0.0, auc_all=off['model']['auc_all'], auc_new=off['model']['auc_new'], strongest_auc_all=off['strongest']['auc_all'], strongest_arg=off['strongest_arg'],
                 skill_all=off['skill_all'], skill_new=off['skill_new'], skill_all_ci=s4['official_all']['skill'], skill_new_ci=s4['official_new']['skill'],
                 top100_stable=off['topk_model_stable']['100'], top1000_stable=off['topk_model_stable']['1000'], top10000_stable=off['topk_model_stable']['10000'],
                 top1000_random5_mean=off['topk_model_random_mean']['1000'], base_top1000=off['topk_base_stable']['1000'], newc_pos=off['new_pos'], newc_top100=off['topk_model_new_stable']['100'], newc_top1000=off['topk_model_new_stable']['1000'],
                 auc_all_randtie=off['auc_all_randtie'], P2='-'))
for w, lab in (('R_self', '공란 복원 주판(R_self)'), ('R_any', '공란 복원 상한(R_any)')):
    r = F[w]; d = r['delta']; dd = dict(auc_all=d['auc_all'], skill_all=d['skill_all'], auc_new=d['auc_new'], skill_new=d['skill_new'])
    rows.append(dict(scenario=lab, pos=r['pos'], pos_change_rate=r['pos_change_rate'], new_pos=r['new_pos'], new_pos_newc=r['new_pos_newc'], auc_all=r['model']['auc_all'], auc_new=r['model']['auc_new'],
                     strongest_auc_all=r['strongest']['auc_all'], strongest_arg=r['strongest_arg'], skill_all=r['skill_all'], skill_new=r['skill_new'],
                     skill_all_ci=s4[f'{w}_all']['skill'], skill_new_ci=s4[f'{w}_new']['skill'],
                     d_auc_all=d['auc_all'], d_auc_all_ci=s4[f'delta_{w}_all']['d_auc'], d_skill_all=d['skill_all'], d_skill_all_ci=s4[f'delta_{w}_all']['d_skill'],
                     d_auc_new=d['auc_new'], d_auc_new_ci=s4[f'delta_{w}_new']['d_auc'], d_skill_new=d['skill_new'], d_skill_new_ci=s4[f'delta_{w}_new']['d_skill'],
                     top100_stable=r['topk_model_stable']['100'], top1000_stable=r['topk_model_stable']['1000'], top10000_stable=r['topk_model_stable']['10000'],
                     top1000_random5_mean=r['topk_model_random_mean']['1000'], base_top1000=r['topk_base_stable']['1000'], newc_pos=off['new_pos'] + r['new_pos_newc'], newc_top100=r['topk_model_new_stable']['100'], newc_top1000=r['topk_model_new_stable']['1000'],
                     auc_all_randtie=r['auc_all_randtie'], skill_lo_gt0=s4[f'{w}_all']['skill_lo_gt0'], model_gt_strongest_boot_share=s4[f'{w}_all']['model_gt_strongest_share'], P2=verdict(r['pos_change_rate'], dd)))
for k_, lab in (('drop_suspicious_all', '이름매칭 의심 양성 전부 제거(최악, 사후)'), ('drop_homonym_only', '동명 다른 코드 양성만 제거(사후)')):
    r = s6[k_]; rows.append(dict(scenario=lab, pos=r['pos'], pos_change_rate=r['pos_change_rate'], auc_all=r['auc_all'], auc_new=r['auc_new'], strongest_auc_all=r['strongest_all'], skill_all=r['skill_all'], skill_new=r['skill_new'],
                                 d_auc_all=r['d_auc_all'], d_skill_all=r['d_skill_all'], d_auc_new=r['d_auc_new'], d_skill_new=r['d_skill_new'], P2='(사후·판정 외)'))
for c, v in s3['dev'].items():
    rows.append(dict(scenario=f'개발 원점 {c} 공란 복원(R_self)', pos=v['pos'], pos_change_rate=v['pos_change_rate'], new_pos=v['new_pos_R_self'], auc_all=v['auc_all'], strongest_auc_all=v['strongest'], skill_all=v['skill'], skill_new=v['skill_new'],
                     d_auc_all=v['dauc'], d_skill_all=v['dskill'], d_skill_new=v['dskill_new'], pub_only_share=v['pub_only_share'], P2='(P4 진단)'))
pd.DataFrame(rows).to_csv(W/'rescore_sensitivity.csv', index=False, encoding='utf-8-sig')
print(pd.DataFrame(rows)[['scenario', 'pos', 'pos_change_rate', 'd_auc_all', 'd_skill_all', 'd_auc_new', 'd_skill_new', 'P2']].to_string())
