# -*- coding: utf-8 -*-
# C19-T4 s7: 눈가림 판정 확정(blind_judgments.csv) + 숫자 대조 결과 기입 + sha1 기록. sample_key.csv는 읽지 않는다.
import csv, os, hashlib, json
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.DictReader(open(os.path.join(D, 'cache', 'judg_partial.tsv'), encoding='utf-8'), delimiter='\t'))
NUM = {
 'E-C01-20': '일치: 약 9%=점추정 0.0892, 39.3%=cap_champ 0.3929, 33.3%=cap_SB 0.3333, 차이 하한 0=dcap_SB owner·bundle 하한 0.0 (C01-T3 label_composition.csv, rescoring_sensitivity.csv L_inv)',
 'E-C02-T3-01': '일치(원천 직접): registry_full.sqlite ro 이전 등록사유 있는 특허 88,323/656,360=13.46%, 소유자 행 2개 이상 122,974 (scripts\\s4_registry_check.py)',
 'E-C09T2-01': '일치: 미재현 KR 44.49%(100−55.51)~JP 51.99%(100−48.01) → 44–52% (C10-T3 s2_agree.csv free5_reg, 출원2015-19_발명)',
 'E-C11T2-03': '일치: free_F_reg 31/16/5/0.373, txt_F_reg 29/12/3/0.195 (C11-T2 res\\s7_landscape.csv)',
 'E-C10T1-01': '일치: 0.0022(C18-T3 recheck 143행, C10-T1 S4·S5)',
 'E-C01-02': '일치: P3 5cc 재가중 A·B 한국 1위 9·9, RTA>1 29·27 (C01-T1 rta_sensitivity.csv)',
 'E-C09T2-02': '일치: 해외 free5_reg JP 48.01~CN 53.48 → 48–53% (s2_agree.csv)',
 'E-C06T1-01': '일치: 14,872=10,603+3,458+811, NPS 1,633/14,872=11.0%, DART 811/14,872=5.45%, IPW 0.0618 [0.0145, 0.1095] (C06-T1 out\\s2_meta.json, rescore_B.csv)',
 'E-C03-T2-09': '일치(원천 직접): fusion\\ksic_hs6_map.csv 1,229쌍(중복 0)·HS6 1,019·KSIC 48',
 'E-C12T1-04': '일치: 21묶음 skill 0.1263–0.1685, 하한>0 5/21 (C12-T1 verify_0\\v1_agg_off_*.json), S1 0.1532 [−0.0244, 0.3656] (sens_results.csv)',
 'E-C11T2-02': '일치: 0.19–0.37·최대 0.48 (C18-T3 recheck 168행, C11-T2 verify_5)',
 'E-C15T4-07b': '일치(원천 직접): 656,360=615,042(접두 10)+41,318(접두 20) (s4_registry_check.py)',
 'E-C08T2-01': '일치: 직전 1건 셀 fut 4면 38/38 양성, fut 3이면 0/52 (C08-T2 verify_0 refute_E-C08T2-01.txt, FINAL_R3 parquet 근거)',
 'E-C11T2-01': '일치: prov_vs_txt dAUC 0.0017 [−0.00044, 0.00384], 6/9 (C11-T2 res\\s5_summary_mg__reg.json)',
 'E-C01-08': '일치: US 0.8731·JP 0.8696·EP 0.9047 → 87–90% (_shared_report\\label_origin_R10\\res_fam_check.json)',
 'E-C15T3-01': '일치: C18-T3 recheck(s5_t3_summary.json)',
 'E-C10T3-03': '일치: txt5_reg JP 29.36~EP 30.69 → 29.4–30.7% (s2_agree.csv)',
 'E-C14T2-03': '일치: C18-T3 recheck(pkr_ipc_altA.csv 대안A_REG)',
 'E-C15T3-02': '일치: C18-T3 recheck',
 'E-C13T3-09': '일치: 39.3%·33.3%·하한 0.0 (rescoring_sensitivity.csv L_inv)',
 'E-C12T3-01': '일치: 41%=0.4067(이름·코드)/0.4147(공동출원), 0.44=0.4378/0.4445 (C12-T3 a_concentration.csv)',
 'E-C05T2-01': '일치: 연결률 ~2005 0.179, 2021–25 0.961 (C05-T2 selection_bias.csv)',
 'E-C11T2-04': '일치: txt AUC 0.7950 (s5_summary_mg__reg.json)',
 'E-C02-T3-08': '일치: 46.87–48.21% (C18-T3 recheck, C03-T3 _s8_unseen.json)',
 'E-C05T1-03': '일치: 45%·0.38–0.42 (C18-T3 recheck)',
 'E-C01-21': '일치: hits 39, prior 30, new 9; SB new 4; 78=108−30 (C01-T3 s04_toplist.json)',
 'E-C02-T3-02': '일치: 44–52%(E-C09T2-01과 같은 대조)',
 'E-C10T3-07': '일치: 보존율 hyb_txt_reg 28.6%·hyb_free_reg 22.6% → 23–29%, drop_ros−hyb_txt_reg 0.0242 [0.0194, 0.0289], −hyb_free_reg 0.0262 [0.0212, 0.0308] 9/9 (C10-T3 res\\s6_t3_summary.json)',
 'E-C15T4-07': '일치(원천 직접): 656,360 (s4_registry_check.py)',
 'E-C10T3-05': '일치: reg txt 0.444 [0.162, 0.652], d 0.101 [−0.121, 0.353]; reg free 0.608 [0.382, 0.772], d −0.064 [−0.244, 0.083]; pub 0.554, pure 0.596 (C10-T3 res\\s3_rho_table.csv)',
 'E-C07T3-02': '일치: 2.04=220.3/108(N8 FINAL 보고), 1.57 [1.28, 1.95] (C07-T3 calibration.csv main champ)',
 'E-C10T3-01': '일치: free5_reg US 50.45→50.5, JP 48.01→48.0, 범위 48.0–53.5; 공개번호 연결 JP 46.95(0.469499)→46.9–53.5 (s2_agree.csv)',
}
out = []
for r in rows:
    r = dict(r)
    if r['숫자대조'] == '대조 예정':
        r['숫자대조'] = NUM.get(r['ID'], '미대조')
    out.append(r)
fn = os.path.join(D, 'blind_judgments.csv')
with open(fn, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
h = hashlib.sha1(open(fn, 'rb').read()).hexdigest()
left = [r['ID'] for r in out if r['숫자대조'] == '미대조']
json.dump({'blind_judgments_sha1_before_unblind': h, 'n': len(out), 'num_unchecked': left}, open(os.path.join(D, 'cache', 's7_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
