# -*- coding: utf-8 -*-
"""원점별로 다른 spec 을 쓰는 '결측 규칙' 실험(재학습 없음, 이미 학습된 예측 재조합).
  python r3compose.py <EXP> <기본spec> <대체spec> <대체 원점,..> "<변경 설명>"
규칙은 사전에 정한 것: 학습 프레임에 그 외부 피처의 관측값이 하나도 없는 원점은 그 피처가 없는 모형(=기본 spec)을 쓴다."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3lib, r3run
R = r3lib.R
exp, base, alt, alt_os, change = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(','), sys.argv[5]
sb, sa = r3run.load_spec(base)['spec'], r3run.load_spec(alt)['spec']
res = {o: R.score_origin(o, sa if o in alt_os else sb) for o, _, _ in R.ORIGINS}
champ = json.load(open(r3run.CHAMP))['exp']
cres = json.load(open(os.path.join(r3run.RUNS, f'res_{champ}.json'), encoding='utf-8'))
json.dump(dict(spec=sa, parent=champ, change=change, exp=exp, mod=f'compose:{base}+{alt}@{",".join(alt_os)}'),
          open(os.path.join(r3run.SPECS, f'{exp}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
row = r3run.log_exp(exp, champ, change, res, cres, f'재조합(재학습 없음): {",".join(alt_os)}={alt}, 나머지={base}')
if row['decision'] == 'ACCEPT':
    json.dump(dict(exp=exp), open(r3run.CHAMP, 'w')); print('>>> 새 챔피언', exp)
