# C18-T2 verify_0 r2: m3_leadlag.py, s3_forecast.py 사본(경로만 변경) 생성
import sys, os
FA = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared_report\fusion_advantage'
R = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(R, 'scr'), exist_ok=True)
s = open(os.path.join(FA, 'map5', 'm3_leadlag.py'), encoding='utf-8').read()
a = 'OUT = os.path.dirname(os.path.abspath(__file__))'; assert s.count(a) == 1
s = s.replace(a, "OUT = os.environ['C18_OUT']")
open(os.path.join(R, 'scr', 'm3_leadlag.py'), 'w', encoding='utf-8').write('# [C18-T2 verify_0 r2 copy] OUT -> env\n' + s)
s = open(os.path.join(FA, 'split34', 's3_forecast.py'), encoding='utf-8').read()
a = 'HERE = os.path.dirname(os.path.abspath(__file__))'
b = 'MC = os.path.abspath(os.path.join(HERE, r"..' + '\\' + '..' + '\\' + '.."))'
assert s.count(a) == 1 and s.count(b) == 1, (s.count(a), s.count(b))
s = s.replace(a, "HERE = os.environ['C18_OUT']").replace(b, 'MC = ' + repr(os.path.abspath(os.path.join(FA, '..', '..'))))
open(os.path.join(R, 'scr', 's3_forecast.py'), 'w', encoding='utf-8').write('# [C18-T2 verify_0 r2 copy] HERE, MC changed\n' + s)
print('ok')
