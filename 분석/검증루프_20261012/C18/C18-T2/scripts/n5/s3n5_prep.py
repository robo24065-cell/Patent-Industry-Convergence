# C18-T2 N5 준비: n5lib.py 사본(c18n5lib.py) 생성 — CACHE → C18_N5CACHE(이 폴더), 패밀리 메타 → C18_META. 공식 kr_/ext_ 캐시는 복사해 쓴다(읽기만).
import os, sys, shutil
W = os.environ['C18_W']
N5 = os.path.abspath(os.path.join(W, '..', '..', '..', '모델경쟁_20260927', 'N5_표준특허선언레이더'))
src = open(os.path.join(N5, 'n5lib.py'), encoding='utf-8').read()
reps = [('CACHE = os.path.join(OUT, "cache")', 'CACHE = os.environ["C18_N5CACHE"]  # [C18-T2] 이 폴더'),
        ('    meta = os.path.join(MC, "_shared", "meta", "*.parquet").replace("\\\\", "/")', '    meta = os.environ["C18_META"].replace("\\\\", "/")  # [C18-T2] 변형 메타')]
for a, b in reps:
    assert src.count(a) == 1, a
    src = src.replace(a, b)
src = '# [C18-T2 사본] N5 n5lib.py 사본 — CACHE·패밀리 메타 경로만 바꿈\n' + src
D = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(D, 'c18n5lib.py'), 'w', encoding='utf-8').write(src)
for V in ('P0', 'V1', 'V1E'):
    cd = os.path.join(W, 'cache', f'n5_{V}'); os.makedirs(cd, exist_ok=True)
    for f in ('kr_26429.parquet', 'ext_26429.pkl'):
        if not os.path.exists(os.path.join(cd, f)):
            shutil.copyfile(os.path.join(N5, 'cache', f), os.path.join(cd, f))
print('ok')
