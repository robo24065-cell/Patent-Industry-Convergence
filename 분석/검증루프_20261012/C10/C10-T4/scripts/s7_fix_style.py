# build_test\_style.py 의 BASE 줄을 절대 경로로 바꾼다(테스트 렌더 전용, 이 폴더 안에서만 씀)
import os
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C10\C10-T4\build_test"
NEW = r'BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"'
b = open(os.path.join(T, "_style.py"), "rb").read().split(b"\n")
for i, l in enumerate(b):
    if l.startswith(b"BASE = "):
        b[i] = NEW.encode("utf-8")
open(os.path.join(T, "_style.py"), "wb").write(b"\n".join(b))
print("ok")
