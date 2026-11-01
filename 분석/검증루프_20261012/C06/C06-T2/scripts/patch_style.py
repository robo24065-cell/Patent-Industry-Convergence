# regen/_style.py 15행(BASE)을 절대경로로 바꾼다(샌드박스 재생성용). 원본 figs 폴더는 건드리지 않는다.
import os
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'regen')
p = os.path.join(D, '_style.py')
b = open(p, 'rb').read().split(b'\n')
b[14] = 'BASE = os.path.normpath(os.path.join(OUT, "..", "..", "..", "..", "모델경쟁_20260927"))'.encode('utf-8')
open(p, 'wb').write(b'\n'.join(b))
import sys; sys.path.insert(0, D)
import _style
print(_style.BASE, os.path.isdir(_style.BASE))
