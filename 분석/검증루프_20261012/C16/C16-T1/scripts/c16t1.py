# -*- coding: utf-8 -*-
"""C16-T1 트랙 import 공용: guard16(쓰기 가드·메모리 감시)을 먼저 설치하고 트랙 R3 모듈을 읽기 전용 import 한다."""
import sys, os, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard16 as G
from c16common import *
sys.path.insert(0, str(R3)); os.chdir(str(T1))
A15 = C15/'cache'/'t1'/'A'/'o2019'
MD = CC/'models'; MD.mkdir(exist_ok=True)
P0D = CC/'p0'; P0D.mkdir(exist_ok=True)
