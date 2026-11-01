# -*- coding: utf-8 -*-
"""C17-T1 트랙 import 공용: guard17(쓰기 가드·메모리 감시)을 먼저 설치하고 트랙 R3 모듈을 읽기 전용 import 한다(C16-T1 c16t1.py와 같은 방식)."""
import sys, os, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
sys.path.insert(0, str(R3)); os.chdir(str(T1))
MD = C16C/'models'          # C16-T1 재적합 모형(공식 FROZEN과 비트 일치 확인됨), 읽기 전용
P0D = C16C/'p0'
