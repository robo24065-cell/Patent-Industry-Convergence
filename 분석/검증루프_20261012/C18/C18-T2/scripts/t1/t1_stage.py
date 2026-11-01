# -*- coding: utf-8 -*-
r"""C18-T2 T1 동결 학습 단계 실행(공식 R3/final_run_r3.py 의 train_origin 을 그대로 호출, FC·RD 만 이 폴더로). 종료코드 3 = 다음 단계 있음.
 사용: python t1_stage.py P0|V1E"""
import os, sys, importlib.util
sys.dont_write_bytecode = True
VAR = sys.argv[1]
W = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C18\C18-T2'
R3 = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\T1_산업대기실\R3'
import pathlib
FC = pathlib.Path(W) / 'cache' / 't1' / VAR / 'FC'; RD = pathlib.Path(W) / 'cache' / 't1' / VAR / 'RD'
sys.argv = ['final_run_r3.py']
spec = importlib.util.spec_from_file_location('frr3', os.path.join(R3, 'final_run_r3.py')); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.train_origin(2019, FC, 'prep_tr.parquet', RD, m.EC)
print('train_origin complete', VAR, flush=True)
