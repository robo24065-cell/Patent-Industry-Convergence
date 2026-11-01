# -*- coding: utf-8 -*-
"""R5 레드팀(N4) 1단계: 동결 n4lib 함수로 기초표를 원자료에서 새로 만든다(원본 cache 미사용, 새 폴더)."""
import sys, os, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import n4lib
n4lib.CACHE = os.path.join(HERE, "cache")
t = time.time()
R = n4lib.load_registry(); print("registry", {k: len(v) for k, v in R.items()}, round(time.time() - t), flush=True)
N = n4lib.load_nps("final"); print("nps", len(N), round(time.time() - t), flush=True)
AF = n4lib.load_app_firm(); print("app_firm", len(AF), round(time.time() - t), flush=True)
ow, sur = n4lib.load_registry_extra(); print("own/sur", len(ow), len(sur), round(time.time() - t), flush=True)
AFd = AF[(AF.applicantCd.astype(str).str[0] == "1") & AF.known_date.notna()]
ev = n4lib.load_foreign_events(AFd); print("foreign", len(ev), round(time.time() - t), flush=True)
