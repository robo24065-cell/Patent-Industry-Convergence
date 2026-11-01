import sys, time, soc, t4lib as T
T.load()
for L in sys.argv[1:]:
    t=time.time(); s=soc.soc_features(L); print(L, s.shape, round(time.time()-t,1), flush=True)
