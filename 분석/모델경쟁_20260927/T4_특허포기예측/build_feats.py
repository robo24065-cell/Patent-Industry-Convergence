import sys, time, t4lib as T
Ls=sys.argv[1:]
T.load()
for L in Ls:
    t=time.time(); f=T.features(L, use_cache=False); print(L, f.shape, round(time.time()-t,1), flush=True)
