import t3lib as L, time
t=time.time()
for t0 in range(2005,2016):
    d=L.get_feats(t0); print(t0,len(d),round(time.time()-t,1))
