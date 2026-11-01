"""E00: 1라운드 동결 v30을 2라운드 개발 원점(2010..2016)에서 재채점 → 출발 챔피언"""
import r2core as C, time
t=time.time(); D=C.get_D(); print('D', D.shape, D.groupby('t0').size().to_dict(), round(time.time()-t))
cfg=C.v30_cfg(); res=C.evaluate(cfg,D)
C.record('E00','-','출발 챔피언 = 1라운드 동결 v30 (로지스틱39+XGB77+음이항77 순위평균) 재채점',res,'CHAMPION','기준선 = kr_mom 단독 로지스틱(고정)')
C.save_champ('E00',cfg,res)
for v in res: print(v, {k:round(x,4) for k,x in res[v].items()})
print(C.summarize(res), round(time.time()-t))
