"""채택 변경 조합 점검(INFO): 최종 챔피언에서 채택 변경을 하나씩 빼 본다."""
import copy, r2core as C, exps as X
D=C.get_D(); ch=C.load_champ()['cfg']
def drop_nbwide(cfg):
    c2=copy.deepcopy(cfg); X.comp(c2,'nb').pop('train_min_total',None); return c2
A={'A1':('[조합점검] 챔피언 − 생산지수 6피처(E18)', X.feats(drop=X.PI)),
   'A2':('[조합점검] 챔피언 − 음이항 확대 표본(E38)', drop_nbwide),
   'A3':('[조합점검] 챔피언 − 음이항 600라운드(E43, 400으로)', X.set_param('nb',rounds=400)),
   'A4':('[조합점검] 챔피언 − E38 − E43 (생산지수만 남김)', X.chain(drop_nbwide, X.set_param('nb',rounds=400)))}
for k,(chg,f) in A.items(): C.challenge(k,chg,f(ch),'채택 변경의 한계기여 확인(판정 없음)',decide=False,D=D)
