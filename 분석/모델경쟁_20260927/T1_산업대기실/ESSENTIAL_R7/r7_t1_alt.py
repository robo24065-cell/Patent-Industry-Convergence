# -*- coding: utf-8 -*-
"""T1 R7 대체 시험 S_TXT 입력 생성: 제공 KSIC 라벨 대신 공식 IPC-KSIC 연계표(2013) 배정으로 제목→산업 분류기를 다시 학습.
- 학습 라벨: 범위안 등록특허(cache/ins_pat, 등록 ≤ cap)의 메인 IPC → 연계표 소분류(제공 라벨공간 25개 안 argmax, 'in'). 미배정 IPC는 학습 제외.
- 연계표는 세세분류(5자리)를 낼 수 없으므로 25개 소분류 확률을 같은 소분류의 5자리 코드에 그대로 복사(50열).
- 분류기 5종(범위밖 50-class, open-set 51, 범위안 2차 산업 교차적합, 디자인권, 범위밖 실용신안)을 동결 코드와 같은 벡터라이저·SGD 설정으로 다시 학습.
- 패널 피처는 1라운드 동결 함수 fr.cut_features를 그대로 호출(분류기 행렬만 대체, 결과창 h=0으로 레이블 미계산) → 비텍스트 열이 동결 캐시와 같은지 대조.
- prep_tr/prep_te: 동결 캐시(R3/cache/o<c>, 읽기만)의 행·레이블·가중을 그대로 두고 분류기 유래 열만 교체.
산출(쓰기는 이 폴더만): alt_txt/o<c>/{P_base,P_os51,PO_os51,P_ins_cf,P_design,P_um}.npy, feat_c<t>.parquet, prep_tr.parquet, prep_te.parquet, check.json
사용: python r7_t1_alt.py <원점>   (종료코드 3 = 단계 종료 → 재실행)"""
import os, sys, json, time, pathlib
HERE = pathlib.Path(__file__).resolve().parent; T1 = HERE.parent; R3 = T1/'R3'
N9E = T1.parent/'N9_정부RnD과제_특허성과예측'/'ESSENTIAL_R7'
sys.path.insert(0, str(R3)); sys.path.insert(0, str(N9E))
os.chdir(str(T1))
import numpy as np, pandas as pd
import r3
from r3 import fr, lib, r2, CACHE, KEEP, feat_path, fc_r3, cfg_of, f32
from r7_concord import assign, G50

TXT_COLS = ['t_mean', 't_hi', 't_lsum', 't_max', 't_n20', 't_top3', 't_rank', 't_ent', 't_rw', 't_rec3', 't_mom', 't_lrec_n', 't_grow', 't_nbr',
            'ot_mean', 'ot_max', 'ot_top3', 'ot_hi', 'ot_rw', 'ot_nbr', 'oin50', 'tsib3', 'tsib2', 'ti_mean', 'ti_max', 'ti_n20', 'ti_lsum',
            't_rrw', 't_rrec2', 't_rrec2_max']
T0 = time.time()
def el(): return time.time() - T0
def stop(msg): print('[단계 종료]', msg, round(el()), flush=True); sys.exit(3)


def vec_clf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    return (TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=5, max_features=400000, sublinear_tf=True, dtype=np.float32),
            SGDClassifier(loss='log_loss', alpha=2e-6, max_iter=12, tol=None, random_state=0))


_LAB = {}
def ins_alt():
    """ins_pat + 연계표 소분류 라벨(alt)"""
    if 'k' in _LAB: return _LAB['k']
    kr = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['rgst_no', 'title', 'ksic', 'rgst_date', 'mipc'])
    g, _ = assign(kr.mipc.values, 'in'); kr['alt'] = g
    _LAB['k'] = kr; return kr


def bmat():
    """25 소분류 → 50 코드 복사 행렬 (열 순서 = lib K)"""
    K = lib.load()['K']; return K, {g: [i for i, k in enumerate(K) if k[:3] == g] for g in G50}


def broadcast(pr, classes):
    K, M = bmat(); out = np.zeros((pr.shape[0], 50), np.float16)
    for j, g in enumerate(classes):
        if g in M:
            for i in M[g]: out[:, i] = pr[:, j]
    return out


def clf_base_alt(cf, FC):
    p = FC/'P_base.npy'
    if p.exists(): return
    D = lib.load(); oos = D['oos']; kr = ins_alt()
    tr = kr[(kr.rgst_date <= cf['cap']) & (kr.alt != '')].dropna(subset=['title'])
    vec, clf = vec_clf(); clf.fit(vec.fit_transform(tr.title), tr.alt); cls = list(clf.classes_)
    P = np.zeros((len(oos), 50), np.float16)
    for s in range(0, len(oos), 200000):
        P[s:s + 200000] = broadcast(clf.predict_proba(vec.transform(oos.title.iloc[s:s + 200000].fillna(''))), cls)
    np.save(p, P); stop('clf_base_alt')


def clf_os51_alt(cf, FC):
    pP, pO = FC/'P_os51.npy', FC/'PO_os51.npy'
    if pP.exists(): return
    D = lib.load(); oos = D['oos']; kr = ins_alt()
    tr_in = kr[(kr.rgst_date <= cf['cap']) & (kr.alt != '')].dropna(subset=['title'])
    fold = (pd.util.hash_array(oos.applicant_cd.values.astype(object)) % 2).astype(int)
    for fo in (0, 1):
        fp = FC/f'os51_fold{fo}.npz'
        if fp.exists(): continue
        oth = oos[(fold != fo) & (oos.rgst_date <= cf['cap'])]
        titles = pd.concat([tr_in.title, oth.title.fillna('')]); ys = np.concatenate([tr_in.alt.values, np.array(['OTHER'] * len(oth))])
        vec, clf = vec_clf(); clf.fit(vec.fit_transform(titles), ys, sample_weight=np.ones(len(ys)))
        cls = list(clf.classes_); oi = cls.index('OTHER')
        idx = np.nonzero(fold == fo)[0]; Pf = np.zeros((len(idx), 50), np.float16); POf = np.zeros(len(idx), np.float16)
        for s in range(0, len(idx), 200000):
            pr = clf.predict_proba(vec.transform(oos.title.iloc[idx[s:s + 200000]].fillna('')))
            Pf[s:s + 200000] = broadcast(pr, cls); POf[s:s + 200000] = pr[:, oi].astype(np.float16)
        np.savez(fp, P=Pf, PO=POf); stop(f'os51 fold {fo}')
    P = np.zeros((len(oos), 50), np.float16); PO = np.zeros(len(oos), np.float16)
    for fo in (0, 1):
        z = np.load(FC/f'os51_fold{fo}.npz'); idx = np.nonzero(fold == fo)[0]; P[idx] = z['P']; PO[idx] = z['PO']
    np.save(pP, P); np.save(pO, PO)


def clf_ins_alt(cf, FC):
    p = FC/'P_ins_cf.npy'
    if p.exists(): return
    kr = ins_alt().copy(); kr['title'] = kr.title.fillna('')
    fold = (pd.util.hash_array(kr.rgst_no.values.astype(object)) % 2).astype(int)
    P = np.zeros((len(kr), 50), np.float16)
    for fo in (0, 1):
        tr = kr[(fold != fo) & (kr.rgst_date <= cf['cap']) & (kr.alt != '')]
        vec, clf = vec_clf(); clf.fit(vec.fit_transform(tr.title), tr.alt); cls = list(clf.classes_)
        idx = np.nonzero(fold == fo)[0]
        P[idx] = broadcast(clf.predict_proba(vec.transform(kr.title.iloc[idx])), cls)
    np.save(p, P); kr[['rgst_no']].to_parquet(FC/'P_ins_cf_index.parquet', index=False); stop('clf_ins_alt')


def clf_ext_alt(cf, FC):
    kinds = [k for k in ('design', 'um') if not (FC/f'P_{k}.npy').exists()]
    if not kinds: return
    kr = ins_alt(); tr = kr[(kr.rgst_date <= cf['cap']) & (kr.alt != '')].dropna(subset=['title'])
    vec, clf = vec_clf(); clf.fit(vec.fit_transform(tr.title), tr.alt); cls = list(clf.classes_)
    for k in kinds:
        s = pd.read_parquet(R3/'cache'/f'src_{k}.parquet'); s = s[s.rgst_date <= cf['cap']].reset_index(drop=True)
        P = np.zeros((len(s), 50), np.float16)
        for i in range(0, len(s), 200000): P[i:i + 200000] = broadcast(clf.predict_proba(vec.transform(s.title.iloc[i:i + 200000])), cls)
        np.save(FC/f'P_{k}.npy', P); s[['applicant_cd', 'rgst_no', 'ay', 'rgst_date']].to_parquet(FC/f'P_{k}_index.parquet', index=False)
    stop('clf_ext_alt')


def feats(c, cf, FC):
    cuts = list(cf['train_cuts']) + [c]
    todo = [t for t in cuts if not (FC/f'feat_c{t}.parquet').exists()]
    if not todo: return
    _bp = lib.build_panel
    fr.build_panel = lambda cut, h=5, cap=None, **kw: _bp(cut, 0, cap=cap, **kw)     # 결과창 0: 레이블 미계산(피처만)
    P = np.load(FC/'P_base.npy').astype(np.float32); P51 = np.load(FC/'P_os51.npy').astype(np.float32); PO = np.load(FC/'PO_os51.npy').astype(np.float32)
    Pins = np.load(FC/'P_ins_cf.npy'); ins_idx = pd.read_parquet(FC/'P_ins_cf_index.parquet').rgst_no
    src = feat_path(c, c).parent if feat_path(c, c) is not None else fc_r3(c)
    ep = pd.read_parquet(r3.fc_r2(c)/'ep_firm.parquet') if (r3.fc_r2(c)/'ep_firm.parquet').exists() else pd.read_parquet(fc_r3(c)/'ep_firm.parquet')
    cdf = fr.lag_cdf(cf)
    t = todo[0]
    fr.cut_features(t, None if t == c else cf['cap'], cf, FC, P, P51, PO, Pins, ins_idx, ep, cdf)
    stop(f'feat c{t}')


def assemble(c, cf, FC):
    if (FC/'prep_tr.parquet').exists() and (FC/'prep_te.parquet').exists(): return
    chk = {}
    for kind, cuts in (('tr', list(cf['train_cuts'])), ('te', [c])):
        orig = pd.read_parquet(fc_r3(c)/f'prep_{kind}.parquet')
        alt = pd.concat([pd.read_parquet(FC/f'feat_c{t}.parquet', columns=['firm', 'k', 'cut'] + TXT_COLS + ['lin', 'dens', 'sib4', 'ep_k', 'pa_k', 'p_k', 'now1']) for t in cuts], ignore_index=True)
        assert len(alt) == len(orig) and (alt.firm.values == orig.firm.values).all() and (alt.k.values == orig.k.values).all() and (alt.cut.values == orig.cut.values).all(), '행 정렬 불일치'
        same = {cc: float(np.nanmax(np.abs(alt[cc].astype(np.float64).values - orig[cc].astype(np.float64).values))) for cc in ['lin', 'dens', 'sib4', 'ep_k', 'pa_k', 'p_k', 'now1']}
        diff = {cc: float(np.corrcoef(np.nan_to_num(alt[cc].astype(np.float64).values), np.nan_to_num(orig[cc].astype(np.float64).values))[0, 1]) for cc in ['t_top3', 't_lsum', 't_max', 'ot_top3', 'ti_max', 't_rrec2_max']}
        chk[kind] = dict(rows=int(len(orig)), nontext_max_abs_diff=same, text_corr_alt_vs_provided=diff)
        for cc in TXT_COLS:
            orig[cc] = alt[cc].astype(orig[cc].dtype).values
        orig.to_parquet(FC/f'prep_{kind}.parquet', index=False); del orig, alt
    json.dump(chk, open(FC/'check.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(chk, ensure_ascii=False))


def main(c):
    cf = cfg_of(c); FC = HERE/'alt_txt'/f'o{c}'; FC.mkdir(parents=True, exist_ok=True)
    lib.load()   # 범위밖 제목 표(oos) 포함 전체 적재(분류기 채점·패널용)
    clf_base_alt(cf, FC); clf_os51_alt(cf, FC); clf_ins_alt(cf, FC); clf_ext_alt(cf, FC)
    feats(c, cf, FC); assemble(c, cf, FC)
    print('ALTDONE', c, round(el()), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
