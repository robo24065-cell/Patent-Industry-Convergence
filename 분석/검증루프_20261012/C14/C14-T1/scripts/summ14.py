# -*- coding: utf-8 -*-
r"""집계: E1·E2(C10 npz), E3(선형화), E4·E4b(성분 부트스트랩) → se_compare.csv(폴더 루트), res/verdict.json"""
import sys, glob
sys.dont_write_bytecode = True
from lib14 import *
from scipy.stats import t as tdist

D, G = build_data()
BT = np.load(os.path.join(CACHE, "boot_c10_all.npz"))
P0 = json.load(open(os.path.join(RES, "p0_repro.json"), encoding="utf-8"))
E3 = json.load(open(os.path.join(RES, "e3.json"), encoding="utf-8"))
ST = json.load(open(os.path.join(OUT, "selftest.json"), encoding="utf-8"))
PSI = np.load(os.path.join(CACHE, "psi14.npz"))


def load_e4(mode):
    fs = glob.glob(os.path.join(CACHE, f"e4_{mode}_*.npz"))
    parts = sorted([(int(os.path.basename(f)[:-4].split("_")[-2]), np.load(f)["auc"]) for f in fs], key=lambda x: x[0])
    pos = 0
    for lo_, A_ in parts:
        assert lo_ == pos, (mode, [p[0] for p in parts]); pos += len(A_)
    assert pos == 1000, (mode, pos)
    A = np.concatenate([p[1] for p in parts]); assert not np.isnan(A).any()
    return A


E4B = {m: load_e4(m) for m in ("cell_fs", "cell_fk", "add_fs", "add_fk")}
STR = {"fs": ("firm", "sc", "cell_fs", "firm_sc"), "fk": ("firm", "ksic_t", "cell_fk", "firm_ksic_t")}
AUC = {(c, a): placement(D[c]["y"], D[c][a])[0] for c in C_ORIG for a in ARMS}
rows = []; V = {}


def add(est, s, pair, origin, pt, se, lo, hi, note=""):
    rows.append({"estimator": est, "structure": {"fs": "법인×서브클래스", "fk": "법인×목표KSIC", "-": "단일(법인)"}[s], "pair": pair, "origin": origin,
                 "point": pt, "se": se, "lo": lo, "hi": hi, "lo_gt0": bool(lo > 0), "note": note})


for a, b in PAIRS:
    ia, ib = ARMS.index(a), ARMS.index(b); pair = f"{a}-{b}"
    for oi, origin in enumerate(C_ORIG + ["pooled"]):
        sel = slice(None) if origin == "pooled" else [oi]
        cs = C_ORIG if origin == "pooled" else [origin]
        pt = float(np.mean([AUC[(c, a)] - AUC[(c, b)] for c in cs]))
        dif = lambda A: (A[:, ia, :][:, sel] - A[:, ib, :][:, sel]).mean(1)
        # E1
        d1 = dif(BT["firm"]); add("E1 법인 Poisson B=2000(공식)", "-", pair, origin, pt, d1.std(ddof=1), *np.percentile(d1, [2.5, 97.5]), "백분위")
        for s, (A_, B_, AB_, clt) in STR.items():
            d2 = dif(BT[clt]); v2 = d2.var(ddof=1)
            add("E2 곱 가중 Poisson B=2000(C10)", s, pair, origin, pt, np.sqrt(v2), *np.percentile(d2, [2.5, 97.5]), "백분위")
            # E3 선형화
            comp = {}
            for k in (A_, B_, AB_):
                S_ = np.zeros(G[k]); cnt = np.zeros(G[k])
                for c in cs:
                    S_ += np.bincount(D[c][k], weights=(PSI[f"{c}_{a}"] - PSI[f"{c}_{b}"]) / len(cs), minlength=G[k]); cnt += np.bincount(D[c][k], minlength=G[k])
                m = int((cnt > 0).sum()); comp[k] = (m / (m - 1) * (S_ ** 2).sum(), m)
            v3 = comp[A_][0] + comp[B_][0] - comp[AB_][0]; fix3 = v3 <= 0
            if fix3: v3 = max(comp[A_][0], comp[B_][0])
            df = min(comp[A_][1], comp[B_][1]) - 1; q = tdist.ppf(0.975, df)
            add("E3 CGM 선형화", s, pair, origin, pt, np.sqrt(v3), pt - q * np.sqrt(v3), pt + q * np.sqrt(v3), f"t df={df}" + (" 음수보정" if fix3 else ""))
            # E4 / E4b
            vc = dif(E4B[AB_]).var(ddof=1); va = dif(E4B["add_" + s]).var(ddof=1)
            v4 = v2 - 2 * vc; fix4 = v4 <= 0
            if fix4: v4 = max(comp[A_][0], comp[B_][0])
            v4b = 4 * va - vc; fix4b = v4b <= 0
            if fix4b: v4b = max(comp[A_][0], comp[B_][0])
            add("E4 보정 pigeonhole(Var_E2−2V_AB)", s, pair, origin, pt, np.sqrt(v4), pt - q * np.sqrt(v4), pt + q * np.sqrt(v4),
                f"t df={df}; V_AB,boot={vc:.3e}(B=1000)" + (" 음수보정" if fix4 else ""))
            add("E4b 가산 가중(4Var−V_AB)", s, pair, origin, pt, np.sqrt(v4b), pt - q * np.sqrt(v4b), pt + q * np.sqrt(v4b),
                f"t df={df}" + (" 음수보정" if fix4b else ""))
            if origin == "pooled":
                V[(pair, s)] = {"v2": v2, "vcell_boot": vc, "vadd4_boot": 4 * va, "vAB_lin": comp[AB_][0], "vA_lin": comp[A_][0], "vB_lin": comp[B_][0],
                                "v3": v3, "v4": v4, "v4b": v4b}
SC = pd.DataFrame(rows); SC.to_csv(safe_path(os.path.join(OUT, "se_compare.csv")), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
print(SC[SC.origin == "pooled"].round(6).to_string())

# ---------- 판정
PASS = ST["pass_both"]
VER = {"selftest_pass": PASS, "components": {f"{k[0]}|{k[1]}": v for k, v in V.items()}}
for a, b in PAIRS:
    pair = f"{a}-{b}"; VER[pair] = {}
    for s in STR:
        g = SC[(SC.pair == pair) & (SC.origin == "pooled") & (SC.structure == {"fs": "법인×서브클래스", "fk": "법인×목표KSIC"}[s])]
        get = lambda e: g[g.estimator.str.startswith(e + " ")].iloc[0]
        e3, e4, e2 = get("E3"), get("E4"), get("E2")
        e3_ok = PASS["E3"]; e4_ok = PASS["E4"]
        sd = E3[pair][s]
        st_over = any(ST[S_]["est"]["E2"]["coverage"] > 0.97 for S_ in ("S1", "S2"))
        st_share = min([ST[S_]["vAB_share_mean"] for S_ in ("S1", "S2") if ST[S_]["est"]["E2"]["coverage"] > 0.97] or [np.inf])
        diag = {"①vAB>0&share≥0.10": bool(sd["vAB"] > 0 and sd["vAB_share"] >= 0.10),
                "②E2boot/lin(VA+VB+VAB) in 0.8–1.25": bool(0.8 <= sd["ratio_boot_e2_to_lin_pigeon"] <= 1.25),
                "③selftest E2 과대 & (share≥ 그 구조 or SE비≥1.15)": bool(st_over and (sd["vAB_share"] >= st_share or e2.se / e3.se >= 1.15))}
        diag_ok = all(diag.values())
        used = [x for x, ok in (("E3", e3_ok), ("E4", e4_ok)) if ok]
        lo_pos = {"E3": bool(e3.lo > 0), "E4": bool(e4.lo > 0)}
        if not used:
            v = "판정 불가(E3·E4 자기 검증 실패)"
        elif all(lo_pos[x] for x in used) and diag_ok:
            v = "(i) 원고 단서 오류(과도)" + ("" if len(used) == 2 else f" — 자기 검증 통과 추정량({','.join(used)})만으로 판정, 나머지 판정 불가")
        elif all(not lo_pos[x] for x in used) and e2.lo <= 0:
            v = "(iii) 단서 확인" + ("" if len(used) == 2 else f" — {','.join(used)}만으로 판정")
        elif any(not lo_pos[x] for x in used):
            v = "(ii) 경계(단서 유지·문구 정밀화)" + ("" if len(used) == 2 else f" — {','.join(used)}만으로 판정")
        else:
            v = "(ii) 경계(단서 유지·문구 정밀화) — 구조 진단 또는 자기 검증 조건 미충족"
        VER[pair][s] = {"verdict": v, "E2": [e2.lo, e2.hi], "E3": [e3.lo, e3.hi], "E4": [e4.lo, e4.hi], "E4b": [get("E4b").lo, get("E4b").hi],
                        "se": {"E1": float(SC[(SC.pair == pair) & (SC.origin == "pooled") & SC.estimator.str.startswith("E1")].se.iloc[0]),
                               "E2": e2.se, "E3": e3.se, "E4": e4.se, "E4b": get("E4b").se},
                        "diag": diag, "used": used}
json.dump(VER, open(safe_path(os.path.join(RES, "verdict.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
for a, b in PAIRS:
    print(f"{a}-{b}", {s: VER[f"{a}-{b}"][s]["verdict"] for s in STR})
