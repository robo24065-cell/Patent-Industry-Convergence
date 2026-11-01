# -*- coding: utf-8 -*-
"""fig02~fig09 (데이터 그림) v3: 최종 인쇄 크기(폭 17cm, 높이 cm 지정, 300dpi)로 그린다.
원자료·트랙 폴더는 읽기만 한다. 그림 제목·번호·출처는 PNG에 넣지 않는다(문서 캡션에서 붙임).
그림 안 글자는 모두 7pt 이상(_style.MINPT). 값은 v2와 같은 원천·같은 계산이다(v2 사본: _v2/make_data_figs.py).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from _style import *  # noqa

# ---------------- KSIC 이름 ----------------
prof = pd.read_csv(os.path.join(FA, "map5", "m2_ksic_profile.csv"), encoding="utf-8-sig")


def short(n):
    n = str(n)
    for s in [" 제조업", "제조업"]:
        if n.endswith(s):
            n = n[: -len(s)]
    return n.strip()


NAME = {int(k): short(n) for k, n in zip(prof.ksic, prof.name)}

# 인쇄 크기에서 한 줄에 들어가도록 줄인 표시용 이름(값과 무관, 산업 식별용).
SHORTNAME = {
    24191: "표면 처리 강재", 31312: "무인 항공기·비행장치", 20491: "감광 재료·화학제품",
    33309: "기타 운동·경기용구", 33301: "체력 단련용 장비", 26111: "메모리용 집적회로",
    26223: "연성·기타 인쇄회로기판", 28902: "전기용 탄소·절연제품", 29132: "기체 펌프·압축기",
    27301: "광학 렌즈·광학 요소", 29133: "탭·밸브·유사 장치", 20202: "합성수지·플라스틱 물질",
    20411: "일반용 도료", 29150: "산업용 오븐·노·버너", 27211: "레이더·항행·측량 기구",
    26112: "비메모리 집적회로", 25924: "절삭 가공·유사 처리", 26519: "비디오·영상 기기",
    27212: "전자기 측정·분석 기구", 25914: "그 외 금속 압형제품", 28111: "전동기·발전기",
    24219: "비철금속 제련·합금", 29241: "건설·채광용 기계장비", 31202: "철도 차량 부품",
    27199: "그 외 기타 의료용 기기", 29210: "농업·임업용 기계", 30393: "자동차용 신품 의자",
}


def sname(k):
    return SHORTNAME.get(int(k), NAME[int(k)])


def ftext(fig, x, y, s, **kw):
    """cm 좌표(그림 왼쪽 아래 기준)로 글자를 놓는다."""
    fw, fh = fig.get_size_inches() / CM
    return fig.text(x / fw, y / fh, s, **kw)


def ycm(fig, ax, yd):
    """축 데이터 y → 그림 cm."""
    fh = fig.get_size_inches()[1] / CM
    return ax.transData.transform((0, yd))[1] / fig.bbox.height * fh


# =========================================================
# fig02. 9개 트랙 공식 성적표 (R6 §3.1·§3.2) - 17 × 8.4 cm
# =========================================================
def fig02():
    rows = [
        # stage, track, name, auc, base, skill, lo, hi, cons_label, cons_val, grade
        ("뜬다", "T3", "기술셀 부상", 0.794, 0.642, 0.423, 0.366, 0.473, "KR이력 XGB 대비", "0.182", "견고"),
        ("들어온다", "T1", "산업 대기실", 0.900, 0.811, 0.473, 0.454, 0.488, "고정 엄격 기준선 대비", "0.395", "견고"),
        ("다툰다", "N8", "특허 분쟁 표적", 0.870, 0.780, 0.409, 0.243, 0.540, "강한 기준선(SB) 대비", "0.312", "견고"),
        ("버려진다", "T4", "특허 포기", 0.701, 0.604, 0.246, 0.133, 0.320, "13변수 로지스틱 대비", "0.124", "조건부"),
        ("쓰러진다", "N4", "기업 폐업", 0.803, 0.760, 0.177, 0.147, 0.209, "NPS 키 보정", "0.140", "하향 보정 후 유지"),
        ("표준이 된다", "N5", "표준특허 선언", 0.858, 0.830, 0.165, 0.007, 0.360, "최근 선언률 기준선 대비", "0.000", "조건부"),
        ("특허↔R&D", "N9", "R&D 과제 특허 성과", 0.894, 0.826, 0.392, 0.362, 0.420, "강한 기준선 지연판 대비", "0.356", "견고"),
        ("특허↔R&D", "N6", "R&D기업 첫 특허", 0.747, 0.703, 0.149, 0.115, 0.185, "M0-ens 대비", "0.142", "견고"),
        ("특허↔R&D", "N7", "특허기업 R&D 첫 수행", 0.746, 0.645, 0.286, 0.211, 0.364, "NPS 키 보정", "0.251", "견고(하향 보정 후)"),
    ]
    df = pd.DataFrame(rows, columns=["단계", "트랙", "이름", "AUC_모형", "AUC_기준선", "skill", "skill_CI_lo",
                                     "skill_CI_hi", "보수적_기준", "보수적_skill", "검증등급"])
    df["출처"] = "R6_결과_20261004.md §3.1(공식)·§3.2(보수적)"
    csv(df, "fig02_scorecard_9tracks")

    n = len(df)
    fig = figure(8.4)
    ax = cm_axes(fig, 5.05, 0.95, 4.15, 6.45)
    y = np.arange(n)[::-1]
    for yi, r in zip(y, df.itertuples()):
        ax.plot([r.AUC_기준선, r.AUC_모형], [yi, yi], color=G3, lw=2.2, solid_capstyle="round", zorder=1)
        ax.scatter(r.AUC_기준선, yi, s=26, color=G2, zorder=2, edgecolor=WHITE, linewidth=0.8)
        cond = r.검증등급 == "조건부"
        ax.scatter(r.AUC_모형, yi, s=32, zorder=3, linewidth=1.2,
                   facecolor=WHITE if cond else ACC, edgecolor=ACC)
        ax.text(r.AUC_모형 + 0.017, yi, f"{r.AUC_모형:.3f}", va="center", fontsize=FS_S, color=INK)
        ax.text(r.AUC_기준선 - 0.014, yi, f"{r.AUC_기준선:.3f}", va="center", ha="right", fontsize=FS_S, color=G1)
    ax.set_xlim(0.52, 0.97)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0.6, 0.7, 0.8, 0.9])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("AUC (잠긴 최종시험 1회)")
    ax.grid(axis="x", color=G4, lw=0.5)
    ax.set_axisbelow(True)

    prev = None
    for yi, r in zip(y, df.itertuples()):
        yy = ycm(fig, ax, yi)
        if r.단계 != prev:
            ftext(fig, 0.05, yy, r.단계, va="center", fontsize=FS_S, color=G1)
        ftext(fig, 1.55, yy, f"{r.트랙}  {r.이름}", va="center", fontsize=FS, color=INK, fontweight="bold")
        prev = r.단계
        ftext(fig, 9.7, yy, f"{r.skill:.3f} [{r.skill_CI_lo:.3f}, {r.skill_CI_hi:.3f}]", va="center", fontsize=FS)
        ftext(fig, 12.6, yy + 0.15, f"{r.보수적_skill}", va="center", fontsize=FS)
        ftext(fig, 12.6, yy - 0.15, f"{r.보수적_기준}", va="center", fontsize=FS_S, color=G1)
        strong = r.검증등급.startswith("견고")
        ftext(fig, 16.95, yy, r.검증등급.replace("(하향 보정 후)", "\n(하향 보정 후)").replace("하향 보정 후 유지", "하향 보정 후\n유지"),
              va="center", ha="right", multialignment="right", fontsize=FS_S, linespacing=1.15,
              color=ACC if strong else G1, fontweight="bold" if strong else "normal")
    top = ycm(fig, ax, n - 0.4) + 0.12
    hk = dict(fontsize=FS_S, color=G1, fontweight="bold", va="bottom")
    ftext(fig, 0.05, top, "단계", **hk)
    ftext(fig, 1.55, top, "트랙", **hk)
    ftext(fig, 9.7, top, "공식 skill [95% CI]", **hk)
    ftext(fig, 12.6, top, "보수적 skill (기준)", **hk)
    ftext(fig, 16.95, top, "검증 등급", ha="right", **{k: v for k, v in hk.items()})
    h = [Line2D([], [], marker="o", ls="", color=G2, ms=4.5, label="기준선 AUC"),
         Line2D([], [], marker="o", ls="", color=ACC, ms=5, label="동결 모형 AUC"),
         Line2D([], [], marker="o", ls="", mfc=WHITE, mec=ACC, ms=5, label="동결 모형(조건부 등급)")]
    fw, fh = fig.get_size_inches() / CM
    fig.legend(handles=h, loc="center left", bbox_to_anchor=(6.6 / fw, (top + 0.55) / fh), ncol=3, fontsize=FS_S,
               handletextpad=0.3, columnspacing=1.0, borderaxespad=0, borderpad=0)
    ftext(fig, 0.05, top + 0.55, "skill = (AUC모형 - AUC기준선) / (1 - AUC기준선)", va="center", fontsize=FS_S, color=G1)
    return save(fig, "fig02_scorecard_9tracks")


# =========================================================
# fig03. 필수성 근거 A: 기업 업종 일치 (R12 사전 등록 확인 시험, R10 §2.2 순환 점검) - 17 × 7.2 cm
# =========================================================
def _r(v, nd=1):
    """저장된 소수 4자리 값을 %p로 바꿔 반올림(사사오입). 원고 표기와 같게 한다."""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(str(round(float(v), 4))) * 100
    return float(q.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def fig03():
    R12 = os.path.join(SR, "label_origin_R10", "confirm_R12")
    m = pd.read_csv(os.path.join(R12, "res_main.csv"), encoding="utf-8-sig").set_index("기준")
    p1 = pd.read_csv(os.path.join(SR, "label_origin_R10", "res_p1.csv"), encoding="utf-8-sig")
    p1 = p1[(p1["패밀리"] == "가족 다수결") & (p1["가중"] == "기업평균")]

    recs = []
    for base in ["NPS5", "DART5"]:
        d = m.loc[base]
        recs.append(dict(패널="(a) R12 확인 시험", 기준=base, 기업수=d["기업"], 특허수=d["특허"], BEST후보=d["BEST"],
                         제공=d["PROV"], 공개최강=d["BEST값"], FREE_IPC기본=d["BASE(H-sg-3)"], TXT_FREE=d["T0(TXT-FREE)"],
                         연계표기대값=d["CWEQ"], Δ제공_BEST=d["PROV-BEST"], lo=d["PROV-BEST_lo"], hi=d["PROV-BEST_hi"],
                         Δ제공_BEST_특허가중=d["PROV-BEST_w"], Δ제공_TXTFREE=d["PROV-T0"], lo_t=d["PROV-T0_lo"], hi_t=d["PROV-T0_hi"],
                         출처="_shared_report/label_origin_R10/confirm_R12/res_main.csv (평가 표본 법인 KR 2015-19, 표 <=2014, 기업평균)"))
    for _, d in p1.iterrows():
        recs.append(dict(패널="(b) 순환 점검", 기준=d["기준"], 기업수=d["기업"], 특허수=d["특허"],
                         KR라벨=d["KR라벨"], 해외라벨=d["해외라벨"], TXT_FREE=d["TXTFREE"],
                         ΔKR_해외=d["KR-해외"], lo=d["KR-해외_lo"], hi=d["KR-해외_hi"],
                         Δ해외_FREE=d["해외-FREE"], lo2=d["해외-FREE_lo"], hi2=d["해외-FREE_hi"],
                         출처="_shared_report/label_origin_R10/res_p1.csv (가족 다수결, 기업평균); R10 §2.2"))
    out = pd.DataFrame(recs)
    csv(out, "fig03_necessity_firm_industry_match")
    NL = chr(10)

    fig = figure(7.2)
    # (a)
    ax = cm_axes(fig, 1.2, 0.95, 8.9, 5.55)
    labs = ["제공 KSIC 라벨", "공개 최강(개발 표본 1위)", "해설서+IPC 최빈 기본(FREE-IPC)",
            "해설서 단독(TXT-FREE)", "연계표(그룹 안 기대값)"]
    cols = [ACC, G1, G2, G3, G4]
    w = 0.155
    a = out[out["패널"] == "(a) R12 확인 시험"].set_index("기준")
    for gi, base in enumerate(["NPS5", "DART5"]):
        r = a.loc[base]
        vals = [r.제공, r.공개최강, r.FREE_IPC기본, r.TXT_FREE, r.연계표기대값]
        for j, v in enumerate(vals):
            x = gi + (j - 2) * (w + 0.015)
            ax.bar(x, v * 100, width=w, color=cols[j], edgecolor=WHITE if j < 4 else G3, linewidth=0.6,
                   label=labs[j] if gi == 0 else None)
            ax.text(x, v * 100 + 1.2, f"{_r(v):.1f}", ha="center", va="bottom", fontsize=FS_S,
                    color=INK if j < 2 else G1, fontweight="bold" if j < 2 else "normal")
        top = max(vals) * 100
        ax.text(gi, top + 8, "제공 - 공개 최강" + NL + f"{_r(r.Δ제공_BEST):+.1f}%p [{_r(r.lo):+.1f}, {_r(r.hi):+.1f}]",
                ha="center", va="bottom", fontsize=FS, color=ACC, fontweight="bold", linespacing=1.15)
        ax.text(gi, top + 22, "(보조) 제공 - TXT-FREE" + NL + f"{_r(r.Δ제공_TXTFREE):+.1f}%p [{_r(r.lo_t):+.1f}, {_r(r.hi_t):+.1f}]",
                ha="center", va="bottom", fontsize=FS_S, color=G1, linespacing=1.15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["NPS 등록업종" + NL + f"({int(a.loc['NPS5'].기업수):,}사, 공개 최강 {a.loc['NPS5'].BEST후보})",
                        "DART 업종" + NL + f"({int(a.loc['DART5'].기업수):,}사, 공개 최강 {a.loc['DART5'].BEST후보})"],
                       fontsize=FS)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel("기업 5자리 업종 일치율 (%)")
    ax.grid(axis="y", color=G4, lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=1, fontsize=FS_S, handlelength=1.0, handletextpad=0.4, labelspacing=0.3,
              bbox_to_anchor=(-0.01, 1.03), borderaxespad=0)
    ftext(fig, 0.05, 6.9, "(a) 특허 라벨과 출원 기업 실제 업종의 일치 (사전 등록 확인 시험, 기업 평균)", fontsize=FS_B,
          fontweight="bold", va="center")
    # (b)
    ax = cm_axes(fig, 11.75, 0.95, 5.15, 5.55)
    b = out[out["패널"] == "(b) 순환 점검"].set_index("기준")
    labs2 = ["KR 문서 라벨", "같은 발명 해외(US·JP·EP) 라벨", "TXT-FREE"]
    cols2 = [ACC, ACC_L, G1]
    wb = 0.22
    for gi, base in enumerate(["NPS", "DART"]):
        r = b.loc[base]
        vals = [r.KR라벨, r.해외라벨, r.TXT_FREE]
        for j, v in enumerate(vals):
            x = gi + (j - 1) * (wb + 0.03)
            ax.bar(x, v * 100, width=wb, color=cols2[j], edgecolor=WHITE, linewidth=0.6, label=labs2[j] if gi == 0 else None)
            ax.text(x, v * 100 + 1.2, f"{v*100:.1f}", ha="center", va="bottom", fontsize=FS_S, color=INK if j < 2 else G1)
        ax.text(gi, max(vals) * 100 + 8, "KR - 해외" + NL + f"{r.ΔKR_해외*100:+.2f}%p" + NL + f"[{r.lo*100:+.2f}, {r.hi*100:+.2f}]",
                ha="center", va="bottom", fontsize=FS_S, color=INK, linespacing=1.15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["NPS" + NL + f"({int(b.loc['NPS'].기업수)}사)", "DART" + NL + f"({int(b.loc['DART'].기업수)}사)"],
                       fontsize=FS)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60])
    ax.grid(axis="y", color=G4, lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=FS_S, handlelength=1.0, handletextpad=0.4, labelspacing=0.3,
              bbox_to_anchor=(-0.02, 1.03), borderaxespad=0)
    ftext(fig, 11.0, 6.9, "(b) 순환 점검: 해외 문서 라벨로 교체", fontsize=FS_B, fontweight="bold", va="center")
    return save(fig, "fig03_necessity_firm_industry_match")


# =========================================================
# fig04. 연계표 한 코드 안 산업 분화 (split34) - 17 × 8.0 cm
# =========================================================
def fig04():
    t5 = pd.read_csv(os.path.join(FA, "split34", "t5_cases.csv"), encoding="utf-8-sig")
    lg = pd.read_csv(os.path.join(FA, "split34", "fig_smallmultiples_long.csv"), encoding="utf-8-sig")
    # (b)(c)는 KR 출원과 KOSIS를 모두 같은 창(2008–09 → 2016–17 연평균)으로 맞춘다.
    # KOSIS 출하·사업체 열의 창은 s5_cases.py 4행 주석(KOSIS는 2008–09→2016–17)으로 확인.
    kr_col = "KR출원_연성장%(08-09→16-17)"
    sh_col = "KOSIS출하_연성장%"
    es_col = "KOSIS사업체_연성장%"
    sub = t5[t5["사례그룹"].isin(["C2620", "C2610"])][["사례그룹", "ksic", "산업명", kr_col, sh_col, es_col]].copy()
    sub["창"] = "2008–09→2016–17 (KR 출원·KOSIS 출하·KOSIS 사업체 공통)"
    sub["출처"] = "_shared_report/fusion_advantage/split34/t5_cases.csv"
    ts = lg[lg.ksic.isin([26211, 26212])][["grp", "ksic", "name", "year", "구간", "n_KR", "KR출원_지수2005"]].copy()
    ts["출처"] = "_shared_report/fusion_advantage/split34/fig_smallmultiples_long.csv"
    csv(pd.concat([sub.assign(표="(b)(c) 연성장"), ts.assign(표="(a) 지수")], ignore_index=True), "fig04_within_code_divergence")

    fig = figure(8.0)
    AY, AH = 1.0, 5.6
    # (a) 시계열
    ax = cm_axes(fig, 1.15, AY, 3.55, AH)
    for k, c, lab, lx, ly in [(26211, G1, "LCD", 2009.0, 33), (26212, ACC, "OLED", 2011.8, 140)]:
        d = ts[ts.ksic == k].sort_values("year")
        dev = d[d.year <= 2017]
        post = d[d.year >= 2017]
        ax.plot(dev.year, dev["KR출원_지수2005"], color=c, lw=1.6)
        ax.plot(post.year, post["KR출원_지수2005"], color=c, lw=1.3, ls=(0, (2, 1.5)))
        ax.text(lx, ly, lab, va="center", ha="center", fontsize=FS, color=c, fontweight="bold")
    ax.axvspan(2017.5, 2022.5, color=G4, alpha=0.6, lw=0)
    ax.text(2020, 186, "사후", ha="center", va="center", fontsize=FS_S, color=G1)
    ax.axhline(100, color=G3, lw=0.6)
    ax.set_xlim(2004.5, 2022.5); ax.set_ylim(0, 195)
    ax.set_xticks([2005, 2010, 2015, 2020])
    ax.set_ylabel("KR 출원 지수 (2005 = 100)")
    ax.grid(axis="y", color=G4, lw=0.5); ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)

    SHORT4 = {26211: "LCD", 26212: "OLED", 26223: "연성·기타 PCB", 26224: "부품 실장기판", 26291: "전자 축전기",
              26111: "메모리", 26112: "비메모리", 26121: "LED", 26129: "기타 반도체 소자"}

    def fmt(v):
        return "0.0" if abs(v) < 0.05 else f"{v:+.1f}"

    def bars(ax, grp):
        d = sub[sub["사례그룹"] == grp].reset_index(drop=True)
        yy = np.arange(len(d))[::-1] * 1.0
        h = 0.31
        series = [(kr_col, ACC, "KR 출원", +h, INK), (sh_col, G2, "KOSIS 출하액", 0.0, G1), (es_col, G3, "KOSIS 사업체 수", -h, G1)]
        for col, cc, lab, off, tc in series:
            ax.barh(yy + off, d[col], height=h * 0.9, color=cc, label=lab)
            for y0, v in zip(yy, d[col]):
                ax.text(v + (0.6 if v >= 0 else -0.6), y0 + off, fmt(v), va="center",
                        ha="left" if v >= 0 else "right", fontsize=FS_S, color=tc)
        ax.axvline(0, color=G1, lw=0.6)
        ax.set_yticks(yy)
        ax.set_yticklabels([SHORT4.get(int(k), NAME.get(int(k), n)) for k, n in zip(d.ksic, d["산업명"])], fontsize=FS)
        ax.set_ylim(-0.55, len(d) - 0.45)
        ax.set_xlabel("연평균 성장률 (%)")
        ax.grid(axis="x", color=G4, lw=0.5); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    ax = cm_axes(fig, 6.65, AY, 4.35, AH); bars(ax, "C2620")
    ax.set_xlim(-15, 18)
    hh, ll = ax.get_legend_handles_labels()
    ax2 = cm_axes(fig, 13.0, AY, 3.95, AH); bars(ax2, "C2610")
    ax2.set_xlim(-26, 31)
    fw, fh = fig.get_size_inches() / CM
    fig.legend(hh, ll, loc="center right", bbox_to_anchor=(16.95 / fw, 7.75 / fh), fontsize=FS_S, ncol=3,
               handlelength=1.0, columnspacing=1.0, borderaxespad=0, borderpad=0)
    ftext(fig, 3.9, 7.75, "(b)(c) 연평균 성장률, 같은 창 2008–09 → 2016–17", fontsize=FS_S, color=G1, va="center")
    ftext(fig, 0.05, 7.05, "(a) C2620 안 LCD·OLED의 KR 출원", fontsize=FS_B, fontweight="bold", va="center")
    ftext(fig, 5.0, 7.05, "(b) C2620 구성 5개 산업", fontsize=FS_B, fontweight="bold", va="center")
    ftext(fig, 11.45, 7.05, "(c) C2610 구성 4개 산업 (반례)", fontsize=FS_B, fontweight="bold", va="center")
    ftext(fig, 0.05, 0.12, "(a) 실선 개발 기간(2005–2017), 점선 사후 구간", fontsize=FS_S, color=G1, va="bottom")
    return save(fig, "fig04_within_code_divergence")


# =========================================================
# fig05. 5개국 × 50산업 RTA 지형도 (map5 fig1 재렌더) - 17 × 11.0 cm
#   50행 한 열 히트맵 대신 한국 RTA 내림차순 50개를 두 열(25행씩)로 접는다.
#   왼쪽 열 = 한국 RTA > 1 (25개), 오른쪽 열 = 한국 RTA ≤ 1 (25개).
# =========================================================
def fig05():
    r = pd.read_csv(os.path.join(FA, "map5", "fig1_rta_map_P3.csv"), encoding="utf-8-sig")
    c = ["KR", "US", "JP", "CN", "EP"]
    r["leader"] = r[c].idxmax(axis=1)
    r = r.sort_values("KR", ascending=False).reset_index(drop=True)
    r["출처"] = "_shared_report/fusion_advantage/map5/fig1_rta_map_P3.csv (출원 2017–19, 동일가중 RTA)"
    csv(r, "fig05_rta_map_5x50")
    n_gt1 = int((r.KR > 1).sum()); n_lead = int((r.leader == "KR").sum())
    lead_cnt = r.leader.value_counts().to_dict()
    ratio = r.KR / r[c].max(axis=1)          # 한국 RTA ÷ 1위국 RTA (원고 '선도국의 0.7배 미만')
    n_gap = int((ratio < 0.7).sum())
    assert n_gt1 == 25 and n_lead == 7 and n_gap == 24, (n_gt1, n_lead, n_gap)

    FH = 11.0
    fig = figure(FH)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 17); ax.set_ylim(0, FH); ax.axis("off")
    cmap = LinearSegmentedColormap.from_list("ga", [G1, G3, "#F7F7F7", ACC_L, ACC])
    norm = TwoSlopeNorm(vmin=-1.6, vcenter=0, vmax=1.6)

    CW, RW = 0.78, 0.98            # 셀 폭, KR÷1위 열 폭 (cm)
    NAME_R = 3.05                  # 산업명 오른쪽 끝 (패널 기준)
    PANEL_X = [0.0, 8.6]
    ROWS = 25
    y_top, y_bot = 9.05, 0.08
    RH = (y_top - y_bot) / ROWS    # 행 높이 ≈ 0.359 cm
    halves = [r.iloc[:ROWS], r.iloc[ROWS:]]
    heads = [f"한국 RTA > 1  ({n_gt1}개)", f"한국 RTA ≤ 1  ({len(r) - n_gt1}개)"]
    for px, part, hd in zip(PANEL_X, halves, heads):
        x0 = px + NAME_R + 0.08
        ax.text(px + 0.05, y_top + 0.78, hd, fontsize=FS_B, fontweight="bold", va="center",
                color=ACC if px == 0 else G1)
        for j, cc in enumerate(c):
            ax.text(x0 + (j + 0.5) * CW, y_top + 0.25, cc, ha="center", va="center", fontsize=FS, fontweight="bold",
                    color=INK)
        ax.text(x0 + 5 * CW + RW / 2 + 0.04, y_top + 0.25, "KR÷1위", ha="center", va="center", fontsize=FS_S, color=G1)
        for i, (idx, row) in enumerate(part.iterrows()):
            yc = y_top - (i + 0.5) * RH
            kr_lead = row.leader == "KR"
            ax.text(px + NAME_R, yc, sname(row.ksic), ha="right", va="center", fontsize=FS,
                    color=ACC if kr_lead else INK, fontweight="bold" if kr_lead else "normal")
            for j, cc in enumerate(c):
                v = row[cc]
                lv = np.log2(v)
                ax.add_patch(Rectangle((x0 + j * CW, yc - RH / 2), CW, RH, fc=cmap(norm(np.clip(lv, -1.6, 1.6))),
                                       ec=WHITE, lw=0.4, zorder=1))
                lead = row.leader == cc
                ax.text(x0 + (j + 0.5) * CW, yc, f"{v:.2f}", ha="center", va="center", fontsize=FS_S,
                        color=WHITE if abs(lv) > 1.25 else INK, fontweight="bold" if lead else "normal", zorder=3)
                if lead:
                    ax.add_patch(Rectangle((x0 + j * CW + 0.03, yc - RH / 2 + 0.03), CW - 0.06, RH - 0.06,
                                           fill=False, ec=INK, lw=1.0, zorder=2))
            q = ratio[idx]
            gap = q < 0.7
            ax.text(x0 + 5 * CW + RW / 2 + 0.04, yc, f"{q:.2f}", ha="center", va="center", fontsize=FS_S,
                    color=INK if gap else G2, fontweight="bold" if gap else "normal")
        ax.plot([px + 0.05, x0 + 5 * CW + RW], [y_top, y_top], color=G2, lw=0.6)

    # 위쪽 요약·범례
    ax.text(0.05, FH - 0.32,
            f"한국 RTA > 1: 50개 중 {n_gt1}개   |   5개국 1위 산업 수: KR {n_lead} · CN {lead_cnt.get('CN', 0)} · "
            f"EP {lead_cnt.get('EP', 0)} · JP {lead_cnt.get('JP', 0)} · US {lead_cnt.get('US', 0)}   |   "
            f"KR÷1위 < 0.7(굵게): {n_gap}개",
            fontsize=FS_B, fontweight="bold", va="center", color=INK)
    ax.add_patch(Rectangle((0.08, FH - 0.93), 0.42, 0.26, fill=False, ec=INK, lw=1.0))
    ax.text(0.62, FH - 0.80, "그 산업의 5개국 1위", fontsize=FS_S, va="center", color=G1)
    ax.text(3.35, FH - 0.80, "파란 굵은 산업명", fontsize=FS_S, va="center", color=ACC, fontweight="bold")
    ax.text(5.55, FH - 0.80, "= 한국 1위", fontsize=FS_S, va="center", color=G1)
    ax.text(7.05, FH - 0.80, "KR÷1위 = 한국 RTA / 그 산업 1위국 RTA", fontsize=FS_S, va="center", color=G1)
    cax = cm_axes(fig, 13.35, FH - 0.88, 3.3, 0.18)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    ticks = [0.35, 0.5, 0.71, 1, 1.41, 2, 2.83]
    cb.set_ticks(np.log2(ticks)); cb.set_ticklabels(["0.35", "0.5", "", "1", "", "2", "2.83"])
    cb.ax.tick_params(labelsize=FS_S, length=2, pad=1)
    cb.outline.set_visible(False)
    ax.text(13.25, FH - 0.79, "RTA", fontsize=FS_S, va="center", ha="right", color=G1)
    return save(fig, "fig05_rta_map_5x50")


# ---- 아래 fig06·fig07은 원고에 쓰지 않아 v2 방식 그대로 둔다(save_legacy). ----
# =========================================================
# fig06. 기술-수출 사분면 (map5 fig2 재렌더)
# =========================================================
def fig06():
    d = pd.read_csv(os.path.join(FA, "map5", "fig2_quadrant_tech_export.csv"), encoding="utf-8-sig")
    d["출처"] = "_shared_report/fusion_advantage/map5/fig2_quadrant_tech_export.csv (특허 RTA·수출 RCA 2015–17)"
    csv(d, "fig06_quadrant_tech_export")
    cnt = d.quad_tech_export_P2.value_counts().to_dict()
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    gap = d.quad_tech_export_P2.str.startswith("기술 강·수출 약")
    ax.scatter(d.pat_rta_P2[~gap], d.exp_rca_P2[~gap], s=26, color=G2, edgecolor=WHITE, lw=0.7, zorder=3)
    ax.scatter(d.pat_rta_P2[gap], d.exp_rca_P2[gap], s=34, color=ACC, edgecolor=WHITE, lw=0.7, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvline(1, color=G1, lw=0.7); ax.axhline(1, color=G1, lw=0.7)
    ax.set_xlim(0.18, 2.3); ax.set_ylim(0.017, 4.2)
    ax.set_xticks([0.25, 0.5, 1, 2]); ax.set_xticklabels(["0.25", "0.5", "1", "2"])
    ax.set_yticks([0.025, 0.1, 0.25, 0.5, 1, 2]); ax.set_yticklabels(["0.025", "0.1", "0.25", "0.5", "1", "2"])
    ax.minorticks_off()
    ax.set_xlabel("한국 특허 RTA (5개국 대비, 출원 2015–17)")
    ax.set_ylabel("한국 수출 RCA (5개 보고국 대비, 2015–17)")
    kw = dict(fontsize=7.3, fontweight="bold")
    ax.text(2.25, 3.7, f"기술 강·수출 강  {cnt.get('기술 강·수출 강(정합)',0)}", ha="right", va="top", color=G1, **kw)
    ax.text(2.25, 0.0195, f"기술 강·수출 약(괴리)  {cnt.get('기술 강·수출 약(괴리)',0)}", ha="right", va="bottom", color=ACC, **kw)
    ax.text(0.19, 3.7, f"기술 약·수출 강(괴리)  {cnt.get('기술 약·수출 강(괴리)',0)}", ha="left", va="top", color=G1, **kw)
    ax.text(0.19, 0.02, f"기술 약·수출 약  {cnt.get('기술 약·수출 약(정합)',0)}", ha="left", va="bottom", color=G1, **kw)
    lab = {31312: (10, 10), 20491: (12, 4), 15220: (12, -4), 33301: (-14, 10), 33309: (14, -10), 26291: (14, -6),
           26224: (-14, 10), 26422: (-8, 12), 29280: (-14, 10), 26212: (4, -16), 26211: (-16, 2), 20423: (14, -10),
           25924: (-14, 6), 21220: (12, 0), 26323: (12, 0)}
    for k, (dx, dy) in lab.items():
        rr = d[d.ksic == k]
        if rr.empty:
            continue
        rr = rr.iloc[0]
        ax.annotate({26211: "LCD", 26212: "OLED"}.get(int(k), NAME[int(k)]), (rr.pat_rta_P2, rr.exp_rca_P2), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.2, color=ACC if str(rr.quad_tech_export_P2).startswith("기술 강·수출 약") else INK,
                    ha="left" if dx > 0 else "right", va="center", zorder=5,
                    bbox=dict(fc=WHITE, ec="none", pad=0.2, alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color=G3, lw=0.5, shrinkA=0, shrinkB=3))
    ax.grid(color=G4, lw=0.5); ax.set_axisbelow(True)
    ax.set_title("기술-수출 사분면: 한국 특허 RTA × 수출 RCA (48개 KSIC)", fontsize=9.5)
    note(fig, "특허 RTA ↔ 수출 RCA 순위상관 ρ 0.356 [0.026, 0.625], n = 48. 사후 구간에서도 85.4%가 같은 사분면에 남는다. 수출 RCA는 KSIC→HS6 연계(48/50).\n"
              "'괴리 산업은 수출이 늘 것이다'로 해석하지 않는다(map5 쓰지 말 문장). 출처: UN Comtrade, 제공 특허-산업 융합데이터; map5/SUMMARY.md, m4_summary.json.",
         y=-0.06)
    return save_legacy(fig, "fig06_quadrant_tech_export")


# =========================================================
# fig07. T3 미래: KSIC별 예측 부상 셀 비율
# =========================================================
def fig07():
    k = pd.read_csv(os.path.join(BASE, "T3_기술셀부상예측", "R3", "future_forecast_r3_ksic.csv"), encoding="utf-8-sig")
    k["name"] = k.ksic.map(NAME)
    k = k.sort_values("pred_rate", ascending=False).reset_index(drop=True)
    tot_cells, tot_pred = int(k.n_cells.sum()), int(k.n_pred_breakout.sum())
    k["출처"] = "T3_기술셀부상예측/R3/future_forecast_r3_ksic.csv (원점 2024 → KR 공개 2025–27)"
    csv(k, "fig07_t3_future_breakout_by_ksic")
    fig, ax = plt.subplots(figsize=(6.3, 8.6))
    fig.subplots_adjust(left=0.40, right=0.97, top=0.935, bottom=0.11)
    y = np.arange(len(k))[::-1]
    col = [ACC if i < 6 else G3 for i in range(len(k))]
    ax.barh(y, k.pred_rate * 100, color=col, height=0.72)
    for yi, r in zip(y, k.itertuples()):
        ax.text(r.pred_rate * 100 + 0.3, yi, f"{r.pred_rate*100:.1f}%  ({r.n_pred_breakout}/{r.n_cells}셀)",
                va="center", fontsize=5.6, color=INK if r.Index < 6 else G1,
                bbox=dict(facecolor=WHITE, edgecolor="none", pad=0.4), zorder=4)
    ax.axvline(tot_pred / tot_cells * 100, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(tot_pred / tot_cells * 100 + 0.2, len(k) - 0.2, f"전체 {tot_pred/tot_cells*100:.1f}%\n({tot_pred:,}/{tot_cells:,}셀)",
            fontsize=6.2, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.ksic}  {r.name}" for r in k.itertuples()], fontsize=5.8)
    for t, i in zip(ax.get_yticklabels(), range(len(k))):
        if i < 6:
            t.set_color(ACC); t.set_fontweight("bold")
    ax.set_xlim(0, 26); ax.set_ylim(-0.7, len(k) + 1.2)
    ax.set_xlabel("예측 부상 셀 비율 (%)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=G4, lw=0.5); ax.set_axisbelow(True)
    fig.text(0.01, 0.985, "T3 기술셀 부상 전망: KSIC별 예측 부상 셀 비율 (원점 2024 → KR 공개 2025–27)", fontsize=9.5, fontweight="bold", va="top")
    note(fig, "동결 모형(E40-auditD, 공식 skill 0.423 [0.366, 0.473])을 학습 원점 2005–2021로 다시 학습한 미래 전망이며 성적이 아닌 참고용이다. 예측 부상 셀 = 적격 KSIC×IPC 메인그룹 셀 5,328개 중 예측 점수 전체 상위 10%. "
              "예측 상위 10%(532셀) 중 54.3%가 KR 0건 셀, 22셀은 IPC 2023 신설 서브클래스.\n"
              "출처: 제공 특허-산업 융합데이터, KOSIS 광공업생산지수, Reliance on Science(Marx & Fuegi, CC BY-NC); T3 R3 FINAL_REPORT_R3.md §5.",
         y=0.005)
    return save_legacy(fig, "fig07_t3_future_breakout_by_ksic")


# =========================================================
# fig08. T1 산업 대기실 지수 상위 산업 - 17 × 7.0 cm
# =========================================================
def fig08():
    d = pd.read_csv(os.path.join(BASE, "T1_산업대기실", "R3", "future_forecast_r3.csv"), encoding="utf-8-sig")
    d = d[["ksic", "at_risk", "waiting_room_index", "mean_prob", "index_newcomer", "index_share", "window", "info_cutoff"]].copy()  # 후보 기업명 열 제외
    d["name"] = d.ksic.map(NAME)
    d = d.sort_values("waiting_room_index", ascending=False).reset_index(drop=True)
    d["출처"] = "T1_산업대기실/R3/future_forecast_r3.csv (c=2024 → 진입 2025–28; 기업명 열 제외)"
    csv(d, "fig08_t1_waiting_room_index")
    top = d.head(15)
    fig = figure(7.0)
    ax = cm_axes(fig, 5.0, 0.95, 11.7, 5.95)
    y = np.arange(len(top))[::-1]
    ax.barh(y, top.index_share * 100, color=[ACC if i < 5 else G3 for i in range(len(top))], height=0.68)
    for yi, r in zip(y, top.itertuples()):
        ax.text(r.index_share * 100 + 0.12, yi, f"{r.index_share*100:.1f}%  (지수 {r.waiting_room_index:,.0f}, 위험쌍 {r.at_risk:,})",
                va="center", fontsize=FS_S, color=INK if r.Index < 5 else G1)
    ax.axvline(100 / len(d), color=INK, lw=0.7, ls=(0, (3, 2)))
    ax.text(100 / len(d) + 0.12, -0.95, f"균등 배분 {100/len(d):.0f}%", fontsize=FS_S, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.ksic}  {r.name}" for r in top.itertuples()], fontsize=FS)
    for t, i in zip(ax.get_yticklabels(), range(len(top))):
        if i < 5:
            t.set_color(ACC); t.set_fontweight("bold")
    ax.set_xlim(0, 14); ax.set_ylim(-1.45, len(top) - 0.45)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax.set_xlabel("대기실 지수 비중 (%, 50개 산업 합 = 100)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=G4, lw=0.5); ax.set_axisbelow(True)
    return save(fig, "fig08_t1_waiting_room_index")


# =========================================================
# fig09. N9 상위 k 적중 vs 기준선 - 17 × 6.4 cm
# =========================================================
def fig09():
    import json
    j = json.load(open(os.path.join(BASE, "N9_정부RnD과제_특허성과예측", "final_result_r6.json"), encoding="utf-8"))
    base = j["base_rate"]
    a = pd.DataFrame({
        "구간": ["상위 1%", "상위 5%", "상위 10%"],
        "챔피언": [j["top1"], j["top5"], j["top10"]],
        "M0_L2(같은 정보 시점 기준선)": [j["top1_M0L2"], j["top5_M0L2"], j["top10_M0L2"]],
    })
    b = pd.DataFrame({
        "모형": ["챔피언(지연판)", "M0 지연판", "강한 기준선 지연판", "파일럿 지연판",
                 "M0 원판", "강한 기준선 원판", "파일럿 원판"],
        "정보시점": ["같음", "같음", "같음", "같음", "1년 더", "1년 더", "1년 더"],
        "상위1%_적중률": [j["top1"], j["top1_M0_L2"], j["top1_strong_L2"], j["top1_pilot_L2"],
                      j["top1_M0_orig"], j["top1_strong_orig"], j["top1_pilot_orig"]],
        "AUC": [j["auc"], j["auc_M0_L2"], j["auc_strong_L2"], j["auc_pilot_L2"], j["auc_M0_orig"],
                j["auc_strong_orig"], j["auc_pilot_orig"]],
        "상위100건": [93, 72, 62, None, None, 97, 96],
    })
    out = pd.concat([a.assign(표="(a)"), b.assign(표="(b)")], ignore_index=True)
    out["기저율"] = base
    out["출처"] = "N9_정부RnD과제_특허성과예측/final_result_r6.json; 상위100건은 R6_결과_20261004.md §1.1·대조 2번"
    csv(out, "fig09_n9_topk_vs_baseline")

    fig = figure(6.4)
    ax = cm_axes(fig, 1.15, 0.95, 5.6, 4.75)
    x = np.arange(3); w = 0.36
    ax.bar(x - w / 2, a["챔피언"] * 100, w, color=ACC, label="동결 모형(챔피언)")
    ax.bar(x + w / 2, a.iloc[:, 2] * 100, w, color=G2, label="기준선 M0 지연판")
    for xi, v1, v2 in zip(x, a["챔피언"], a.iloc[:, 2]):
        ax.text(xi - w / 2, v1 * 100 + 1.5, f"{v1*100:.1f}", ha="center", va="bottom", fontsize=FS_S, fontweight="bold")
        ax.text(xi + w / 2, v2 * 100 + 1.5, f"{v2*100:.1f}", ha="center", va="bottom", fontsize=FS_S, color=G1)
    ax.axhline(base * 100, color=INK, lw=0.7, ls=(0, (3, 2)))
    ax.text(2.42, base * 100 + 2.2, f"기저율 {base*100:.2f}%", ha="left", va="bottom", fontsize=FS_S)
    ax.set_xticks(x); ax.set_xticklabels(a["구간"], fontsize=FS)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 92); ax.set_xlim(-0.5, 2.95); ax.set_ylabel("적중률 (%)")
    ax.legend(loc="upper right", fontsize=FS_S, handlelength=1.0, borderaxespad=0.1)
    ax.grid(axis="y", color=G4, lw=0.5); ax.set_axisbelow(True)
    ftext(fig, 0.05, 6.1, "(a) 점수 상위 구간 적중률", fontsize=FS_B, fontweight="bold", va="center")

    ax = cm_axes(fig, 10.15, 0.95, 6.75, 4.75)
    yy = np.arange(len(b))[::-1]
    for yi, r in zip(yy, b.itertuples()):
        champ = r.Index == 0
        c = ACC if champ else (G2 if r.정보시점 == "같음" else G1)
        ax.plot([0, r._3 * 100], [yi, yi], color=G4, lw=1.4, zorder=1)
        ax.scatter(r._3 * 100, yi, s=32 if champ else 24, color=c, zorder=3,
                   marker="o" if r.정보시점 == "같음" else "D", edgecolor=WHITE, lw=0.6)
        t100 = "" if pd.isna(r.상위100건) else f" · 상위100 {int(r.상위100건)}건"
        ax.text(r._3 * 100 + 1.8, yi, f"{r._3*100:.1f}%{t100}", va="center", fontsize=FS_S,
                color=INK if champ else G1, fontweight="bold" if champ else "normal")
    ax.axhline(2.5, color=G3, lw=0.6)
    ax.text(86.0, 2.58, "▲ 같은 정보 시점", fontsize=FS_S, color=G1, va="bottom")
    ax.text(86.0, 2.42, "▼ 정보 1년 더 사용", fontsize=FS_S, color=G1, va="top")
    ax.set_yticks(yy); ax.set_yticklabels(b["모형"], fontsize=FS)
    ax.set_xlim(40, 112); ax.set_xticks([40, 50, 60, 70, 80, 90, 100])
    ax.set_ylim(-0.6, len(b) - 0.4)
    ax.set_xlabel("상위 1%(751건) 적중률 (%)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=G4, lw=0.5); ax.set_axisbelow(True)
    ftext(fig, 7.55, 6.1, "(b) 상위 1% 적중률: 기준선 전체와 비교", fontsize=FS_B, fontweight="bold", va="center")
    return save(fig, "fig09_n9_topk_vs_baseline")


USED = [fig02, fig03, fig04, fig05, fig08, fig09]   # 원고에 쓰는 데이터 그림(fig06·fig07은 원고 미사용)

if __name__ == "__main__":
    import sys
    want = sys.argv[1:]
    for f in USED:
        if not want or f.__name__ in want:
            f()
