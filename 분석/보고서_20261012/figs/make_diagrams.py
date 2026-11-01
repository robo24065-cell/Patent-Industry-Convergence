# -*- coding: utf-8 -*-
"""fig01(데이터 연계 프레임워크)과 fig10(검증 설계) 도식 v3.
최종 인쇄 크기(폭 17cm)로 그리며 도식 좌표 단위는 cm다(축 xlim 0–17, ylim 0–높이).
그림 제목·번호·출처는 PNG에 넣지 않는다(문서 캡션에서 붙임). 그림 안 글자 7pt 이상.
세부 수치·출처는 원고 표 3과 캡션에 있으므로 도식 안 설명 문장은 줄였다(숫자는 v2와 같음, v2 사본: _v2/make_diagrams.py).
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from _style import *  # noqa


def box(ax, x, y, w, h, fc=WHITE, ec=G2, lw=0.8, ls="-", r=0.12, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, color=G1, lw=1.0, ls="-", style="-|>", ms=8, z=3, cs="arc3,rad=0"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls, zorder=z,
                        connectionstyle=cs, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def canvas(h_cm):
    fig = figure(h_cm)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 17); ax.set_ylim(0, h_cm); ax.axis("off")
    return fig, ax


def headed_box(ax, x, y, w, h, head, items, head_fc=ACC_XL, ec=G2, lw=0.7, head_h=0.52, step=0.44,
               item_fs=FS, head_fs=FS_B, accent_prefix="→"):
    """머리띠가 있는 상자. 항목 글자는 상자 안에 있어야 한다(검사 등록)."""
    box(ax, x, y, w, h, fc=WHITE, ec=ec, lw=lw)
    box(ax, x, y + h - head_h, w, head_h, fc=head_fc, ec=ec, lw=lw)
    t = ax.text(x + 0.15, y + h - head_h / 2, head, fontsize=head_fs, fontweight="bold", va="center", color=INK, zorder=4)
    boxed(t, ax, x, y + h - head_h, x + w, y + h)
    for i, it in enumerate(items):
        acc = it.startswith(accent_prefix)
        t = ax.text(x + 0.18, y + h - head_h - 0.32 - i * step, it if acc else "· " + it, fontsize=item_fs,
                    va="center", color=ACC if acc else INK, zorder=4)
        boxed(t, ax, x, y, x + w, y + h - head_h)


# =========================================================
# fig01. 데이터 연계 프레임워크 - 17 × 10.0 cm (가로형)
# =========================================================
def fig01():
    groups = [
        ("다리 ①  KSIC 5자리 코드", [
            "KOSIS 광업·제조업조사 · 광공업생산지수",
            "UN Comtrade HS6 무역 (KSIC→HS6 48/50)",
            "산업연관표·물가 (한국은행) · KSIC 해설서",
            "→ KOSIS 5자리: 제공 50/50, 연계표 16/50"]),
        ("다리 ②  출원인코드 → 사업자번호", [
            "국민연금 사업장 (공공데이터포털)",
            "국세청 사업자 상태조회 API",
            "OpenDART 기업개황 · KIND 상장사",
            "NTIS 과제 수행기관"]),
        ("다리 ③  출원·등록번호 (패밀리 포함)", [
            "특허 등록원부 (자체 수집: 납부·분쟁·이전)",
            "NTIS 성과 연결 (과제 → 특허)",
            "Reliance on Science (논문 인용)",
            "ETSI 표준특허 선언 · KPSS · OpenAlex"]),
    ]
    center = [("제공 특허-산업", "h"), ("융합데이터", "h"), ("", "gap"),
              ("5개국(KR·US·JP·CN·EP) 특허", "n"), ("× KSIC 5자리 50개 산업 라벨", "n"), ("(AI 라벨, 특허당 1개)", "s"),
              ("", "gap"),
              ("5개국에 같은 체계 라벨", "n"), ("같은 발명 KR·해외 일치 87–90%", "b"), ("", "gap"),
              ("패밀리·우선권 · IPC · 출원인코드", "s")]
    trend = ("트렌드 분석  (KSIC × 국가 × 연도)", [
        "5개국 × 50산업 RTA 지형도 (3.1.1)",
        "기술-수출 사분면 (3.1.2)",
        "연계표 코드 안 산업 분화 (3.1.3)",
        "50개 산업 대시보드 (트랙 미래 열)"])
    warn = ("9개 조기경보  (잠긴 최종시험 1회)", [
        ("뜬다", "T3 기술셀 부상"), ("들어온다", "T1 산업 대기실"), ("다툰다", "N8 특허 분쟁 표적"),
        ("버려진다", "T4 특허 포기"), ("쓰러진다", "N4 기업 폐업"), ("표준이 된다", "N5 표준특허 선언"),
        ("특허↔R&D", "N9 · N6 · N7")])
    control = ("대조군: 공개 자료 배정", "라벨만 교체, 제공 라벨과 5자리 일치율", [
        "연계표(2013): 그룹 40.0%, 5자리 7.1%",
        "연계표 + 해설서 텍스트(TXT): 29.4%",
        "해설서 단독(TXT-FREE): 40.4%",
        "해외 문서 FREE-IPC: 48–53%"])

    # CSV (도식 구성 요소와 숫자 출처). v3 도식은 표 3과 겹치는 설명을 줄였다(KOSIS 연결의 TXT 50/50은 표 3·원고에 있음).
    rows = []
    for g, items in groups:
        for it in items:
            rows.append(dict(위치="외부 데이터(다리)", 묶음=g, 항목=it))
    for it, kind in center:
        if it:
            rows.append(dict(위치="중심", 묶음="제공 데이터", 항목=it))
    for it in trend[1]:
        rows.append(dict(위치="출력", 묶음=trend[0], 항목=it))
    for s, t in warn[1]:
        rows.append(dict(위치="출력", 묶음=warn[0], 항목=f"{s}: {t}"))
    for it in control[2]:
        rows.append(dict(위치="대조군", 묶음=control[0], 항목=it))
    rows.append(dict(위치="화살표", 묶음="대조군 → 중심", 항목="대체 시험 → 필수성 판정(기업 업종 일치: 3.6)"))
    df = pd.DataFrame(rows)
    df["숫자 출처"] = ("87–90%: res_fam_check.json links US 0.873·JP 0.870·EP 0.905 (검증루프 E-C01-08, E-C02-T3-09) / 40.0%(연계표 배정 문서만 41.9%): t0b cw_group_agree_all 0.39997 (E-C02-T3-05) / 7.1%·29.4%·40.4%: R9 §1.1, R10 §3.1 A (LBL-CW-GROUP·LBL-TXT·LBL-FREE) / 48–53%: 등록번호 연결 KR 라벨 상속 48.0–53.5% (검증루프 C10-T3 s2_agree.csv) / "
                     "50/50·16/50: R9 §1.2, R10 §4.3 / HS6 48/50: 공고문_부합성_대조표 §4")
    csv(df, "fig01_framework")

    H = 10.0
    fig, ax = canvas(H)
    TOP = 9.4
    # 열 머리
    hk = dict(fontsize=FS_B, fontweight="bold", ha="center", va="center", color=G1)
    LX, LW = 0.05, 5.55
    CX, CW = 6.45, 4.15
    RX, RW = 11.45, 5.5
    ax.text(LX + LW / 2, 9.75, "외부 데이터 (세 다리)", **hk)
    ax.text(CX + CW / 2, 9.75, "중심", **hk)
    ax.text(RX + RW / 2, 9.75, "출력", **hk)

    # 왼쪽 외부 데이터 묶음
    gh, gg = 2.95, 0.2
    gy = [TOP - gh, TOP - 2 * gh - gg, TOP - 3 * gh - 2 * gg]
    for (g, items), y0 in zip(groups, gy):
        headed_box(ax, LX, y0, LW, gh, g, items, step=0.5)

    # 중심
    cy0, cy1 = 4.25, TOP
    box(ax, CX, cy0, CW, cy1 - cy0, fc=ACC_XL, ec=ACC, lw=1.8, r=0.18)
    yy = cy1 - 0.5
    for t, kind in center:
        if kind == "gap":
            yy -= 0.16
            continue
        kw = dict(ha="center", va="center", zorder=4)
        if kind == "h":
            tt = ax.text(CX + CW / 2, yy, t, fontsize=9.0, fontweight="bold", color=ACC, **kw); yy -= 0.5
        elif kind == "b":
            tt = ax.text(CX + CW / 2, yy, t, fontsize=FS, fontweight="bold", color=INK, **kw); yy -= 0.45
        elif kind == "s":
            tt = ax.text(CX + CW / 2, yy, t, fontsize=FS_S, color=G1, **kw); yy -= 0.45
        else:
            tt = ax.text(CX + CW / 2, yy, t, fontsize=FS, color=INK, **kw); yy -= 0.45
        boxed(tt, ax, CX, cy0, CX + CW, cy1)

    # 다리 화살표 (외부 ↔ 중심)
    for y0, ye in zip(gy, [gy[0] + gh / 2, gy[1] + gh / 2, cy0 + 0.45]):
        ys = y0 + gh / 2 if y0 != gy[2] else gy[2] + gh - 0.35
        arrow(ax, (LX + LW, ys), (CX, ye), color=ACC, lw=1.4, ms=8, style="<|-|>")

    # 오른쪽 출력
    th = 2.75
    headed_box(ax, RX, TOP - th, RW, th, trend[0], trend[1], head_fc=G4, ec=G1, lw=0.8, step=0.47)
    wy0, wy1 = 0.1, TOP - th - 0.25
    box(ax, RX, wy0, RW, wy1 - wy0, fc=WHITE, ec=G1, lw=0.8)
    box(ax, RX, wy1 - 0.52, RW, 0.52, fc=G4, ec=G1, lw=0.8)
    t = ax.text(RX + 0.15, wy1 - 0.26, warn[0], fontsize=FS_B, fontweight="bold", va="center", zorder=4)
    boxed(t, ax, RX, wy1 - 0.52, RX + RW, wy1)
    step = (wy1 - 0.52 - 0.85 - wy0) / (len(warn[1]) - 0.4)
    for i, (s, tk) in enumerate(warn[1]):
        y = wy1 - 0.52 - 0.42 - i * step
        t1 = ax.text(RX + 0.25, y, s, fontsize=FS_S, va="center", color=G1, zorder=4)
        t2 = ax.text(RX + 1.75, y, tk, fontsize=FS, va="center", fontweight="bold", color=INK, zorder=4)
        boxed(t1, ax, RX, wy0, RX + RW, wy1); boxed(t2, ax, RX, wy0, RX + RW, wy1)
    t = ax.text(RX + RW / 2, wy0 + 0.3, "생애주기 순서 · 트랙끼리 순위 매기지 않음", fontsize=FS_S, ha="center",
                va="center", color=G1, zorder=4)
    boxed(t, ax, RX, wy0, RX + RW, wy1)
    arrow(ax, (CX + CW, TOP - th / 2), (RX, TOP - th / 2), color=G1, lw=1.3, ms=9)
    arrow(ax, (CX + CW, cy0 + 0.9), (RX, (wy0 + wy1) / 2 + 0.6), color=G1, lw=1.3, ms=9)

    # 아래 대조군 (가운데)
    bx, bw, by0, by1 = 5.85, 5.35, 0.1, 3.5
    box(ax, bx, by0, bw, by1 - by0, fc="#F7F7F7", ec=G2, lw=0.9, ls=(0, (4, 2)))
    t = ax.text(bx + 0.15, by1 - 0.35, control[0], fontsize=FS_B, fontweight="bold", va="center", color=G1, zorder=4)
    boxed(t, ax, bx, by0, bx + bw, by1)
    t = ax.text(bx + 0.15, by1 - 0.82, control[1], fontsize=FS_S, va="center", color=G1, zorder=4)
    boxed(t, ax, bx, by0, bx + bw, by1)
    for i, it in enumerate(control[2]):
        t = ax.text(bx + 0.2, by1 - 1.38 - i * 0.5, "· " + it, fontsize=FS, va="center", color=INK, zorder=4)
        boxed(t, ax, bx, by0, bx + bw, by1)
    arrow(ax, (CX + CW / 2, by1), (CX + CW / 2, cy0), color=G2, lw=1.0, ls=(0, (3, 2)), ms=8)
    ax.text(CX + CW / 2 + 0.15, (by1 + cy0) / 2, "필수성 판정 (3.6)", fontsize=FS_S, color=G1,
            va="center", ha="left", zorder=5)
    return save(fig, "fig01_framework")


# =========================================================
# fig10. 검증 설계 - 17 × 9.6 cm
# =========================================================
def fig10():
    steps = [
        ("① 개발 원점 실험", "전진 검증, 원점별 학습·평가", "T3 개발 원점 9개, skill 0.4275"),
        ("② 채택 규칙", "개선 +0.003 이상 AND 원점 60% 이상", "T3 3라운드 판정 55건"),
        ("③ 동결", "사양·코드 해시 고정", "N9 동결 해시 6개 일치"),
        ("④ 실행 전 감사", "누수·피처 시점 교란 시험", "T1 교란 최대 차 0.0, T3 RoS 누수 0.25–0.48% 제거"),
        ("⑤ 잠긴 최종시험 1회", "동결 모형, 결과 구간 첫 조회", "T3 개발 0.428 → 최종 0.423"),
        ("⑥ 독립 재계산", "실행기 함수 없이 원천에서 재구성", "T3 skill 차이 1e-16, T1 최대 차 9.0e-8"),
        ("⑦ 위약(레이블 섞기)", "시험·학습 레이블 섞어 재채점", "N8 0/1,000·0/205, N9 p = 0.001, N7 0/1,000·0/230"),
        ("⑧ 레드팀 재검증", "공식값 유지, 보수적 수치 병기", "N7 0.286 → 0.251, N4 0.177 → 0.140 (NPS 키)"),
    ]
    # 인쇄 폭(상자 3.93cm)에 맞춘 줄바꿈(내용·숫자는 위 steps와 같음)
    layout = [
        ("① 개발 원점 실험", ["전진 검증, 원점별 학습·평가"], ["T3 개발 원점 9개", "skill 0.4275"]),
        ("② 채택 규칙", ["개선 +0.003 이상 AND", "원점 60% 이상"], ["T3 3라운드 판정 55건"]),
        ("③ 동결", ["사양·코드 해시 고정"], ["N9 동결 해시 6개 일치"]),
        ("④ 실행 전 감사", ["누수·피처 시점 교란 시험"], ["T1 교란 최대 차 0.0", "T3 RoS 누수 0.25–0.48% 제거"]),
        ("⑤ 잠긴 최종시험 1회", ["동결 모형, 결과 구간 첫 조회"], ["T3 개발 0.428 → 최종 0.423"]),
        ("⑥ 독립 재계산", ["실행기 함수 없이", "원천에서 재구성"], ["T3 skill 차이 1e-16", "T1 최대 차 9.0e-8"]),
        ("⑦ 위약(레이블 섞기)", ["시험·학습 레이블 섞어 재채점"], ["N8 0/1,000·0/205", "N9 p = 0.001", "N7 0/1,000·0/230"]),
        ("⑧ 레드팀 재검증", ["공식값 유지, 보수적 수치 병기"], ["N7 0.286 → 0.251", "N4 0.177 → 0.140 (NPS 키)"]),
    ]
    tl = [
        ("개발 원점 t0 = 2010", 2005, 2010, 2011, 2013, "dev"),
        ("개발 원점 t0 = 2014", 2005, 2014, 2015, 2017, "dev"),
        ("개발 원점 t0 = 2018", 2005, 2018, 2019, 2021, "dev"),
        ("잠긴 최종시험 t0 = 2021", 2005, 2021, 2022, 2024, "final"),
        ("미래 전망 t0 = 2024", 2005, 2024, 2025, 2027, "future"),
    ]
    df1 = pd.DataFrame(steps, columns=["단계", "내용", "사례 숫자"])
    df1["출처"] = ("T3 R3 FINAL_REPORT_R3.md(개발 0.4275, 판정 55건, RoS 0.25–0.48%, 1e-16), T1 R3 FINAL_REPORT_R3.md(교란 0.0, 9.0e-8), "
                  "R6 §1.1(N8·N9 위약, N9 해시), R6 §2(N7 위약·보정), R6 §3.1·§3.2, 공고문_부합성_대조표 §4(채택 규칙)")
    df2 = pd.DataFrame(tl, columns=["행", "피처창_시작", "원점", "결과창_시작", "결과창_끝", "구분"])
    df2["비고"] = "T3 공개연도판 예시: 학습 원점 2005–2018, 개발 원점 2010–2018(9개), 최종 t0=2021 → 공개 2022–24, 미래 t0=2024 → 2025–27"
    csv(pd.concat([df1.assign(표="흐름"), df2.assign(표="시간축")], ignore_index=True), "fig10_validation_design")

    H = 9.6
    fig = figure(H)
    # 위: 시간축
    ax = cm_axes(fig, 3.75, 6.0, 13.15, 2.45)
    for i, (lab, fs, t0, rs, re, kind) in enumerate(tl):
        y = len(tl) - 1 - i
        ax.plot([fs, t0 + 0.5], [y, y], color=G3, lw=3.2, solid_capstyle="butt")
        if kind == "dev":
            ax.add_patch(Rectangle((rs - 0.5, y - 0.3), re - rs + 1, 0.6, fc=ACC_L, ec="none"))
        elif kind == "final":
            ax.add_patch(Rectangle((rs - 0.5, y - 0.3), re - rs + 1, 0.6, fc=ACC, ec="none"))
            ax.text(re + 0.9, y, "잠김 · 1회 채점", fontsize=FS_S, va="center", color=ACC, fontweight="bold")
        else:
            ax.add_patch(Rectangle((rs - 0.5, y - 0.3), re - rs + 1, 0.6, fc=WHITE, ec=G1, lw=0.8, ls=(0, (2, 1.5))))
            ax.text(re + 0.9, y, "참고용 전망", fontsize=FS_S, va="center", color=G1)
        ax.plot([t0 + 0.5], [y], marker="|", ms=8, color=INK, mew=1.2)
        ax.text(2004.2, y, lab, ha="right", va="center", fontsize=FS,
                color=ACC if kind == "final" else INK, fontweight="bold" if kind == "final" else "normal")
    ax.axvline(2021.5, color=INK, lw=0.7, ls=(0, (3, 2)))
    ax.text(2021.8, 4.0, "결과 구간 잠금 경계 (공개 2022년 이후)", fontsize=FS_S, va="center", ha="left", color=INK)
    ax.text(2021.8, 3.0, "개발 원점 9개(2010–2018) 중 3개 표시", fontsize=FS_S, va="center", ha="left", color=G1)
    ax.set_xlim(2004.5, 2031.5); ax.set_ylim(-0.6, 4.6)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xticks([2005, 2010, 2015, 2020, 2025, 2030])
    ax.set_xlabel("KR 공개연도", labelpad=1)
    h = [Line2D([], [], color=G3, lw=3.2, label="피처 창(원점까지 공개분)"),
         Rectangle((0, 0), 1, 1, fc=ACC_L, label="개발 결과 창(3년)"),
         Rectangle((0, 0), 1, 1, fc=ACC, label="최종시험 결과 창"),
         Line2D([], [], marker="|", ls="", color=INK, ms=8, mew=1.2, label="원점 t0")]
    fw, fh = fig.get_size_inches() / CM
    fig.legend(handles=h, loc="center right", bbox_to_anchor=(16.85 / fw, 8.85 / fh), ncol=4, fontsize=FS_S,
               handlelength=1.4, columnspacing=1.0, borderaxespad=0, borderpad=0)
    ax0 = fig.add_axes([0, 0, 1, 1], zorder=-1); ax0.set_xlim(0, 17); ax0.set_ylim(0, H); ax0.axis("off")
    ax0.text(0.05, 9.3, "(a) 시간축 (T3 기술셀 부상, 공개연도판 예시)", fontsize=FS_B, fontweight="bold", va="center")

    # 아래: 흐름
    ax2 = fig.add_axes([0, 0, 1, 1]); ax2.set_xlim(0, 17); ax2.set_ylim(0, H); ax2.axis("off")
    ax2.text(0.05, 4.95, "(b) 단계와 실제 적용 사례", fontsize=FS_B, fontweight="bold", va="center")
    bw, bh, gap = 3.93, 2.08, 0.34
    xs = [0.05 + i * (bw + gap) for i in range(4)]
    ys = [2.53, 0.08]
    for k, (t, d, ex) in enumerate(layout):
        r, c = divmod(k, 4)
        x, y = xs[c], ys[r]
        lock = k == 4
        assert x + bw <= 16.95, x + bw
        box(ax2, x, y, bw, bh, fc=ACC_XL if lock else WHITE, ec=ACC if lock else G2, lw=1.4 if lock else 0.8)
        lines = [(t, dict(fontsize=FS_B, fontweight="bold", color=ACC if lock else INK))]
        lines += [(s, dict(fontsize=FS_S, color=G1)) for s in d]
        lines += [("· " + s, dict(fontsize=FS_S, color=INK)) for s in ex]
        yy = y + bh - 0.3
        for j, (s, kw) in enumerate(lines):
            tt = ax2.text(x + 0.14, yy, s, va="center", zorder=4, **kw)
            boxed(tt, ax2, x, y, x + bw, y + bh)
            yy -= 0.4 if j == 0 else 0.34
            if j == len(d):           # 설명과 사례 사이 간격
                yy -= 0.06
        if c < 3:
            arrow(ax2, (x + bw, y + bh / 2), (x + bw + gap, y + bh / 2), color=G1, lw=1.0, ms=7)
    # 1행 끝 → 2행 시작
    ym = (ys[0] + ys[1] + bh) / 2
    ax2.plot([xs[3] + bw / 2, xs[3] + bw / 2, xs[0] + bw / 2], [ys[0], ym, ym], color=G1, lw=1.0, zorder=3)
    arrow(ax2, (xs[0] + bw / 2, ym), (xs[0] + bw / 2, ys[1] + bh), color=G1, lw=1.0, ms=6)
    return save(fig, "fig10_validation_design")


if __name__ == "__main__":
    fig01()
    fig10()
