# -*- coding: utf-8 -*-
"""
파일럿 PADI 분석 — 설계안 v2 규칙의 실데이터 검증 (완결 11개 산업)
검증 항목:
  [V1] 이벤트 분류(산식 판정) 분포의 상식성
  [V2] 연차별 기저 위험률 h̄(â) 곡선 형태 (특허 vs 실용신안)
  [V3] KM 생존곡선 산업 간 변별력
  [V4] PADI 프로토타입(분기 O/E, 4Q 이동합) 시계열 — 값 튐/이상치 진단
  [V5] 전체 O/E 항등성(파이프라인 단위테스트), E<10 셀 비율, 만기인접 비중
"""
import sqlite3, datetime, collections, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

SNAP = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\공모전데이터\작업데이터\registry_snapshot_20260811.sqlite"
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\pilot"
os.makedirs(OUT, exist_ok=True)
CUTOFF = datetime.date(2025, 6, 30)      # 패널 종점 2025Q2 (설계안 v2)
COLLECT_DATE = datetime.date(2026, 8, 11)
NAMES = {"15220": "신발부품", "21220": "한의약품", "23992": "연마재", "24219": "기타비철금속",
         "25924": "절삭가공", "26223": "연성PCB", "26224": "실장기판", "28902": "탄소절연",
         "29229": "가공공작기계", "31202": "철도부품", "31312": "무인항공기"}

con = sqlite3.connect(SNAP)
rows = con.execute("""
    SELECT rgst_no, ksic, reg_date, exp_date, last_annl FROM registry
    WHERE status='OK' AND source LIKE 'datago%' AND ksic IN ({})
    AND reg_date IS NOT NULL AND exp_date IS NOT NULL AND last_annl IS NOT NULL
""".format(",".join(f"'{k}'" for k in NAMES))).fetchall()
con.close()
print(f"분석 대상: {len(rows):,}건 (완결 11개 산업, datago 소스)")


def to_date(s):
    return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def add_years(d, n):
    try:
        return d.replace(year=d.year + n)
    except ValueError:            # 2/29
        return d.replace(year=d.year + n, day=28)


# ── 이벤트 테이블 구축 (설계안 v2 §2.1) ─────────────────────────
recs = []
for rgst, ksic, rd, ed, la in rows:
    reg, exp = to_date(rd), to_date(ed)
    if exp <= reg:
        continue
    typ = "실용" if rgst.startswith("2") else "특허"
    M = round((exp - reg).days / 365.25)            # 만기연차(등록 후 잔여 연수)
    la = int(la)
    covered = add_years(reg, la)                    # 커버 만료일 = la+1차 기한일
    if covered >= exp:
        ev, age, evd = "자연만료", None, exp
    elif (covered - COLLECT_DATE).days < -270:      # 유예 6개월+버퍼 3개월 경과
        ev, age, evd = "조기포기", la + 1, covered
    else:
        ev, age, evd = "생존", None, None
    near_expiry = ev == "조기포기" and age is not None and age > M - 2
    recs.append((rgst, ksic, typ, reg, exp, M, la, ev, age, evd, near_expiry))

df = pd.DataFrame(recs, columns=["rgst", "ksic", "type", "reg", "exp", "M", "la", "event", "age", "evd", "near_exp"])
print("\n[V1] 이벤트 분류 분포")
print(df.groupby(["type", "event"]).size().unstack(fill_value=0).to_string())
print(f"만기인접 포기(주지표 제외 대상): {df.near_exp.sum():,}건 "
      f"({df.near_exp.sum() / max((df.event == '조기포기').sum(), 1) * 100:.1f}% of 조기포기)")

# ── [V2] 연차별 기저 위험률 h̄(â): due 기준 (v2 §2.2) ────────────
def hazard_table(sub):
    """a차 기한(reg+(a-1)년)이 CUTOFF 이전에 도래한 특허 중, 그 기한에서 포기한 비율"""
    out = {}
    for a in range(2, 21):
        due_ok = sub[(sub.reg.apply(lambda d: add_years(d, a - 1)) <= CUTOFF)]
        risk = due_ok[(due_ok.la >= a - 1) & (due_ok.M >= a)]       # 직전까지 납부 & 만기 전
        events = risk[(risk.event == "조기포기") & (risk.age == a) & (~risk.near_exp)]
        if len(risk) >= 30:
            out[a] = (len(events), len(risk), len(events) / len(risk))
    return out


hz = {t: hazard_table(df[df.type == t]) for t in ("특허", "실용")}
fig, ax = plt.subplots(figsize=(9, 5))
for t, marker in (("특허", "o"), ("실용", "s")):
    ages = sorted(hz[t])
    ax.plot(ages, [hz[t][a][2] * 100 for a in ages], marker=marker, label=f"{t} (n={sum(hz[t][a][1] for a in ages):,})")
ax.set_xlabel("연차 â (해당 연차료 납부 기한)"); ax.set_ylabel("포기 위험률 (%)")
ax.set_title("연차별 기저 포기위험률 h(a) — 완결 11개 산업 풀링")
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_hazard.png", dpi=130); plt.close(fig)

# ── [V3] KM 생존곡선 (특허만, 산업별) ───────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for ksic, g in df[df.type == "특허"].groupby("ksic"):
    ht = hazard_table(g)
    if len(ht) < 8:
        continue
    ages = sorted(ht)
    surv = np.cumprod([1 - ht[a][2] for a in ages])
    ax.plot(ages, surv * 100, label=f"{ksic} {NAMES[ksic]}", alpha=.8)
ax.set_xlabel("연차"); ax.set_ylabel("유지율 (%)")
ax.set_title("산업별 특허 생존곡선 (KM) — 조기포기만 사건 처리")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_km.png", dpi=130); plt.close(fig)

# ── [V4] PADI 프로토타입: 산업×분기 O/E (4Q 이동합) ─────────────
def quarter(d):
    return d.year * 10 + (d.month - 1) // 3 + 1


hbar = {t: {a: v[2] for a, v in hz[t].items()} for t in hz}
qtrs = []
y, q = 2007, 1
while (y, q) <= (2025, 2):
    qtrs.append(y * 10 + q)
    q += 1
    if q == 5: y, q = y + 1, 1

OE = {}   # (ksic, qtr) -> [O, E]
for _, r in df.iterrows():
    hb = hbar[r.type]
    for a in range(2, min(int(r.M), 20) + 1):
        due = add_years(r.reg, a - 1)
        if due > CUTOFF or due.year < 2007:
            continue
        if r.la >= a - 1:                                   # 위험집합 진입
            qt = quarter(due)
            key = (r.ksic, qt)
            OE.setdefault(key, [0.0, 0.0])
            OE[key][1] += hb.get(a, 0.0)
            if r.event == "조기포기" and r.age == a and not r.near_exp:
                OE[key][0] += 1

padi = pd.DataFrame([(k[0], k[1], v[0], v[1]) for k, v in OE.items()],
                    columns=["ksic", "qtr", "O", "E"]).sort_values(["ksic", "qtr"])
padi["qi"] = padi.qtr.map({q: i for i, q in enumerate(qtrs)})
# 4분기 이동합 O/E
frames = []
for ksic, g in padi.groupby("ksic"):
    g = g.set_index("qi").reindex(range(len(qtrs)), fill_value=0.0)
    g["ksic"] = ksic
    g["O4"] = g.O.rolling(4).sum()
    g["E4"] = g.E.rolling(4).sum()
    g["padi"] = g.O4 / g.E4.replace(0, np.nan)
    frames.append(g.reset_index())
padi4 = pd.concat(frames)

fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
xt = [i for i, q in enumerate(qtrs) if q % 10 == 1 and (q // 10) % 3 == 0]
for ax, (ksic, g) in zip(axes.flat, padi4.groupby("ksic")):
    ax.plot(g["qi"], g.padi, lw=1.2)
    ax.axhline(1.0, color="gray", ls="--", lw=.8)
    ax.set_title(f"{ksic} {NAMES[ksic]}", fontsize=10)
    ax.set_xticks(xt); ax.set_xticklabels([str(qtrs[i] // 10) for i in xt], fontsize=7)
    ax.grid(alpha=.3)
for ax in axes.flat[len(padi4.ksic.unique()):]:
    ax.axis("off")
fig.suptitle("PADI 프로토타입 — 산업별 O/E (4분기 이동합), 1.0 = 연령보정 기대 수준", y=1.0)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_padi.png", dpi=120); plt.close(fig)

# ── [V5] 진단 ───────────────────────────────────────────────────
print("\n[V5] 진단")
tot_O, tot_E = padi.O.sum(), padi.E.sum()
print(f"  전체 O/E 항등성 검사(단위테스트): {tot_O:,.0f}/{tot_E:,.0f} = {tot_O / tot_E:.3f}  (1.0 근처면 파이프라인 정상)")
small = (padi.E < 10).mean()
print(f"  E<10 소표본 분기셀 비율: {small * 100:.0f}% (분기 단독), 4Q 이동합 후: {(padi4.E4 < 10).mean() * 100:.0f}%")
ext = padi4[(padi4.padi > 3) & (padi4.E4 >= 10)]
print(f"  극단값(PADI>3 & E4>=10): {len(ext)}셀")
print(f"  산업별 조기포기율(등록건 대비): ")
for ksic, g in df.groupby("ksic"):
    er = (g.event == "조기포기").mean() * 100
    print(f"    {ksic} {NAMES[ksic]:8s}: {er:5.1f}%  (n={len(g):,})")
print(f"\n차트 저장: {OUT}")
