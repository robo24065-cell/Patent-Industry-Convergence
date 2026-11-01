# -*- coding: utf-8 -*-
r"""robustness_table.md의 위약 B 자리표시(PLACEBO_B_TABLE)를 w8_placeboB.json·청크 csv 실제 집계값으로 채운다(2026-10-05 재시도)."""
import os, glob, json
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "w8_placeboB.json"), encoding="utf-8"))
OBS_ALIAS = json.load(open(os.path.join(HERE, "w2_fit_alias.json"), encoding="utf-8"))
sk_o, sk_a = R["observed"]["skill"], OBS_ALIAS["skill"]["M0"]
d_o = R["observed"]["auc_champ"] - R["observed"]["auc_M0"]; d_a = OBS_ALIAS["auc"]["champ"] - OBS_ALIAS["auc"]["M0"]
st = {}
for s in ["t0nps", "global"]:
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(HERE, "cache", f"placeboB_{s}_*.csv"))]).drop_duplicates("rep")
    df = d.auc_champ - d.auc_M0
    st[s] = dict(n=len(d), ge_a=int((d.skill >= sk_a).sum()), mx=float(d.skill.max()), mxauc=float(d.auc_champ.max()),
                 dge_o=int((df >= d_o).sum()), dge_a=int((df >= d_a).sum()), top=d.sort_values("skill").iloc[-1].to_dict())
    print(s, st[s])
f = lambda v: f"{v:.3f}".replace("-", "−")
def row(name, s):
    r, q = R[s], st[s]
    return (f"| {name} | {r['n_rep']} | {f(r['skill_mean'])} | [{f(r['skill_95'][0])}, {f(r['skill_95'][1])}] | {r['skill_ge_obs']}/{r['n_rep']} | "
            f"{q['ge_a']}/{q['n']} (최대 {f(q['mx'])}) | {f(r['auc_champ_mean'])} [{f(r['auc_champ_95'][0])}, {f(r['auc_champ_95'][1])}] / {f(q['mxauc'])} | {f(r['auc_M0_mean'])} |")
t = st["t0nps"]["top"]
tbl = f"""공식(orig) 패널에서 학습 원점(2015–2020) 레이블만 섞고 동결 스펙(챔피언 ens 3시드, M0 xgb 3시드)으로 다시 학습했다. 그다음 이미 채점된 시험 레이블(2023–24)로 채점했다. 섞기 시드 = 10000 + 반복 번호. 2026-10-05 재시도에서 남은 청크를 확인하고 다시 집계했다(`w8_placeboB_summary.py` → `w8_placeboB.json`, `w9_fill_placeboB.py`).

| 섞기 방식 | 반복 | skill 평균 | 95% 범위 | skill ≥ 공식 {sk_o:.3f} | skill ≥ 보정 {sk_a:.3f} | AUC 챔피언 평균 [95%] / 최대 | AUC M0 평균 |
|---|---|---|---|---|---|---|---|
{row("층 내(원점 × NPS 가입, R5 사전 지정)", "t0nps")}
{row("완전 무작위", "global")}

- 챔피언 AUC는 2방식 모두 최대 {max(st['t0nps']['mxauc'], st['global']['mxauc']):.3f}이다. 관측 AUC는 {R['observed']['auc_champ']:.3f}(공식)과 {OBS_ALIAS['auc']['champ']:.3f}(보정)다. AUC 기준 경험적 p < 1/231이다.
- **주의(skill 비율 지표의 약점)**: 층 내 위약 1회(반복 {int(t['rep'])})에서 skill이 {t['skill']:.3f}로 보정 관측값 {sk_a:.3f} 이상이었다. 이때 AUC는 챔피언 {t['auc_champ']:.3f}, M0 {t['auc_M0']:.3f}였다.
  - M0가 0.5 근처로 떨어지면 분모 (1−AUC_M0)가 커진다. 그래서 순위 정보가 거의 없어도 skill이 커질 수 있다.
  - AUC 차이(챔피언 − M0) 기준으로 위약이 관측 차이 이상인 횟수: 공식 {d_o:.3f} 이상은 층 내 {st['t0nps']['dge_o']}/230, 무작위 {st['global']['dge_o']}/230. 보정 {d_a:.3f} 이상은 층 내 {st['t0nps']['dge_a']}/230, 무작위 {st['global']['dge_a']}/230.
  - 따라서 skill만 단독으로 보고하지 않는다. **AUC 절대값(0.74 vs 위약 최대 0.63)과 상위 k를 함께 보고한다.**
- 층 내 평균 {R['t0nps']['skill_mean']:.3f}이 0보다 큰 것은 층(원점 × NPS 가입) 밖 구성 효과 때문이다(원점별 기저율 차이를 특허 규모가 대신 맞힘). 위약 A의 R5 층 평균 0.081과 같은 현상이다.
- 위약 B는 공식 패널에서만 돌렸다(보정 패널 위약 B는 미실행). 위약 A는 두 패널 모두 0/1,000이다."""
p = os.path.join(HERE, "robustness_table.md")
s = open(p, encoding="utf-8").read()
assert "PLACEBO_B_TABLE" in s
s = s.replace("PLACEBO_B_TABLE", tbl)
s = s.replace("| 위약 B (학습 레이블 섞기 후 재학습, 200회 × 2방식) | §2.2 |",
              f"| 위약 B (학습 레이블 섞기 후 재학습, 230회 × 2방식) | 챔피언 AUC 위약 최대 {max(st['t0nps']['mxauc'], st['global']['mxauc']):.3f}(관측 0.746/0.739). skill ≥ 공식: 0/230·0/230. skill ≥ 보정: {st['t0nps']['ge_a']}/230(층 내)·{st['global']['ge_a']}/230(§2.2 주의) |")
s = s.replace("- 재현·위약 통과. 위약 A는 3방식 모두 관측 이상 0/1,000이다. 위약 B는 §2.2.",
              f"- 재현·위약 통과. 위약 A는 3방식 모두 관측 이상 0/1,000이다. 위약 B(230회 × 2방식)는 공식 skill 이상 0회, 보정 skill 이상 {st['t0nps']['ge_a']}회(층 내)다.\n- 경고: skill(비율)은 M0 AUC가 0.5 근처일 때 부풀 수 있다(위약 1회가 보정값 이상). AUC·상위 k를 함께 보고한다.")
open(p, "w", encoding="utf-8").write(s)
print("filled", "PLACEBO_B_TABLE" not in s)
