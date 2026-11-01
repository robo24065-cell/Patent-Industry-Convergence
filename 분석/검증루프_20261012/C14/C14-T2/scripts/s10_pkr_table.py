# C14-T2 s10: pkr_ipc_altA.csv — 부록 진단용 PKR-IPC 옛값 / R8 재현 / 대안 A(REG) / 보조(PUB·UNI·상속 없음) / 독립 재현 / 판정, + R11-PKR 의존 판정
import sys, json
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
W = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C14\C14-T2"
s5 = pd.read_csv(W + r"\cache\s5_pkr_agree_full.csv", encoding="utf-8-sig"); s5 = s5[s5.window == "출원2015-19_발명"].set_index("cc")
v0 = json.load(open(W + r"\verify_0\v0_out.json", encoding="utf-8"))
st = json.load(open(W + r"\cache\s5_stats.json", encoding="utf-8"))
app = {"US": 68.5, "JP": 71.1, "CN": 68.0, "EP": 68.8}
rows = []
for cc in ["US", "JP", "CN", "EP"]:
    r = s5.loc[cc]; reg = r.pkr5_reg * 100
    d = reg - app[cc]
    rows.append(dict(항목="C-AGREE-%s 진단용 PKR-IPC" % cc, cc=cc, n=int(r.n), 부록값_v15=app[cc], R8재현=round(r.pkr5_r8 * 100, 2), 저장값=round(r.pkr5_stored * 100, 2),
                     대안A_REG=round(reg, 2), 대안A_표기=f"{reg:.1f}%", PUB=round(r.pkr5_pub * 100, 2), UNI=round(r.pkr5_uni * 100, 2), 상속없음=round(r.pkr5_pure * 100, 2),
                     REG_상속비중=round(r.famshare_reg * 100, 2), REG_상속분일치=round(r.famagree_reg * 100, 1) if pd.notna(r.famagree_reg) else "",
                     R8_상속비중=round(r.famshare_r8 * 100, 2), R8_상속분일치=round(r.famagree_r8 * 100, 1) if pd.notna(r.famagree_r8) else "",
                     독립재현_REG=round(v0["REG"][cc]["pkr5"] * 100, 3), 독립재현_차_pp=round(abs(v0["REG"][cc]["pkr5"] * 100 - reg), 4),
                     차이_pp=round(d, 2), 판정="옛값(정정 필요)" if abs(d) > 0.5 else "실질 불변(표시만)",
                     근거=r"c5_R10 c1_assign.py 정의, 연결만 C10-T3 cache\links.parquet REG로 교체; scripts\s5_pkr_altA.py, verify_0\v0_pkr_independent.py"))
rr = {x[0]: x for x in st["r11_fdoc_pkr_vs_pure"]}
for lab, v, ci in [("R11-PKR NPS", "+2.31%p", "[+0.54, +4.13]"), ("R11-PKR DART", "+4.04%p", "[+1.09, +6.96]")]:
    rows.append(dict(항목=lab, cc="US·JP·EP(패밀리 표본)", n=sum(x[1] for x in st["r11_fdoc_pkr_vs_pure"]), 부록값_v15=v + " " + ci,
                     R8재현="", 저장값="", 대안A_REG="재계산 불필요", 대안A_표기=v, PUB="", UNI="", 상속없음="fdoc k_pkr = 상속 없는 IPC 표 100%",
                     REG_상속비중="", REG_상속분일치="", R8_상속비중="fdoc 중 R8 family 행 %s" % ", ".join(f"{c} {rr[c][4]*100:.1f}%" for c in ["US", "JP", "EP"]), R8_상속분일치="",
                     독립재현_REG="", 독립재현_차_pp="", 차이_pp=0, 판정="결함 연결 비의존(ok)",
                     근거="R11 fp_prep.py가 m_pkr='family'(R8) 행을 IPC 표 값으로 재배정(k_pkr_p); s5 대조 fdoc.k_pkr == 상속 없음 100%(US·JP·EP·CN). 표본 연결은 fam_reg(등록번호)"))
pd.DataFrame(rows).to_csv(W + r"\pkr_ipc_altA.csv", index=False, encoding="utf-8-sig")
print(pd.DataFrame(rows)[["항목", "부록값_v15", "R8재현", "대안A_REG", "PUB", "상속없음", "독립재현_REG", "차이_pp", "판정"]].to_string())
