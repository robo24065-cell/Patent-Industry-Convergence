# NTIS 국가R&D 과제 연도별 전수 수집 (임무 B). 이어받기: ntis_B_work/proj/ 에 페이지별 parquet.
# 사용: python ntis_B_collect.py [분 단위 제한=8.3]
import sys, os, re, time, json, xml.etree.ElementTree as ET
import pandas as pd
import ntis_B_common as c
sys.stdout.reconfigure(encoding="utf-8")
LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 8.3
T0 = time.time()
PDIR = c.WORK + r"\proj"; os.makedirs(PDIR, exist_ok=True)
STATE = c.WORK + r"\ntis_B_collect_state.json"
YEARS = list(range(2002, 2025))
PAGE = 1000
nav = json.load(open(c.WORK + r"\ntis_B_year_nav.json", encoding="utf-8"))
state = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {"stopped": None}
if state.get("stopped"):
    print("이전 중단 기록:", state["stopped"]); sys.exit(1)

def t(e, path):
    x = e.find(path)
    return (x.text or "").strip() if x is not None and x.text else None
def a(e, path, attr="code"):
    x = e.find(path)
    return x.get(attr) if x is not None else None

def parse(x):
    root = ET.fromstring(x.encode("utf-8"))
    rows = []
    for h in root.iter("HIT"):
        r = {
            "pjt_no": t(h, "ProjectNumber"), "secret": t(h, "SecretProject"),
            "title_ko": t(h, "ProjectTitle/Korean"), "title_en": t(h, "ProjectTitle/English"),
            "kw_ko": t(h, "Keyword/Korean"), "kw_en": t(h, "Keyword/English"),
            "order_agency": t(h, "OrderAgency/Name"), "research_agency": t(h, "ResearchAgency/Name"),
            "budget_project": t(h, "BudgetProject/Name"), "business_name": t(h, "BusinessName"),
            "bigproject_title": t(h, "BigprojectTitle"), "manage_agency": t(h, "ManageAgency/Name"),
            "ministry": t(h, "Ministry/Name"), "year": t(h, "ProjectYear"),
            "period_start": t(h, "ProjectPeriod/Start"), "period_end": t(h, "ProjectPeriod/End"),
            "total_start": t(h, "ProjectPeriod/TotalStart"), "total_end": t(h, "ProjectPeriod/TotalEnd"),
            "org_pno": t(h, "OrganizationPNumber"),
            "perform_agent_cd": a(h, "PerformAgent"), "perform_agent": t(h, "PerformAgent"),
            "dev_phase_cd": a(h, "DevelopmentPhases"), "dev_phase": t(h, "DevelopmentPhases"),
            "tech_lifecycle": t(h, "TechnologyLifecycle"), "tech_lifecycle_cd": a(h, "TechnologyLifecycle"),
            "region_cd": a(h, "Region"), "region": t(h, "Region"),
            "econ_goal_cd": a(h, "EconomicSocialGoal"), "econ_goal": t(h, "EconomicSocialGoal"),
            "sixt_cd": a(h, "SixTechnology"), "sixt": t(h, "SixTechnology"),
            "apply1_cd": a(h, "ApplyArea/First"), "apply1": t(h, "ApplyArea/First"),
            "apply2_cd": a(h, "ApplyArea/Second"), "apply2": t(h, "ApplyArea/Second"),
            "apply3_cd": a(h, "ApplyArea/Third"), "apply3": t(h, "ApplyArea/Third"),
            "continuous": t(h, "ContinuousFlag"), "policy_flag": t(h, "PolicyProjectFlag"),
            "gov_funds": t(h, "GovernmentFunds"), "sbiz_funds": t(h, "SbusinessFunds"),
            "total_funds": t(h, "TotalFunds"), "corp_rgst_no": t(h, "CorporateRegistrationNumber"),
            "series": t(h, "SeriesProject"), "pre_pjt": t(h, "PrePjtId"),
            "man": t(h, "Researchers/ManCount"), "woman": t(h, "Researchers/WomanCount"),
        }
        for sc in h.findall("ScienceClass"):
            ty = sc.get("type"); seq = sc.get("sequence") or "1"
            p = f"sc_{ty}{seq}"
            for lv in ("Large", "Medium", "Small"):
                r[f"{p}_{lv[0]}_cd"] = a(sc, lv); r[f"{p}_{lv[0]}"] = t(sc, lv)
        for tag, p in (("MinistryScienceClass", "msc"), ("TempScienceClass", "tsc")):
            e = h.find(tag)
            if e is not None:
                for lv in ("Large", "Medium", "Small"):
                    r[f"{p}_{lv[0]}_cd"] = a(e, lv); r[f"{p}_{lv[0]}"] = t(e, lv)
        # 알 수 없는 추가 태그 기록
        known = {"ProjectNumber","SecretProject","ProjectTitle","Manager","Researchers","Goal","Abstract","Effect","Keyword",
                 "OrderAgency","ResearchAgency","BudgetProject","BusinessName","BigprojectTitle","ManageAgency","Ministry",
                 "ProjectYear","ProjectPeriod","OrganizationPNumber","ScienceClass","MinistryScienceClass","TempScienceClass",
                 "PerformAgent","DevelopmentPhases","TechnologyLifecycle","Region","EconomicSocialGoal","SixTechnology",
                 "ApplyArea","ContinuousFlag","PolicyProjectFlag","GovernmentFunds","SbusinessFunds","TotalFunds",
                 "CorporateRegistrationNumber","SeriesProject","PrePjtId"}
        extra = [ch.tag for ch in h if ch.tag not in known]
        for ch in h:
            if ch.tag not in known:
                r["x_" + ch.tag] = (ch.text or "").strip() or json.dumps({k: v for k, v in ch.attrib.items()}, ensure_ascii=False)
        rows.append(r)
    tot = root.findtext("TOTALHITS")
    return rows, int(tot) if tot else None

ncall = 0
for y in YEARS:
    total = nav[str(y)]["total"]
    npages = (total + PAGE - 1) // PAGE
    for p in range(npages):
        f = PDIR + fr"\ntis_proj_{y}_{p:03d}.parquet"
        if os.path.exists(f): continue
        if (time.time() - T0) / 60 > LIMIT:
            print(f"시간 제한 도달, 호출 {ncall}회. 다음: {y} p{p}"); sys.exit(0)
        def fetch(start, cnt):
            x = c.call("public_project", {"collection": "project", "SRWR": "", "addQuery": f"PY={y}/SAME",
                                          "startPosition": start, "displayCnt": cnt}, timeout=180)
            rows, tot = parse(x)
            if len(rows) == 0 or (tot is not None and tot != total):
                raise RuntimeError(f"이상 응답 rows={len(rows)} tot={tot} 본문={x[:200]!r}")
            return rows
        def fetch_split(start, cnt, depth=0):
            try:
                return fetch(start, cnt)
            except Exception as e:
                state.setdefault("errors", []).append(f"{time.strftime('%F %T')} {y} start={start} cnt={cnt}: {type(e).__name__} {str(e)[:150]}")
                ERR[0] += 1
                if ERR[0] > 6:
                    state["stopped"] = f"{time.strftime('%F %T')} 한 페이지 분할 재시도 중 오류 6회 초과 - 중단 ({y} start={start})"
                json.dump(state, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
                if ERR[0] > 6:
                    print("중단:", state["stopped"]); sys.exit(1)
                if cnt <= 10:
                    state.setdefault("gaps", []).append([y, start, cnt]); return []
                time.sleep(20)
                sub = cnt // 10; out = []
                for k in range(10):
                    out += fetch_split(start + k * sub, sub, depth + 1)
                return out
        ERR = [0]
        try:
            rows = fetch(p * PAGE + 1, PAGE)
            fails = 0
        except Exception as e:
            state.setdefault("errors", []).append(f"{time.strftime('%F %T')} {y} p{p}: {type(e).__name__} {str(e)[:150]}")
            fails = state.get("consec_fail", 0) + 1; state["consec_fail"] = fails
            if fails >= 3:
                state["stopped"] = f"{time.strftime('%F %T')} 연속 3회 오류 - 중단"
                json.dump(state, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
                print("중단:", state["stopped"]); sys.exit(1)
            time.sleep(30)
            rows = []
            for k in range(10):
                rows += fetch_split(p * PAGE + 1 + k * 100, 100)
        state["consec_fail"] = 0
        json.dump(state, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        ncall += 1
        df = pd.DataFrame(rows).astype("string")
        df["_page"] = p
        df.to_parquet(f, index=False)
        print(y, p, len(rows), f"{time.time()-T0:.0f}s", flush=True)
print("전체 완료")
