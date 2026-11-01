import json, os
W = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C10\C10-T4"
s = json.load(open(os.path.join(W, "submission_check.json"), encoding="utf-8"))
s["test_builds"] = json.load(open(os.path.join(W, r"build_test\test_builds.json"), encoding="utf-8"))
json.dump(s, open(os.path.join(W, "submission_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok")
