# API 키를 API키_기록.txt 에서 읽는다 (값은 출력·저장하지 않음)
import re, pathlib
KF = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\API키_기록.txt")
def _section(name):
    t = KF.read_text(encoding='utf-8')
    i = t.index(name); j = t.find('\n[', i+1)
    return t[i: j if j > 0 else None]
def ecos_key():
    s = _section('[한국은행 ECOS')
    return re.search(r'키:\s*(\S+)', s).group(1)
