import sqlite3, pathlib
B=pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
def ro(rel):
    return sqlite3.connect(( B/rel).as_uri()+"?mode=ro",uri=True)
BR=lambda: ro("수집파이프라인/business_rights.sqlite")
DT=lambda: ro("공모전데이터/외부데이터/DART/dart_company.sqlite")
RG=lambda: ro("수집파이프라인/registry_full.sqlite")
