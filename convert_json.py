import pandas as pd

# 엑셀 파일 불러오기
df = pd.read_excel("faq.xlsx")

# 3) 병합된 셀 NaN 처리 → 앞의 값으로 채우기
df = df.fillna(method="ffill")
# JSON 변환 (한글 깨짐 방지, 들여쓰기 적용)
df.to_json("faq.json", orient="records", force_ascii=False, indent=2)

print("변환 완료! data.json 파일 생성됨")