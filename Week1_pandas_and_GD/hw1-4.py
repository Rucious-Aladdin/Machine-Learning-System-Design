#2018112571 김수성
import pandas as pd

data = {
'year':[2016, 2017, 2018],
'car': ['그랜저', '그랜저', '소나타'],
'name': ['홍길동', '고길동', '김둘리' ],
'number' : ['123하4567', '123허4567', '123호4567']
}

df = pd.DataFrame(data)

df.loc[3] = [2017, "일론", "테슬라", "987하6543"]

print(df["year"])
print(df["car"])
print(df["number"])

print(df["year" < 2017])