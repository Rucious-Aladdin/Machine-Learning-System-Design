#2018112671 김수성

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

dir_path = "C:/Users/IT대학_000/HW1"
file_path_2014 = dir_path + "/" + str(2014) + ".csv"
file_path_2015 = dir_path + "/" + str(2015) + ".csv"
file_path_2016 = dir_path + "/" + str(2016) + ".csv"

df1 = pd.read_csv(file_path_2014, index_col="구분", encoding='cp949')
df2 = pd.read_csv(file_path_2015, index_col="구분", encoding='cp949')
df3 = pd.read_csv(file_path_2016, index_col="구분", encoding='cp949')

#1번
df = pd.concat([df1, df2, df3])


#2번
df["년도"] = [df.index[i][0:4] for i in range(len(df.index))]
df["월"] = [df.index[i][5:-1] for i in range(len(df.index))]

df = df.set_index(["년도", "월"])

#3번

print(df["사망(명)"].groupby(by="년도").mean())
print(df["사망(명)"].groupby(by="월").mean())

#4번
a = df.loc["2016"]["사고(건)"].sum()
print("전체사고(건):  %d" % a)

b = df.loc["2016"]["사망(명)"].sum()
print("사망자(명):  %d" % b)

print("사고대비 사망율: %.2f%%" % ((b / a) * 100))

#5번

y1 = df.loc["2014"]["사망(명)"]
y2 = df.loc["2014"]["부상(명)"]

months = range(1, 13)
bar_width = 0.4
bar_positions1 = np.arange(len(months))
bar_positions2 = bar_positions1 + bar_width
plt.figure(figsize=(10, 6))
plt.bar(bar_positions1, y1, label='사망자 수', width=bar_width, alpha=0.7)
plt.bar(bar_positions2, y2, label='부상자 수', width=bar_width, alpha=0.7)
plt.xlabel('월')
plt.ylabel('인원')
plt.title('2014년 월별 사망자 수 및 부상자 수')
plt.xticks(bar_positions1 + bar_width / 2, labels=[str(month) for month in months])
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.savefig("C:/Users/IT대학_000/HW1/hw1-5(5)")


#6번


# 2015년과 2016년의 데이터를 추출합니다.
y1 = df.loc["2015"]["사망(명)"]
y2 = df.loc["2016"]["사망(명)"]

# 각 월별 사망자 수의 차이를 계산합니다.
death_increase = y2 - y1

# 가장 많이 증가한 2개의 월을 구합니다.
death_increase = death_increase.nlargest(2)

print("2016년에 사망자 수가 가장 많이 증가한 월:")
for month, increase in death_increase.items():
    print("%s월, %d명 증가" % (month, increase))

