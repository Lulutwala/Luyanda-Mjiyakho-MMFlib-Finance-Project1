import pandas as pd

path = r"D:\MASTERS\Luyanda Mjiyakho Project1\Luyanda-Mjiyakho-MMFlib-Finance-Project1\data\News\newsapi_recent_news.csv"
df = pd.read_csv(path)

print(df.columns)
print(df.head())
