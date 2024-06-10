from finance_get import GetData

gett = GetData()
df = gett.func_start()
print(df.drop('vol',axis=1).head())