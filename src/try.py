from finance_get import GetData

gett = GetData(ticker_symbol='NVDA')
df = gett.get_data_df()
val, vol = gett.val_vol_datasets()

print(val.head())
print(vol.head())
print(df.head())