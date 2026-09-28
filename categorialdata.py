import pandas as pd


data ={'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],}
df = pd.DataFrame(data)
df_encoded = pd.get_dummies(df, columns=['Color'])
print('Original DataFrame:')
print(df)
print('\nEncoded DataFrame:')
print(df_encoded)