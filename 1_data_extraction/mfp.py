from datetime import datetime

import numpy as np
import pandas as pd

from config import private_folder_path

output_name = str(private_folder_path)+'mfp_daily_summaries.csv'
df = pd.read_csv('/home/chrei/PycharmProjects/correlate/0_data_raw/MFP/meals.csv')  # , index_col=0
currentDay = datetime.strptime(df.columns[0], '%B %d, %Y')
first_date = currentDay
df.columns = df.iloc[0]
sugar = []
Cholest = []
date = []

# get date
for _, row in (df.iterrows()):
    try:
        currentDay = datetime.strptime(row['FOODS'], '%B %d, %Y')
    except:
        pass
    date.append(currentDay)

# remove units
cols_to_check = ['Calories', 'Cholest', 'Sugars', 'Protein', 'Fat', 'Carbs', 'Sodium', 'Fiber']
df[cols_to_check] = df[cols_to_check].replace({'--': 0}, regex=True)
df[cols_to_check] = df[cols_to_check].replace({',': ''}, regex=True)
df[cols_to_check] = df[cols_to_check].replace({np.nan: 0}, regex=True)
df[['Cholest', 'Sodium']] = df[['Cholest', 'Sodium']].replace({'mg': ''}, regex=True)
df[cols_to_check] = df[cols_to_check].replace({'g': ''}, regex=True)

# add date
df['Date'] = date

# remove unwanted headers inside of df
df = df[df.Calories != 'Calories']

# drop foods
# df.reset_index(level=0, inplace=True)
df = df.drop(['FOODS'], axis=1)  # , 'Calories', 'Cholest', 'Protein', 'Fat', 'Carbs', 'Sodium', 'Fiber'

# to int conversion
df[['Calories', 'Cholest', 'Suars', 'Protein', 'Fat', 'Carbs', 'Sodium', 'Fiber']] = df[
    ['Calories', 'Cholest', 'Suars', 'Protein', 'Fat', 'Carbs', 'Sodium', 'Fiber']].astype('int32')

# aggregate
df = df.groupby(df['Date']).sum()

# fill missing dates with NaN
df.reset_index(level=0, inplace=True)
lastDate = currentDay
df = df.set_index('Date')  # set date as index
idx = pd.date_range(first_date, lastDate)
df.index = pd.DatetimeIndex(df.index)
df = df.reindex(idx, fill_value=np.nan)

# save
df.to_csv(output_name)
print(str(output_name) + ' written')

print()
