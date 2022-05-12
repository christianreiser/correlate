import datetime
import pylunar
"""
This script is used to extract the moon illumination data for all dates between "base" and "today".
data is printed and can be copy pasted in to csv.
uncomment printing date-list temporarily and check at https://www.timeanddate.com/moon/phases/germany/stuttgart if correct

"""
mi = pylunar.MoonInfo((42, 21, 30), (-71, 3, 35)) # stuttgart location
today = datetime.datetime.today()
base = datetime.datetime(2019,2,11,22,00,00)
timedelta = today  - base
date_list = [datetime.timedelta(days=x) + base for x in range(int(timedelta.days+2))]

for i in range(len(date_list)):
    mi.update(date_list[i])

    # print(date_list[i])
    print(round(mi.fractional_phase(),2))





