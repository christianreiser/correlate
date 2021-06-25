import datetime
import pylunar
mi = pylunar.MoonInfo((42, 21, 30), (-71, 3, 35))
mi.update((2021, 6, 23, 22, 0, 0))
base = datetime.datetime.today()
base = datetime.datetime(2019,2,12,22,00,00)
date_list = [datetime.timedelta(days=x) + base for x in range(870)]

#    mi.update(str(date_list[0].strftime('%Y,%m,%d,%H,%M,%S')))
for i in range(len(date_list)):
    mi.update(date_list[i])
    print(date_list[i])
    print(round(mi.fractional_phase(),2))
