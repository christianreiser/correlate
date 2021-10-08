import pandas as pd
my_list = []
for i in range(1000):
    my_list.append(14.7498 - 13.2869 * (i+30) ** 0.0101585)
df = pd.DataFrame(my_list)
print(my_list)