import datetime
import pandas as pd
import collections

id_value = collections.defaultdict(lambda: [])
with open("data.txt", "r") as file:
    for i in file:
        tmp = i[:-1].split(" ")
        if len(tmp) == 8 and tmp[3] != '' and int(tmp[3]) <= 58:
            id_value[tmp[3]].append(i[:-1])
