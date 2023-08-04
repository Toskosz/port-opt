import pandas as pd
from datetime import datetime

data = {}
with open('./data/COTAHIST_D01082023.TXT') as file:
    for line in file:
        try:
            code = line[12:24].strip()

            price = line[108:121].strip()
            price = int(price) / 100

            date_string = line[2:10].strip()
            datetime_object = datetime.strptime(date_string, '%Y%m%d')
            
            # If i just do data.get(code) python would evaluate {} as False
            if data.get(code, 0) == 0:
                data[code] = {}
            else:
                data[code][datetime_object.strftime('%d-%m-%Y')] = price
        except:
            continue

for asset in data:
    x = pd.Series(data=data[asset], index=list(data[asset].keys()))
    print(x)
    data[asset] = pd.Series(data=data[asset], index=list(data[asset].keys()))

data = pd.concat(data, axis=1)

print(data)

x=1  