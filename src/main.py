
data = {}

with open('./data/COTAHIST_D01082023.txt') as file:
    for line in file:
        code = line[12:24].strip()
        price = line[108:121].strip()
        price = int(price) / 100
        data[code] = price


    