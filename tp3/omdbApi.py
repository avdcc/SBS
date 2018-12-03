
import omdb
import csv
import pandas
import json


# SO MUDAR ISTO -----------------------------------------------------------
pathFile = './ml-latest-small/imdblinks0.csv'

apiKey = 'd6f82b99'
# ADRIANO = d6f82b99
# LUIS = ?
# EZEQUIEL = ?

# SO MUDAR ISTO -----------------------------------------------------------

omdb.set_default('apikey', apiKey) 
d = {}

with open(pathFile, 'rb') as csvfile:
    for row in csvfile:

        ref = row.decode("utf-8")[0:-1]
        res = omdb.get(imdbid=ref, fullplot=False, tomatoes=False)
        d[ref] = res


final = pandas.read_json(json.dumps(d, ensure_ascii=False))
print(final)



