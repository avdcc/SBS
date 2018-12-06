import omdb
import csv
import pandas
import json


# SO MUDAR ISTO -----------------------------------------------------------
numero_link = 0
pathFile = './ml-latest-small/' + 'links' + str(numero_link) + '.csv'

ADRIANO = 'd6f82b99'
ADRIANO2 = ''
LUIS = '3da75bf3'
EZEQUIEL = '14459687'

apiKey = LUIS

# SO MUDAR ISTO -----------------------------------------------------------

omdb.set_default('apikey', apiKey) 
d = {}

i = 0
with open(pathFile, 'rb') as csvfile:
    for row in csvfile:
        i+=1
        ref = row.decode("utf-8")[0:-1]
        res = omdb.get(imdbid=ref, fullplot=False, tomatoes=False)
        d[ref] = res

        #if(i==3):
        #   final = pandas.read_json(json.dumps(d, ensure_ascii=False))
        #   final.to_csv('ups.csv', header = True, index = False)

final = pandas.read_json(json.dumps(d, ensure_ascii=False))
final.to_csv('resultado.csv', header = True, index = True)

