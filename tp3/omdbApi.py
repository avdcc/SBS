import omdb
import csv
import pandas
import json


# SO MUDAR ISTO -----------------------------------------------------------
numero_link = 3
pathFile = './ml-latest-small/' + 'link' + str(numero_link) + '.csv'

ADRIANO = 'd6f82b99'
ADRIANO2 = 'c4bf8b93'
LUIS = '3da75bf3'
LUIS2 = 'fb25258'
LUIS3 = 'ade8eee5'
EZEQUIEL = '14459687'

apiKey = ADRIANO

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

        #if(i==2):
        #   final = pandas.read_json(json.dumps(d, ensure_ascii=False))
        #   final.to_csv('res' + str(numero_link)+'.csv', header = True, index = False)
        #   exit()

final = pandas.read_json(json.dumps(d, ensure_ascii=False))
final.to_csv('res' + str(numero_link)+'.csv', header = True, index = False)

