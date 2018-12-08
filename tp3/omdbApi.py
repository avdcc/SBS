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
AUX = '34b11df4' 
AUX2 = '1e6f36b'
AUX3 = 'e85a658a'

apiKey = AUX3

# SO MUDAR ISTO -----------------------------------------------------------

omdb.set_default('apikey', apiKey) 
d = {}

i = 0
with open(pathFile, 'rb') as csvfile:
    for row in csvfile:
        i+=1
        ref = row.decode("utf-8").strip("\n")
        res = omdb.get(imdbid=ref, fullplot=False, tomatoes=False)
        d[ref] = res

        #if(i==2):
         #  final = pandas.read_json(json.dumps(d, ensure_ascii=False))
         #  final.to_csv('resAuxiliar' + str(numero_link)+'.csv', header = True, index = True)
        #   exit()

final = pandas.read_json(json.dumps(d, ensure_ascii=False))
final.to_csv('resAux' + str(numero_link)+'.csv', header = True, index = True, sep = ';')
