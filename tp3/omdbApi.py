
import omdb
import csv

omdb.set_default('apikey', 'd6f82b99') #MUDAR PARA CADA APIKEY

with open('./dif.csv', 'rb') as csvfile:
    for row in csvfile:
        #res = omdb.get(imdbid=row, fullplot=False, tomatoes=False)
        #print res
        print(row)

    

