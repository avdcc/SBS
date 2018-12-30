import pandas as pd
import csv
#utilizador = pd.read_csv("utilizador.csv", sep = ';')
#filmes  = pd.read_csv("utilizador.csv", sep = ';')

with open("votesMovie.csv", "rb") as fd1:
    votes = csv.reader(fd1, delimiter = ';')

with open("filmesn.csv") as fd2:
    filmes = csv.reader(fd2, delimiter = ';')

def stripall(string,List): return list(map(lambda x: x.strip(string),List))

def column(table,ind):
    res = []
    for i in range(len(table)):
        res.append(table[i][ind])
    return res

#print(utilizador)
#print(utilizador.userId)
#aux = utilizador.userId
#print(aux[0])
## --- ler os ficheiros-----
with open("votesMovie.csv") as fd1:
    votes = fd1.readlines()

with open("filmesn.csv") as fd2:
    filmes = fd2.readlines()
SEP = "§"
#aux = list(map(lambda x: len(x.split(SEP)),filmes))
#for x in range(len(aux)):
#    if(aux[x]==24):
#        print(x)
#print(min(aux))
#aux = list(map(lambda x: x.split(SEP),filmes))
#print(filmes[8539])
#print(len(aux[8539]))

votes = list(map(lambda x: stripall("\"",x.strip("\n").split(";")), votes))
filmes = list(map(lambda x: stripall("\"",x.strip("\n").split(SEP)), filmes))
#print(votes[0])
#print(utilizadores[0])
#print(column(utilizadores,0)) #print(column(filmes,0))



def createDic(table):
    dic = {}
    for x in range(len(table[0])):
        dic[ table[0][x] ] = column(table,x)[1:]
    return dic
#dicUtilizadores = createDic(utilizadores)
dicFilmes = createDic(filmes)
#print(utilizadores[0][2])
#print(dicFilmes.keys())
aux1 = []
#print(dicFilmes["Internet Movie Database"])
semvalor = 0
for x in dicFilmes["Internet Movie Database"]:
    if(x==""):
        aux1.append(0)
        semvalor +=1
    else:
        try:
            aux1.append(int(float(x.split("/")[0])*10))
        except:
            try:
                aux1.append(int(x.strip("%")))
            except:
                aux1.append(0)
                semvalor += 1
media =int( sum(aux1) / (len(aux1) - semvalor) )
#print(media)
res = []
for x in aux1:
    if(x==0):
        res.append(media)
    else:
        res.append(x)
#print(res)

dicFilmes["Internet Movie Database"] =res


#print(aux1)
aux2 = dicFilmes["Rotten tomatoes"]
res = []
semvalor = 0
for x in aux2:
    #print(x)
    A = x.split("/")
    if(x==""):
        res.append(-1)
        semvalor +=1
    #elif(x=="[]"): res.append("0/10")
    elif(len(A)>1):
        res.append( int((float(A[0]) / float(A[1]))*100))
    #    res.append( str((float(A[0])*float(A[1]))/10)+"/10" )
    #else:
    #    res.append( str(float(x)/10)+"/10" )
    else: 
        try:
            res.append(int(str(int(x)/10))) 
        except:
            try:
                res.append(int(x))
            except:
                res.append(-1)
                semvalor +=1
#print(res)
media =int( sum(res) / (len(res) - semvalor) )
#print(media)
res2 = []
for x in res:
    if(x==-1):
        res2.append(media)
    else:
        res2.append(x)
media =int( sum(res2) / (len(res2)) )
#print(media)
dicFilmes["Rotten tomatoes"]=res2

aux2 = dicFilmes["Metacritic"]
res = []
semvalor = 0
for x in aux2:
    #print(x)
    A = x.split("/")
    if(x==""):
        res.append(0)
        semvalor +=1
    #elif(x=="[]"): res.append("0/10")
    elif(len(A)>1):
        res.append( int((float(A[0]) / float(A[1]))*100))
    #    res.append( str((float(A[0])*float(A[1]))/10)+"/10" )
    #else:
    #    res.append( str(float(x)/10)+"/10" )
    else: 
        #try:
        #    res.append(int(int(x)/10))
        #except:
        res.append(int(x))
media =int( sum(res) / (len(res) - semvalor) )
#print(media)
res2 = []
for x in res:
    if(x==0):
        res2.append(media)
    else:
        res2.append(x)
media =int( sum(res2) / (len(res2)) )
#print(media)
#print(res)
dicFilmes["Metacritic"]=res2

#print(dicFilmes["metascore"])
res = []
semvalor = 0
for x in dicFilmes["metascore"]:
    if(x=="N/A"):
        res.append(0)
        semvalor += 1
    else:
        res.append(int(x))
#print(res)
media =int( sum(res) / (len(res) - semvalor) )
#print(media)
res2 = []
for x in res:
    if(x==0):
        res2.append(media)
    else:
        res2.append(x)
media =int( sum(res2) / (len(res2)) )
#print(media)
dicFilmes["metascore"] = res2





#print(dicFilmes["imdb_rating"])
#print(dicFilmes["metascore"])
res = []
semvalor = 0
for x in dicFilmes["imdb_rating"]:
    if(x=="N/A"):
        res.append(0)
        semvalor += 1
    else:
        res.append(int(float(x)*10))
#print(res)
media =int( sum(res) / (len(res) - semvalor) )
#print(media)
res2 = []
for x in res:
    if(x==0):
        res2.append(media)
    else:
        res2.append(x)
media =int( sum(res2) / (len(res2)) )
#print(media)
dicFilmes["imdb_rating"]=res2


#print(res2)
#print(dicUtilizadores["ratings"][0])
keys = list(dicFilmes.keys())
#print(SEP.join(keys))
#for i in range(len(dicFilmes[ "Metacritic" ])):
#    for x in keys:
#        if(x == keys[len(keys)-1]):
#            print(dicFilmes[x][i], end ="")
#        else:
#            print(str(dicFilmes[x][i])+SEP, end ="")
#    print("")


#print(dicFilmes['imdb_id'])
#print(len(votes))
#print(len(dicFilmes['imdb_id']))
res = []
for x in votes:
    if(x[0] in dicFilmes['imdb_id']):
        res.append(x)
#print(len(res))
print(";".join(votes[0]))
for i in res:
    print(";".join(i))
#print(votes)
