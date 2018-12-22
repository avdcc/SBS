import pandas as pd
import csv
#utilizador = pd.read_csv("utilizador.csv", sep = ';')
#filmes  = pd.read_csv("utilizador.csv", sep = ';')

with open("utilizador.csv", "rb") as fd1:
    utilizador = csv.reader(fd1, delimiter = ';')

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
with open("utilizador.csv") as fd1:
    utilizadores = fd1.readlines()

with open("filmesn.csv") as fd2:
    filmes = fd2.readlines()

utilizadores = list(map(lambda x: stripall("\"",x.strip("\n").split(";")), utilizadores))
filmes = list(map(lambda x: stripall("\"",x.strip("\n").split(";")), filmes))
#print(utilizadores[0])
#print(column(utilizadores,0)) #print(column(filmes,0))



def createDic(table):
    dic = {}
    for x in range(len(table[0])):
        dic[ table[0][x] ] = column(table,x)[1:]
    return dic
dicUtilizadores = createDic(utilizadores)
dicFilmes = createDic(filmes)
#print(utilizadores[0][2])
#print(dicFilmes.keys())
aux1 = []
#print(dicFilmes["Internet Movie Database"])
for x in dicFilmes["Internet Movie Database"]:
    if(x==""): aux1.append("0")
    else:
        try:
            aux1.append(str(float(x.split("/")[0])*10))
        except:
            aux1.append(x.strip("%"))
dicFilmes["Internet Movie Database"] =aux1

#print(aux1)
aux2 = dicFilmes["Rotten tomatoes"]
res = []
for x in aux2:
    #print(x)
    A = x.split("/")
    if(x==""): res.append("0")
    #elif(x=="[]"): res.append("0/10")
    elif(len(A)>1):
        res.append( str((float(A[0]) / float(A[1]))*100))
    #    res.append( str((float(A[0])*float(A[1]))/10)+"/10" )
    #else:
    #    res.append( str(float(x)/10)+"/10" )
    else: 
        #try:
        #    res.append(str(int(x)/10) + "/10")
        #except:
        res.append(x)
#print(res)
dicFilmes["Rotten tomatoes"]=res

aux2 = dicFilmes["Metacritic"]
res = []
for x in aux2:
    #print(x)
    A = x.split("/")
    if(x==""): res.append("0")
    #elif(x=="[]"): res.append("0/10")
    elif(len(A)>1):
        res.append( str((float(A[0]) / float(A[1]))*100))
    #    res.append( str((float(A[0])*float(A[1]))/10)+"/10" )
    #else:
    #    res.append( str(float(x)/10)+"/10" )
    else: 
        #try:
        #    res.append(str(int(x)/10) + "/10")
        #except:
        res.append(x)
#print(res)
dicFilmes["Metacritic"]=res
#print(dicUtilizadores["ratings"][0])
keys = list(dicFilmes.keys())
print(";".join(keys))
for i in range(len(dicFilmes[ "Metacritic" ])):
    for x in keys:
        if(x == keys[len(keys)-1]):
            print(dicFilmes[x][i], end ="")
        else:
            print(dicFilmes[x][i]+";", end ="")
    print("")

