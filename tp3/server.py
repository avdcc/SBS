import pandas as pd
utilizador = pd.read_csv("utilizador.csv", sep = ';')
filmes  = pd.read_csv("utilizador.csv", sep = ';')
#print(list(utilizador))
#print(utilizador.userId)
aux = utilizador.userId
print(aux[0])
## --- ler os ficheiros-----
#with open("utilizador.csv") as fd1:
#    utilizadores = fd1.readlines()
#
#with open("filmes.csv") as fd2:
#    filmes = fd2.readlines()
