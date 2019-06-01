
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np

# --------------------------------------------

def saveImage(lista, path):

  x = list(map(lambda x: x[0], lista))
  y = list(map(lambda x: x[1], lista))

  if (len(lista[0]) == 3):
    c = list(map(lambda x: x[2], lista))
    plt.scatter(x,y,c=c)
  else:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    z = list(map(lambda x: x[2], lista))
    c = list(map(lambda x: x[3], lista))
    ax.scatter(x, y, z, c=c)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

  plt.savefig(path)


"""
## Mostrar o grafico ##
    INPUT:
       * "lista" -- lista de n_uplos , sendo que as primeiras
                    componentes correspondem as cordenadas
                    e a ultima ao label / cor
         
    RETURN:
       * "Pontos" -- escreve no std output os pontos e a sua classe
"""
def save(lista):

  print(len(lista), len(lista[0]) - 1 ,len(set(map(lambda x: x[-1], lista))) - 1)
  for i in lista:
    print (" ".join(map(str,i)))

"""
## Mostrar o grafico ##
    INPUT:
       * "lista" -- lista de n_uplos , sendo que as primeiras
                    componentes correspondem as cordenadas
                    e a ultima ao label / cor
"""
def pprint(lista):

  x = list(map(lambda x: x[0], lista))
  y = list(map(lambda x: x[1], lista))

  if (len(lista[0]) == 3):
    c = list(map(lambda x: x[2], lista))
    plt.scatter(x,y,c=c)
  else:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    z = list(map(lambda x: x[2], lista))
    c = list(map(lambda x: x[3], lista))
    ax.scatter(x, y, z, c=c)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

  plt.show()


# --------------------------------------------
# LINHA
"""
## falcular o ponto na funcao##
    INPUT:
       * "x" -- valor de x

       * "eq" -- equacao da funcao
                onde e representada como uma lista de coeficientes
         
    RETURN:
       * "y" -- valor da funcao
"""
def f(x,eq):
  res = 0
  for i in range(len(eq[:-1])):
    res += x**(len(eq)-i -1) * eq[i]
  res += eq[-1]
  return res

"""
## Avalia o ponto em funcao da linha ##
    INPUT:
       * "x" -- valor de x
       * "y" -- f(x)

       * "eq" -- [f] equacao da funcao
                onde e representada como uma lista de coeficientes
         
    RETURN:
       * "res" -- tag
"""
def avalLinha(x,y,eq):
  res = f(x,eq)

  if (res <= y):
    return 1
  else:
    return 0


"""
## gera os pontos ##
    INPUT:
       * "n" -- numero de pontos
       * "eq" -- [f] equacao da funcao
                onde e representada como uma lista de coeficientes
       * "g" -- dimensao
       * "m" -- margem
         
    RETURN:
       * "res" -- conjunto de pontos
"""
def gen(n, eq, g,m=0):
  # y = mx + b
  l = []
  for i in range(n):

    while(True):
        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        if(g==2):
            if( abs(y - f(x,eq)) > m ):
                break
        elif(g==3):
            z = random.uniform(-1,1)
            if( (abs(x) > m) and (abs(y) > m) and (abs(z) > m) ):
                break
        else:
            if( (abs(x) > m) and (abs(y) > m)):
                break

    if (g == 3):
      c = avalQuadrantes3(x,y,z,eq)
      l.append(((x/2) + 1/2,(y/2) + 1/2,(z/2) + 1/2,c))
    elif(g ==2):
      c = avalLinha(x,y,eq)
      l.append(((x/2) + 1/2,(y/2)+ 1/2,c))
    else:
      c = avalQuadrantes2(x,y,eq)
      l.append(((x/2)+1/2,(y/2)+1/2,c))

  return (l)

# --------------------------------------------
# QUADRANTES

"""
## Avalia o ponto em funcao do Quadrante ##
    INPUT:
       * "x" -- valor de x
       * "y" -- f(x)

       * "lista" -- lista com as classes
                    ou seja lista[i] = classe dos valores no quadrante i+1
         
    RETURN:
       * "res" -- tag
"""
def avalQuadrantes2(x,y,lista):
  # len(lista) = 4
  if(x > 0 and y > 0):
    return lista[0]
  elif(x < 0 and y > 0):
    return lista[1]
  elif(x < 0 and y < 0):
    return lista[2]
  elif(x > 0 and y < 0):
    return lista[3]
  else:
    print("...")

"""
## Avalia o ponto em funcao do Quadrante ##
    INPUT:
       * "x" -- valor de x
       * "y" -- valor de y
       * "z" -- valor de z

       * "lista" -- lista com as classes
                    ou seja lista[i] = classe dos valores no quadrante i+1
         
    RETURN:
       * "res" -- tag
"""
def avalQuadrantes3(x,y,z,lista):
  # len(lista) = 8
  if(x > 0 and y > 0 and z > 0):
    return lista[0]
  elif(x < 0 and y > 0 and z > 0):
    return lista[1]
  elif(x < 0 and y < 0 and z > 0):
    return lista[2]
  elif(x > 0 and y < 0 and z > 0):
    return lista[3]
  elif(x > 0 and y > 0 and z < 0):
    return lista[4]
  elif(x < 0 and y > 0 and z < 0):
    return lista[5]
  elif(x < 0 and y < 0 and z < 0):
    return lista[6]
  elif(x > 0 and y < 0 and z < 0):
    return lista[7]
  else:
    print("...")


# --------------------------------------------
# IMAGES

def file2Image(path):

  file = open(path, "r") 

  aux = file.readline()
  dim = int(aux.split(" ")[1])

  res = []

  for l in file.readlines():
    if len(l) > 2:
      res.append(tuple(map(float,l.rstrip().split(" "))))  

  #pprint(res)
  saveImage(res,path[:-4]+".png")

# --------------------------------------------

#plt.xkcd()

file2Image('lin.txt')


#x(lin)
#pprint(gen(150, [1,0], 2,0.2))
#save(gen(150, [1,0], 2,0.2))

#3*x + 1(3linP1)
#pprint(gen(150, [3,1], 2,0.2))
#save(gen(150, [3,1], 2,0.2))

#x^2(sqr)
#pprint(gen(150, [1,0,0], 2,0.2))
#save(gen(150, [1,0,0], 2,0.2))

#x^2 + x(sqrPlin)
#pprint(gen(150, [1,1,0], 2,0.2))
#save(gen(150, [1,1,0], 2,0.2))

#x^3(cubed)
#pprint(gen(150, [1,0,0,0], 2,0.2))
#save(gen(150, [1,0,0,0], 2,0.2))

#OR
#pprint(gen(150, [1,1,0,1],1,0.2))
#save(gen(150, [1,1,0,1],1,0.2))


#OR 3D
#pprint(gen(300, [1,1,1,1,1,1,0,1],3,0.2))
#save(gen(300, [1,1,1,1,1,1,0,1],3,0.2))

#AND 3D
# pprint(gen(300, [1,0,0,0,0,0,0,0],3,0.2))
# save(gen(300, [1,0,0,0,0,0,0,0],3,0.2))

# pprint(gen(1500, [1,2,3,4],1,0.2))
#pprint(gen(19500, [1,2,3,4,5,6,7,8],3,0.3))
#print(gen(1500, [1,2,3,4,5,6,7,8],3,0))

# import glob

# for f in glob.glob("*.txt"):
#   print(f)
#   if f not in ["OR.txt","CAND.txt","sqr.txt","XOR.txt","3linP1.txt","sqr_big.txt","lin.txt"] :
#     file2Image(f)
