
import matplotlib.pyplot as plt
import random
import numpy as np

# --------------------------------------------

def save(lista):

  print(len(lista), len(lista[0]) ,len(set(map(lambda x: x[-1], lista))))
  for i in lista:
    print (" ".join(map(str,i)))


def pprint(lista):

  x = list(map(lambda x: x[0], lista))
  y = list(map(lambda x: x[1], lista))

  if (len(lista[0]) == 3):
    c = list(map(lambda x: x[2], lista))
    plt.scatter(x,y,c=c)
  else:
    z = list(map(lambda x: x[2], lista))
    c = list(map(lambda x: x[3], lista))
    plt.scatter(x,y,z,c=c)

  plt.show()

# --------------------------------------------
# LINHA

def avalLinha(x,y,lista):

  res = 0
  for i in range(len(lista[:-1])):
    res += x**(len(lista)-i -1) * lista[i]
  res += lista[-1]

  if (res <= y):
    return 1
  else:
    return 0


def gen(n, lista, gen):
  # y = mx + b
  l = []
  for i in range(n):

    x = random.uniform(-1,1)
    y = random.uniform(-1,1)

    if (gen == 3):
      z = random.uniform(-1,1)
      c = avalQuadrantes(x,y,z,lista)
      l.append((x,y,z,c))
    else:
      c = avalLinha(x,y,lista)
      # c = avalQuadrantes2(x,y,lista)
      l.append((x,y,c))

  return (l)

# --------------------------------------------
# QUADRANTES

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
# DONUT

# def donut (n, r, R, mx, my):

  # for i in range(0,n//2):










# --------------------------------------------

plt.xkcd()
save(gen(1500, [1,2,1,0], 2))
# pprint3(gen(1500, [0,1,2,3,4,5,6,7]),3)