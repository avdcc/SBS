
import random

l = [(-1, -1, 0)
,(-1, 1, 0)
,(1 ,-1, 0)
,(1 ,1 ,1)]

for i in range(len(l)):
  for a in range(10):
    rand_x = random.random() * l[i][0]
    rand_y = random.random() * l[i][1]
    if(rand_x> 0 and rand_y > 0):
      tag = 1
    else:
      tag = 0
    l.append((rand_x,rand_y,tag))
  
file = open("./dataset/CAND.txt","a")

for w in l:
  txt = str(w[0]) + " " + str(w[1]) + " " + str(w[2]) + "\n"
  file.write(txt)


#print( [(x,y,1)  for x in range(-0.5,0.5) for y in range(-0.5,0.5) if(x>0 and y>0)] )
