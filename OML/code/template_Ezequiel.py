#!python
'''
logistic_classifier
'''
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
#para nº aleatórios
from random import randint

# ============ FILE load and write stuff ===========================
#def load_csv(filename):
#    dataset = list()
#    with open(filename, 'r') as file:
#        csv_reader = reader(file)
#        for row in csv_reader:
#            if not row:
#                continue
#            dataset.append(row)
#    return dataset

def read_asc_data(filename):
    f= open(filename,'r')
    tmp_str=f.readline()
    tmp_arr=tmp_str[:-1].split(' ')
    N=int(tmp_arr[0]);n_row=int(tmp_arr[1]);n_col=int(tmp_arr[2])
    print("N=%d, row=%d, col=%d" %(N,n_row,n_col))
    data=np.zeros([N,n_row*n_col+1])
    for n in range(N):
        tmp_str=f.readline()
        tmp_arr=tmp_str[:-1].split(' ')
        for i in range(n_row*n_col+1):
            data[n][i]=int(tmp_arr[i])
    f.close() 
    return N,n_row,n_col,data

def plot_data(row,col,n_row,n_col,data):
    fig=plt.figure(figsize=(row,col))
    for n in range(1, row*col +1):
        img=np.reshape(data[n-1][:-1],(n_row,n_col))
        fig.add_subplot(row, col, n)
        plt.imshow(img,interpolation='none',cmap='binary')
    plt.show()

def plot_tagged_data(row,col,n_row,n_col,X,Y,ew):
    fig=plt.figure(figsize=(row,col))
    for n in range(row*col):
        img=np.reshape(X[n],(n_row,n_col))
        fig.add_subplot(row, col, n+1)
        #if(Y[n]>0):#exact case
        if(predictor(X[n],ew)>0.5):
            plt.imshow(img,interpolation='none',cmap='RdPu')
        else:
            plt.imshow(img,interpolation='none',cmap='cool')
    plt.show()

def plot_error(err):
    plt.plot(range(len(err)), err, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Number of misclassifications')
    plt.ylim([0,5])
    plt.show()
    return

def confusion(Xeval,Yeval,N,ew):
    C=np.zeros([2,2]);
    for n in range(N):
        y=predictor(Xeval[n],ew);
        if(y<0.5 and Yeval[n]<0.5): C[0,0]=C[0,0]+1;
        if(y>0.5 and Yeval[n]>0.5): C[1,1]=C[1,1]+1;
        if(y<0.5 and Yeval[n]>0.5): C[1,0]=C[1,0]+1;
        if(y>0.5 and Yeval[n]<0.5): C[0,1]=C[0,1]+1;
    return C

def sigmoid(s):
    large=30
    if s<-large: s=-large
    if s>large: s=large
    return (1 / (1 + np.exp(-s)))


def recall(C):
    return (C[0,0]) / (C[0,0] + C[1,0])

def accuracy(C):
    return (C[0,0] + C[1,1])/(C[0,0] + C[0,1] + C[1,0] + C[1,1])

def precision(C):
    return (C[0,0]) / (C[0,0] + C[0,1])

#============== Logistic classifier Stuff ==================

def predictor(x,ew):
    #retirar 1 do inicios de ew(W tilde) para obter W
    s = ew[0]
    #calcular produto interno entre X e W e adicionar a s
    #isto é porque no calculo do dot, as 2 primeiras componentes de X_tilde e W_tilde são 1,
    #pelo que o seu valor no produto interno é 1(que corresponde a s)
    s = s + np.dot(x,ew[1:])
    #calcular probabilidade final e returnar usando sigmoid
    res = sigmoid(s)
    return res


def cost(X,Y,N,ew):
    #calcular sumatório
    soma = 0
    #threashold
    epsi = 1e-12
    for i in range(N):
        #elementos atuais
        #esperado
        y_i = Y[i]
        #previsto
        y_p_i = predictor(X[i],ew)
        #verificar threashold
        if y_p_i < epsi : y = epsi
        if y_p_i > 1 - epsi : y = 1 - epsi
        #calcular primeiro elemento da soma
        sum1 = y_i*np.log(y_p_i)
        #calcular segundo elemento da soma
        sum2 = (1 - y_i)*np.log(1 - y_p_i)
        #sumar ambos à soma final com o abs
        soma = soma + sum1 + sum2
    #no final returnar a média
    med = -soma/N
    return med


def update(x,y,eta,ew):
    #calcular previsão para x
    r=predictor(x,ew)
    #calcular s(gradiente)
    s=(y-r)
    #calculos para melhorar previsão
    r=2*(r-0.5)
    #multiplicar com eta tendo em conta r
    s=s*eta/(1+3.7*r*r)
    #atualizar ew
    ew[0]=ew[0]+s
    ew[1:]=ew[1:]+s*x
    #returnar novo W tilde
    return ew




def run_stocastic(X,Y,N,eta,MAX_ITER,ew,err):
    #valor de paragem para o erro
    err_stop_value = 1e-8
    #valor de iteração atual
    j = 0
    #enquanto o erro é maior que certo valor e não passamos das iterações máximas
    while((err[-1] > err_stop_value) and (j<MAX_ITER)):
        #obter elemento da base de dados
        ri = randint(0,N-1)
        #extrair dados de dito elemento
        x_r = X[ri]
        y_r = Y[ri]
        #update do eta
        new_eta = eta * math.exp(-j/850)
        #fazer update
        ew = update(x_r,y_r,new_eta,ew)
        #debug
        print('iter %d, cost=%f, eta=%e     \r' %(j,err[-1],new_eta),end='')
        #atualizar variáveis
        j = j+1
        err.append( cost(X,Y,N,ew) )
    #returnar os dados no final
    return ew,err



#=========== MAIN CODE ===============
# read the data file
#N,n_row,n_col,data=read_asc_data('./dataset/AND.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/XOR.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/rectangle60.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/rectangle600.txt')
N,n_row,n_col,data=read_asc_data('./dataset/line600.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/line1500.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/my_digit.txt');np.place(data[:,-1], data[:,-1]!=1, [-1])
print('find %d images of %d X %d pixels' % (N,n_row,n_col))

#plot_data(10,6,n_row,n_col,data)

training_percentage = 0.5
Nt=int(N*training_percentage)
I=n_row*n_col
Xt=data[:Nt,:-1];Yt=data[:Nt,-1]
ew=np.ones([I+1])
err=[];err.append(cost(Xt,Yt,Nt,ew))

ew,err=run_stocastic(Xt,Yt,Nt,1,200,ew,err)
ew,err=run_stocastic(Xt,Yt,Nt,0.1,199,ew,err)
ew,err=run_stocastic(Xt,Yt,Nt,0.03,199,ew,err)
plot_error(err)

print('in-samples error=%f ' % (cost(Xt,Yt,Nt,ew)))
C =confusion(Xt,Yt,Nt,ew)
print("Confusion matrix:")
print(C)
print("in-samples confusion matrix evaluations (recall,accuracy,precision) = (",recall(C),",",accuracy(C),",",precision(C),")")

print("\n\n")

Ne=N-Nt;Xe=data[Nt:N,:-1];Ye=data[Nt:N,-1];
print('out-samples error=%f' % (cost(Xe,Ye,Ne,ew)))
C =confusion(Xe,Ye,Ne,ew)
print("Confusion matrix:")
print(C)
print("out-samples confusion matrix evaluations (recall,accuracy,precision) = (",recall(C),",",accuracy(C),",",precision(C),")")


print('bye')
