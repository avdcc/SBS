#!python
'''
stocastic_logistic_classifier with dual method
'''
import csv
import numpy as np
import matplotlib.pyplot as plt
import math


#nota: este ficheiro é para a versão estocática do algoritmo dual(primal em logical_classifier.py)
#a versão batch requer estratégia ligeiramente diferente(versão primal deste em template2.py)











# ============ FILE load and write stuff ===========================


#nota: as coisas nesta secção ainda não foram alteradas






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
    
def plot_tagged_data(row,col,n_row,n_col,X,Y,al,N): 
    fig=plt.figure(figsize=(row,col))
    for n in range(row*col):
        img=np.reshape(X[n],(n_row,n_col))
        fig.add_subplot(row, col, n+1)
        #if(Y[n]>0):#exact case
        if(predictor(X[n],X,al,N)>0.5):
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







#============== Confusion matrix ==================

def confusion(Xeval,Yeval,N,al):
    C=np.zeros([2,2])
    for n in range(N):
        y=predictor(Xeval[n],Xeval,al,N)
        if(y<0.5 and Yeval[n]<0.5): C[0,0]=C[0,0]+1
        if(y>0.5 and Yeval[n]>0.5): C[1,1]=C[1,1]+1
        if(y<0.5 and Yeval[n]>0.5): C[1,0]=C[1,0]+1
        if(y>0.5 and Yeval[n]<0.5): C[0,1]=C[0,1]+1
    return C

        



#============== Logistic classifier Stuff ==================


#retorna o valor de aplicar a função sigmoid ao argumento que lhe é passado
def sigmoid(s):
  # garantir que se o valor de s for demasiado afastado do 0 é maximizado por certo valor(30 neste caso) 
  large=30
  if s<-large: s=-large
  if s>large: s=large
  # returnar valor do sigmoid de s
  return (1 / (1 + np.exp(-s)))



#versão primal
#def predictor(x,ew):
#    s=ew[0]
#    s=s+np.dot(x,ew[1:])
#    sigma=sigmoid(s)
#    return sigma

#versão dual
# dado X,um x seu elemento e um array al de pesos aplicados a cada valor xi 
# retorna a previsão feita para dito valor 
#corresponde a sigmoid da transposta de sum_i(al_i*x_tilde_i) com x_tilde, tendo em conta os tildes
def predictor(x,X,al,N):
  #relembrar que x não está em forma tilde quando é passado a esta função
  #assim, começamos a nossa soma tirando o primeiro elemento de al, que está sobre forma tilde
  s=al[0]
  #em seguida calculamos o nosso sumatório de al com x
  #para isto funcionar bem temos de começar o valor do sumatório com 
  #o primeiro passo já calculado(por causa de x0 ser array numpy)
  x0 = X[0]
  al0 = al[1]
  sum_xi_ali = x0 * al0
  #agora sumamos com o resto dos elementos
  for i in range(1,N):
    #obter elementos atuais
    #relembrar que cada elemento de x é um array numpy de valores 
    #e que cada elemento de al é uma constante
    x_i = X[i]
    al_i = al[i+1]
    #adicionar valores ao sumatório multiplicados
    sum_xi_ali += x_i * al_i
  #finalmente fazemos o produto dot entre x e o sumatório que criamos
  s=s+np.dot(x,sum_xi_ali)
  #prevemwa a previsão para o nosso valor
  sigma=sigmoid(s)
  #e returnamos a previsão feita
  return sigma



#versão primal
#def cost(X,Y,N,ew):
#    En=0;epsi=1.e-12
#    for n in range(N):
#        y=predictor(X[n],ew)
#        if y<epsi: y=epsi
#        if y>1-epsi:y=1-epsi
#        En=En+Y[n]*np.log(y)+(1-Y[n])*np.log(1-y)
#    En=-En/N
#    return En

#versão dual
#dado a matriz X de features ,o array Y de valores que queremos obter
#, o número de linhas N da matriz X e a lista de valores alpha al
#calcula o custo associado(i.e., a perda que o modelo atual tem)
def cost(X,Y,N,al):
  #variaveis auxiliares
  #valor da perda, inicializado a 0
  En=0
  #garante que os valores previstos nunca serão considerados demasiado perto de 0 ou 1
  #(a razão para isto tem a ver com o facto de querermos suavizar a curva de previsões)
  epsi=1.e-12
  #para cada linha de x
  for n in range(N):
    #prevemos o valor de y associado
    y_n = predictor(X[n],X,al,N)
    #normalizamos o valor
    if y_n < epsi: y_n = epsi
    if y_n > 1-epsi: y_n = 1-epsi
    #calculamos a perda obtida pelo modelo
    #usando a formula y_n*log(y^_n) + (1-y_n)*log(1-y^_n)
    #note-se que a formula real é o inverso desta na reta real(i.e., multiplicada por -1)
    #, mas isto será resolvido mais abaixo
    err_n = Y[n]*np.log(y_n)+(1-Y[n])*np.log(1-y_n)
    En = En + err_n
  #finalmente ajustasmos aquilo que foi referido acima (do -1 estar em falta)
  #e fazemos a média do valor
  En = - En / N
  #retornamos o valor do custo calculado
  return En


#versão primal
#def update(x,y,eta,ew):
#    r=predictor(x,ew)
#    s=(y-r)
#    #r=2*(r-0.5);
#    #s=s*eta/(1+3.7*r*r)
#    #new_eta=eta
#    ew[0]=ew[0]+s
#    ew[1:]=ew[1:]+s*x
#    return ew

#versão dual
#dado um elemento de X e seu correspondente valor em Y, 
#um learning rate eta, os valores  al e o tamanho N
#faz update dos valores em al com base em previsões feitas
#para cada linha da base de dados
def update(x,X,y,eta,al,N):
  #prevermos o valor dado pelo modelo
  pred = predictor(x,X,al,N)
  #primeiro: obter y^_N - y_N
  diff = y-pred
  #segundo: calcular o sumatório dos valores dos elementos de X
  sum = np.sum(X,axis=0)
  #terceiro: fazer produto dot de x tilde por sum 
  x_tilde = np.ones([len(x) + 1])
  x_tilde[0] = 1
  x_tilde[1:] = x 
  dot_x_sum = np.dot(x_tilde,sum)
  #quarto: atualizar al
  al = al + (diff*dot_x_sum)
  #returnamos os novos valores
  return al



#versão primal
#def run_stocastic(X,Y,N,eta,MAX_ITER,ew,err):
#    epsi=0
#    it=0
#    while(err[-1]>epsi):
#        n=int(np.random.rand()*N)
#        #new_eta=eta*math.exp(-it/850) 
#        new_eta=eta
#        ew=update(X[n],Y[n],new_eta,ew)  
#        err.append(cost(X,Y,N,ew))
#        print('iter %d, cost=%f, eta=%e     \r' %(it,err[-1],new_eta),end='')
#        it=it+1    
#        if(it>MAX_ITER): break
#    return ew, err

#versão dual
#corre o algoritmo estocástico por MAX_ITER de iterações
def run_stocastic(X,Y,N,eta,MAX_ITER,al,err):
  #erro minimo que estamos a tentar chegar no programa
  epsi=0
  #número de iterações atual
  it=0
  #enquanto o erro é maior do que o que queremos
  #e ainda não chegamos ao número máximo de iterações
  while((err[-1]>epsi) and (it< MAX_ITER)):
    #obtemos um valor aleatório da base de dados
    n=int(np.random.rand()*N)
    #nota: podemos ajustar o learning rate com base nas iterações
    #(e de facto é aconcelhavel), mas por agora ficará comentado
    #new_eta=eta*math.exp(-it/850) 
    new_eta = eta
    #atualizamos o valor dos alphas com base no elemento escolhido
    al = update(X[n],X,Y[n],new_eta,al,N)  
    #adicionamos o custo atual ao array de custos que estamos a acumular
    err.append(cost(X,Y,N,al))
    #debug
    print('iter %d, cost=%f, eta=%e     \r' %(it,err[-1],new_eta),end='')
    #aumentamos as iterações feitas
    it = it + 1    
  #no final returnamos os valores que calculamos
  return al, err






#=========== MAIN CODE ===============

#nota: as coisas abaixo não foram alteradas da aula





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

Nt=int(N*1)
I=Nt
Xt=data[:Nt,:-1];Yt=data[:Nt,-1]
al=np.ones([I+1])
err=[];err.append(cost(Xt,Yt,Nt,al))

al,err=run_stocastic(Xt,Yt,Nt,0.01,200,al,err);print("\n")
al,err=run_stocastic(Xt,Yt,Nt,0.1,1999,al,err);print("\n")
al,err=run_stocastic(Xt,Yt,Nt,0.03,1999,al,err);print("\n")
plot_error(err)


print('in-samples error=%f ' % (cost(Xt,Yt,Nt,al)))
C =confusion(Xt,Yt,Nt,al)
print(C)
#print('True positive=%i, True Negative=%i, False positive=%i, False negative=%i, ' % (TP,TN,FP,FN))

Ne=N-Nt;Xe=data[Nt:N,:-1];Ye=data[Nt:N,-1]
print('out-samples error=%f' % (cost(Xe,Ye,Ne,al)))
C =confusion(Xe,Ye,Ne,al)
print(C)
#TP,TN,FP,FN =confusion(Xe,Ye,Ne,al)
#print('True positive=%i, True Negative=%i, False positive=%i, False negative=%i, ' % (TP,TN,FP,FN))
#plot_tagged_data(10,6,n_row,n_col,Xe,Ye,al)

print('bye')
