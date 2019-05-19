#!python
'''
stocastic_logistic_classifier with dual method
'''
import csv
import numpy as np
import matplotlib.pyplot as plt
import math

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
    
def plot_tagged_data(row,col,n_row,n_col,X,X_tilde,Y,al,N): 
    fig=plt.figure(figsize=(row,col))
    for n in range(row*col):
        img=np.reshape(X[n],(n_row,n_col))
        fig.add_subplot(row, col, n+1)
        #if(Y[n]>0):#exact case
        if(predictor(n,X_tilde,al)>0.5):
            plt.imshow(img,interpolation='none',cmap='RdPu')
        else:
            plt.imshow(img,interpolation='none',cmap='cool')               
    plt.show()
    
def plot_error(err):
    plt.plot(range(len(err)), err, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Number of misclassifications')
    plt.ylim([0,20])
    plt.show()
    return 



#============== Confusion matrix ==================

def confusion(Xeval,Yeval,N,al):
    C=np.zeros([2,2])
    Xeval_tilde = np.array( [ np.insert(Xeval[i], 0, 1, axis=0) for i in range(len(Xeval))] )
    for n in range(N):
        y=predictor(n,Xeval_tilde,al)
        if(y<0.5 and Yeval[n]<0.5): C[0,0]=C[0,0]+1
        if(y>0.5 and Yeval[n]>0.5): C[1,1]=C[1,1]+1
        if(y<0.5 and Yeval[n]>0.5): C[1,0]=C[1,0]+1
        if(y>0.5 and Yeval[n]<0.5): C[0,1]=C[0,1]+1
    return C

#calculos de precisão do modelo
def recall(C):
    return (C[0,0]) / (C[0,0] + C[1,0])

def accuracy(C):
    return (C[0,0] + C[1,1])/(C[0,0] + C[0,1] + C[1,0] + C[1,1])

def precision(C):
    return (C[0,0]) / (C[0,0] + C[0,1])



#============== Logistic classifier Stuff ==================


#retorna o valor de aplicar a função sigmoid ao argumento que lhe é passado
def sigmoid(s):
  # garantir que se o valor de s for demasiado afastado do 0 é maximizado por certo valor(30 neste caso) 
  large=30
  if s<-large: s=-large
  if s>large: s=large
  # returnar valor do sigmoid de s
  return (1 / (1 + np.exp(-s)))



#versão dual
# dado X,um x seu elemento e um array al de pesos aplicados a cada valor xi 
# retorna a previsão feita para dito valor 
#corresponde a sigmoid da transposta de sum_i(al_i*x_tilde_i) com x_tilde, tendo em conta os tildes
def predictor(n,X_tilde,al):
  #calculamos o valor de sum_n(alpha_n * X_tilde_n)
  arr_res = np.array( [X_tilde[i] * al[i] for i in range(len(X_tilde))] )
  sum_xi_ali = np.sum(arr_res,axis=0)
  #finalmente fazemos o produto dot entre x_tilde e o sumatório que criamos
  x_tilde = X_tilde[n]
  s = np.dot(sum_xi_ali,x_tilde)
  #calcular a previsão para o nosso valor
  #, arredondado para 0 ou 1
  sigma = int(sigmoid(s))
  #e returnamos a previsão feita
  return sigma




#versão dual
#dado a matriz X de features ,o array Y de valores que queremos obter
#, o número de linhas N da matriz X e a lista de valores alpha al
#calcula o custo associado(i.e., a perda que o modelo atual tem)
def cost(X_tilde,Y,N,al):
  #variaveis auxiliares
  #valor da perda, inicializado a 0
  En = 0
  #garante que os valores previstos nunca serão considerados demasiado perto de 0 ou 1
  #(a razão para isto tem a ver com o facto de querermos suavizar a curva de previsões)
  epsi=1.e-12
  #para cada linha de x
  for n in range(N):
    #prevemos o valor de y associado
    y_n = predictor(n,X_tilde,al)
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




#versão dual
#dado um elemento de X e seu correspondente valor em Y, 
#um learning rate eta, os valores  al e o tamanho N
#faz update dos valores em al com base em previsões feitas
#para cada linha da base de dados
def update(n,X_tilde,y,eta,al,N):
  #prevermos o valor dado pelo modelo
  pred = predictor(n,X_tilde,al)
  #obter y^_N - y_N
  diff = y - pred
  #calculos para melhorar previsão
  #pred = 2*(pred - 0.5)
  #diff = diff * eta/(1+3.7*pred*pred)
  #calcular x tilde
  x_tilde = X_tilde[n]
  #para cada linha x_n em X calculamos x_n_tilde tranposto dot x_tilde
  #e colocamos num array (porque o produto dot entre x_n_tilde e x_tilde dá um valor)
  X_calc = np.matmul(X_tilde,x_tilde)
  #quarto: atualizar al
  al += eta * diff * X_calc
  #returnamos os novos valores
  return al




#versão dual
#corre o algoritmo estocástico por MAX_ITER de iterações
def run_stocastic(X_tilde,Y,N,eta,MAX_ITER,al,err):
  #erro minimo que estamos a tentar chegar no programa
  epsi=0
  #número de iterações atual
  it=0
  #enquanto o erro é maior do que o que queremos
  #e ainda não chegamos ao número máximo de iterações
  while((err[-1]>epsi) and (it< MAX_ITER)):
    #obtemos um valor aleatório da base de dados
    n=int(np.random.rand()*N)
    #update do eta
    new_eta = eta * math.exp(-it/850)
    #atualizamos o valor dos alphas com base no elemento escolhido
    al = update(n,X_tilde,Y[n],new_eta,al,N)  
    #adicionamos o custo atual ao array de custos que estamos a acumular
    err.append(cost(X_tilde,Y,N,al))
    #debug
    print('iter %d, cost=%f, eta=%e     \r' %(it,err[-1],new_eta),end='')
    #aumentamos as iterações feitas
    it = it + 1    
  #no final returnamos os valores que calculamos
  return al, err







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

#calcular tamanhos
training_percentage = 0.8
Nt=int(N*training_percentage)
#inicializar X e Y
Xt=data[:Nt,:-1];Yt=data[:Nt,-1]
#inicializar array de pesos
al=np.ones([Nt])
#calcular X tilde
Xt_tilde = np.array( [ np.insert(Xt[i], 0, 1, axis=0) for i in range(Nt)] )
#inicializar array de erros
err=[];err.append(cost(Xt_tilde,Yt,Nt,al))

#correr modelo
al,err=run_stocastic(Xt_tilde,Yt,Nt,1,200,al,err);print("\n")
al,err=run_stocastic(Xt_tilde,Yt,Nt,0.1,500,al,err);print("\n")
al,err=run_stocastic(Xt_tilde,Yt,Nt,0.03,1000,al,err);print("\n")
#mostrar gráfico de erro
plot_error(err)


#print('in-samples error=%f ' % (cost(Xt,Yt,Nt,al)))
C =confusion(Xt,Yt,Nt,al)
print(C)
print("in-samples confusion matrix evaluations (recall,accuracy,precision) = (",recall(C),",",accuracy(C),",",precision(C),")")
#print('True positive=%i, True Negative=%i, False positive=%i, False negative=%i, ' % (TP,TN,FP,FN))

Ne=N-Nt;Xe=data[Nt:N,:-1];Ye=data[Nt:N,-1]
#print('out-samples error=%f' % (cost(Xe,Ye,Ne,al)))
C =confusion(Xe,Ye,Ne,al)
print(C)
print("out-samples confusion matrix evaluations (recall,accuracy,precision) = (",recall(C),",",accuracy(C),",",precision(C),")")
#TP,TN,FP,FN =confusion(Xe,Ye,Ne,al)
#print('True positive=%i, True Negative=%i, False positive=%i, False negative=%i, ' % (TP,TN,FP,FN))
#plot_tagged_data(10,6,n_row,n_col,Xe,Ye,al)

print('bye')
