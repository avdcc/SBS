#!python
'''
stocastic_logistic_classifier with dual method
'''
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from pprint import pprint

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
            data[n][i]=float(tmp_arr[i])
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
    plt.ylim([0,5])
    plt.show()
    return 



#============== Kernel stuff ==================

#calcula kernel linear n para todos os elementos de X_tilde
def calc_linear_kernel(X_tilde,n):
  return np.array( [ np.power( np.matmul(X_tilde,X_tilde[i]),n ) for i in range(len(X_tilde)) ] ) 

#calcular kernel exponencial para todos os elementos de X_tilde
def calc_exponential_kernel(X_tilde):
  return np.array( [ np.exp( np.matmul(X_tilde,X_tilde[i]) )  for i in range(X_tilde) ] )




#============== Confusion matrix ==================

def confusion(X_calc_mat,Yeval,N,al):
    C=np.zeros([2,2])
    R = []
    for n in range(N):
      y=predictor(n,X_calc_mat,al)
      R.append((Yeval[n],y))
      if(y<0.5 and Yeval[n]<0.5): C[0,0]=C[0,0]+1
      if(y>0.5 and Yeval[n]>0.5): C[1,1]=C[1,1]+1
      if(y<0.5 and Yeval[n]>0.5): C[1,0]=C[1,0]+1
      if(y>0.5 and Yeval[n]<0.5): C[0,1]=C[0,1]+1
    return C,R

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
def predictor(n,X_calc_mat,al):
  #calculamos o valor de sum_n(alpha_n * X_tilde_n).x_tilde
  #para tal começamos por obter o elemento n de X_calc_mat
  X_calc = X_calc_mat[n]
  #, depois multiplica-se cada linha desse elemento pelos valores em al
  X_calc_mult = np.array( [ X_calc[i] * al[i] for i in range(len(X_calc)) ] )
  #e finalmente faz-se o sumatório dos elementos
  s = np.sum(X_calc_mult,axis = 0)
  #calcular a previsão para o nosso valor
  sigma = sigmoid(s)
  #e returnamos a previsão feita
  return sigma




#versão dual
#dado a matriz X de features ,o array Y de valores que queremos obter
#, o número de linhas N da matriz X e a lista de valores alpha al
#calcula o custo associado(i.e., a perda que o modelo atual tem)
def cost(X_calc_mat,Y,N,al):
  #variaveis auxiliares
  #valor da perda, inicializado a 0
  En = 0
  #garante que os valores previstos nunca serão considerados demasiado perto de 0 ou 1
  #(a razão para isto tem a ver com o facto de querermos suavizar a curva de previsões)
  epsi=1.e-12
  #para cada linha de x
  for n in range(N):
    #prevemos o valor de y associado
    y_n = predictor(n,X_calc_mat,al)
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
def update(n,X_calc_mat,y,eta,al):
  #prevermos o valor dado pelo modelo
  pred = predictor(n,X_calc_mat,al)
  #obter y^_N - y_N
  diff = y - pred
  #calculos para melhorar previsão
  #pred = 2*(pred - 0.5)
  #diff = diff * eta/(1+3.7*pred*pred)
  
  #para cada linha x_n em X calculamos x_n_tilde tranposto dot x_tilde
  #e colocamos num array (porque o produto dot entre x_n_tilde e x_tilde dá um valor)
  #este valor corresponde a X_calc_mat[n]
  X_calc = X_calc_mat[n] #np.matmul(X_tilde,X_tilde[n]) 
  #quarto: atualizar al
  al += eta * diff * X_calc
  #returnamos os novos valores
  return al




#versão dual
#corre o algoritmo estocástico por MAX_ITER de iterações
def run_stocastic(X_calc_mat,Y,N,eta,MAX_ITER,al,err):
  #erro minimo que estamos a tentar chegar no programa
  epsi = 1e-8
  #número de iterações atual
  it = 0
  #enquanto o erro é maior do que o que queremos
  #e ainda não chegamos ao número máximo de iterações
  while((err[-1]>epsi) and (it< MAX_ITER)):
    #obtemos um valor aleatório da base de dados
    n=int(np.random.rand()*N)
    #update do eta
    new_eta = eta
    #new_eta = eta* math.exp(-it/850)

    #new_eta = eta/( 2 * (it + 1) )
    #new_eta = eta/( (int(it/850) + 2) * (it + 1)) 

    #ideia: observamos uma janela de certo tamanho de erros e calculamos a média
    #depois vemos se essa média está próxima dos últimos valores de erro calculados
    #se estiver, reduzimos o eta

    #tamanho da janela: MAX_ITER/50
    window_size = int(MAX_ITER/50)
    #se tivermos elementos suficientes em err(equivalente a dizer que já valor superior a window_size em it)
    if((len(err) > window_size) and (it%window_size == 0) ):
      #obtemos os elementos da nossa janela
      #e calculamos a média
      med_err_window = np.mean(err[-window_size])
      #vemos se a média dos últimos window_size/10 elementos está perto de med_err_window
      sub_window_size = int(window_size/10)
      med_sub_window = np.mean(err[-sub_window_size])
      #perto significa: med_sub_window está entre +/- 10% med_err_window
      if((med_sub_window >= med_err_window * 0.9) and (med_sub_window <= med_err_window * 1.1)):
        #enquanto isto se verifica, reduzimos o eta
        eta = eta / 5
      #se o valor estiver demasiado baixo para o eta temos de ajustar
      if((med_sub_window >= med_err_window * 0.99) and (med_sub_window <= med_err_window * 1.01)):
        eta = eta * 10

    #atualizamos o valor dos alphas com base no elemento escolhido
    al = update(n,X_calc_mat,Y[n],new_eta,al)  
    #adicionamos o custo atual ao array de custos que estamos a acumular
    err.append(cost(X_calc_mat,Y,N,al))
    #debug
    print('iter %d, cost=%f, eta=%e     \r' %(it,err[-1],new_eta),end='')
    #aumentamos as iterações feitas
    it = it + 1    
  #no final returnamos os valores que calculamos
  return al, err







#=========== MAIN CODE ===============



#corre um teste para um dataset
def run_test(dataset_name,training_percentage=0.8,kernel_deg=1,learning_rate=0.1,MAX_ITER=10000):
  #read data file
  datafile = './dataset/' + dataset_name + '.txt'
  N,n_row,n_col,data=read_asc_data(datafile)
  #transformar os dados

  #transformação base de dados quadratica
  #(x1,x2) -> (sqrt(2)*x1,sqrt(2)*x2,x1*x1,x2*x2,sqrt(2)*x1*x2)

  #transformação base de dados x^3
  #...

  #shuffle dos dados
  np.random.shuffle(data)
  #calcular tamanhos  
  Nt=int(N*training_percentage)
  #inicializar X e Y
  Xt=data[:Nt,:-1];Yt=data[:Nt,-1]
  #inicializar array de pesos
  al=np.ones([Nt])
  #calcular X tilde
  Xt_tilde = np.array( [ np.insert(Xt[i], 0, 1, axis=0) for i in range(Nt)] )
  #calculamos uma matriz 3D contendo todos os valores
  #que podemos calcular para usar durante o update usando o kernel
  X_calc_mat = calc_linear_kernel(Xt_tilde,kernel_deg) 
  #inicializar array de erros
  err=[];err.append(cost(Xt_tilde,Yt,Nt,al))

  #correr modelo
  print("Iniciando teste em",dataset_name,"com",N,"linhas de dimensão",n_row,"X",n_col)
  al,err=run_stocastic(X_calc_mat,Yt,Nt,learning_rate,MAX_ITER,al,err);print("\n")

  #avaliar modelo
  print("Avaliando modelo")
  statistics = {}

  #in-samples
  statistics['in-samples'] = {}
  C,R = confusion(X_calc_mat,Yt,Nt,al)
  statistics['in-samples']['C'] = C
  statistics['in-samples']['R'] = R
  statistics['in-samples']['recall'] = recall(C)
  statistics['in-samples']['accuracy'] = accuracy(C)
  statistics['in-samples']['precision'] = precision(C)

  #out-samples
  statistics['out-samples'] = {}
  if(training_percentage < 1):
    Ne=N-Nt;Xe=data[Nt:N,:-1];Ye=data[Nt:N,-1]
    C,R = confusion(Xe,Ye,Ne,al)
    statistics['out-samples']['C'] = C
    statistics['out-samples']['R'] = R
    statistics['out-samples']['recall'] = recall(C)
    statistics['out-samples']['accuracy'] = accuracy(C)
    statistics['out-samples']['precision'] = precision(C)
  
  #terminado
  return statistics



#datasets
#alguns foram omitidos de propósito
datasets = ['AND','CAND','OR',
            'AND3D','OR3D',
            'lin','3linP1',
            'sqr','cubed']


#correr um teste
statistics = run_test('CAND')
pprint(statistics, width=1)



# read the data file
#N,n_row,n_col,data=read_asc_data('./dataset/AND.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/CAND.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/OR.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/XOR.txt')


#N,n_row,n_col,data=read_asc_data('./dataset/AND3D.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/OR3D.txt')


#N,n_row,n_col,data=read_asc_data('./dataset/lin.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/3linP1.txt')


#N,n_row,n_col,data=read_asc_data('./dataset/sqr.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/sqr_big.txt')


#N,n_row,n_col,data=read_asc_data('./dataset/cubed.txt')


#N,n_row,n_col,data=read_asc_data('./dataset/rectangle60.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/rectangle600.txt')


#N,n_row,n_col,data=read_asc_data('./dataset/line600.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/line1500.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/my_digit.txt');np.place(data[:,-1], data[:,-1]!=1, [-1])
#print('find %d images of %d X %d pixels' % (N,n_row,n_col))

#plot_data(10,6,n_row,n_col,data)


#transformação base de dados quadratica
#(x1,x2) -> (sqrt(2)*x1,sqrt(2)*x2,x1*x1,x2*x2,sqrt(2)*x1*x2)



#shuffle dos dados
#np.random.shuffle(data)

#calcular tamanhos
#training_percentage = 0.8
#Nt=int(N*training_percentage)
#inicializar X e Y
#Xt=data[:Nt,:-1];Yt=data[:Nt,-1]
#inicializar array de pesos
#al=np.ones([Nt])
#calcular X tilde
#Xt_tilde = np.array( [ np.insert(Xt[i], 0, 1, axis=0) for i in range(Nt)] )
#calculamos uma matriz 3D contendo todos os valores
#que podemos calcular para usar durante o update usando o kernel
#X_calc_mat = calc_linear_kernel(Xt_tilde,1) 
#X_calc_mat = calc_linear_kernel(Xt_tilde,2) 
#X_calc_mat = calc_linear_kernel(Xt_tilde,3) 
#inicializar array de erros
#err=[];err.append(cost(Xt_tilde,Yt,Nt,al))

#correr modelo
#al,err=run_stocastic(X_calc_mat,Yt,Nt,0.1,10000,al,err);print("\n")
#print("\n",al,"\n")
#al,err=run_stocastic(X_calc_mat,Yt,Nt,0.2,500,al,err);print("\n")
#print("\n",al,"\n")
#al,err=run_stocastic(X_calc_mat,Yt,Nt,0.003,1000,al,err);print("\n")
#print("\n",al,"\n")
#mostrar gráfico de erro
#plot_error(err)


#print('in-samples error=%f ' % (cost(Xt,Yt,Nt,al)))
#C,R =confusion(X_calc_mat,Yt,Nt,al)
#print("################  in-samples statistics  ################")
#print(C)
#print(R)
#print("in-samples confusion matrix evaluations (recall,accuracy,precision) = (",recall(C),",",accuracy(C),",",precision(C),")")
#print('True positive=%i, True Negative=%i, False positive=%i, False negative=%i, ' % (TP,TN,FP,FN))

#if(training_percentage < 1):
#  Ne=N-Nt;Xe=data[Nt:N,:-1];Ye=data[Nt:N,-1]
#  #print('out-samples error=%f' % (cost(Xe,Ye,Ne,al)))
#  C,R =confusion(Xe,Ye,Ne,al)
#  print("################  out-samples statistics  ################")
#  print(C)
#  print(R)
#  print("out-samples confusion matrix evaluations (recall,accuracy,precision) = (",recall(C),",",accuracy(C),",",precision(C),")")
  #TP,TN,FP,FN =confusion(Xe,Ye,Ne,al)
  #print('True positive=%i, True Negative=%i, False positive=%i, False negative=%i, ' % (TP,TN,FP,FN))
  #plot_tagged_data(10,6,n_row,n_col,Xe,Ye,al)

print('Programa terminado')
