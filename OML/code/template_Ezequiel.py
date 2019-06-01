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
import json

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
    #print("N=%d, row=%d, col=%d" %(N,n_row,n_col))
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
    C=np.zeros([2,2])
    R = []
    for n in range(N):
        y=predictor(Xeval[n],ew)
        R.append((Yeval[n],y))
        if(y<0.5 and Yeval[n]<0.5): C[0,0]=C[0,0]+1
        if(y>0.5 and Yeval[n]>0.5): C[1,1]=C[1,1]+1
        if(y<0.5 and Yeval[n]>0.5): C[1,0]=C[1,0]+1
        if(y>0.5 and Yeval[n]<0.5): C[0,1]=C[0,1]+1
    return C,R



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
        if y_p_i < epsi : y_p_i = epsi
        if y_p_i > 1 - epsi : y_p_i = 1 - epsi
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




def run_stocastic(X,Y,N,eta,MAX_ITER,ew,err,verbose):
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
        if(verbose > 3):
            print('iter %d, cost=%f, eta=%e     \r' %(j,err[-1],new_eta),end='')
        #atualizar variáveis
        j = j+1
        err.append( cost(X,Y,N,ew) )
    #returnar os dados no final
    return ew,err



#=========== MAIN CODE ===============



#corre um teste para um dataset
def run_test(dataset_name,training_percentage=0.8,learning_rate=0.1,MAX_ITER=10000,verbose = 1):
  #read data file
  datafile = './dataset/' + dataset_name + '.txt'
  N,n_row,n_col,data=read_asc_data(datafile)
  #transformar os dados


  #shuffle dos dados
  np.random.shuffle(data)
  #calcular tamanhos  
  Nt=int(N*training_percentage)
  I=n_row*n_col
  #inicializar X e Y
  Xt=data[:Nt,:-1];Yt=data[:Nt,-1]
  #inicializar array de pesos
  ew=np.ones([I+1])
  #inicializar array de erros
  err=[];err.append(cost(Xt,Yt,Nt,ew))

  #correr modelo
  if(verbose >= 2):
    print("Iniciando teste em",dataset_name,"com",N,"linhas de dimensão",n_row,"X",n_col)
    
  ew,err=run_stocastic(Xt,Yt,Nt,learning_rate,MAX_ITER,ew,err,verbose)
  if(verbose >= 3):
    print("\n")

  #avaliar modelo
  if(verbose >= 2):
    print("Avaliando modelo")
  statistics = {}

  #in-samples
  statistics['in-samples'] = {}
  C,R = confusion(Xt,Yt,Nt,ew)
  statistics['in-samples']['C'] = C
  statistics['in-samples']['R'] = R
  statistics['in-samples']['recall'] = recall(C)
  statistics['in-samples']['accuracy'] = accuracy(C)
  statistics['in-samples']['precision'] = precision(C)
  #debug
  if(verbose >= 1):
    print("avaliação in-samples: (",recall(C),",",accuracy(C),",",precision(C),")")

  #out-samples
  statistics['out-samples'] = {}
  if(training_percentage < 1):
    Ne=N-Nt;Xe=data[Nt:N,:-1];Ye=data[Nt:N,-1]
    C,R = confusion(Xe,Ye,Ne,ew)
    statistics['out-samples']['C'] = C
    statistics['out-samples']['R'] = R
    statistics['out-samples']['recall'] = recall(C)
    statistics['out-samples']['accuracy'] = accuracy(C)
    statistics['out-samples']['precision'] = precision(C)
    #debug
    if(verbose > 1):
      print("avaliação out-samples: (",recall(C),",",accuracy(C),",",precision(C),")")
  
  #terminado
  return statistics





#corre vários testes para uma única base de dados
def run_battery_tests(dataset_name,num_test=10,training_percentage=0.8,learning_rate=0.1,MAX_ITER=10000,verbose=1):
  #
  if(verbose >= 1):
    print("Iniciando bateria de",num_test,"testes para",dataset_name)

  #
  statistics_arr = []
  #correr vários testes
  for i in range(num_test):
    #
    if(verbose >= 1):
      print("Iniciando teste",i+1)
    #correr modelo
    stats = run_test(dataset_name,training_percentage,learning_rate,MAX_ITER,verbose)
    #adicionar estatisticas ao array
    statistics_arr.append(stats)
  #objeto de retorno
  ret_val = {
    'in-samples': {
      'C': [],
      'R': [],
      'recall': [],
      'accuracy': [],
      'precision': []
    },
    'out-samples': {
      'C': [],
      'R': [],
      'recall': [],
      'accuracy': [],
      'precision': []
    }
  }
  #adicionar os vários C e R existentes e scores para os vários modelos
  for stat in statistics_arr:
    #in-samples
    for key in stat['in-samples'].keys():
      ret_val['in-samples'][key].append( stat['in-samples'][key] )

    #out-samples
    if(learning_rate < 1):
      for key in stat['out-samples'].keys():
        ret_val['out-samples'][key].append( stat['out-samples'][key] )


  #calculo de média dos valores das avaliações
  #médias para in-samples
  ret_val['in-samples']['avg_recall'] = np.mean( ret_val['in-samples']['recall'] )
  ret_val['in-samples']['avg_accuracy'] = np.mean( ret_val['in-samples']['accuracy'] )
  ret_val['in-samples']['avg_precision'] = np.mean( ret_val['in-samples']['precision'] )
  #médias para out-samples
  ret_val['out-samples']['avg_recall'] = np.mean( ret_val['out-samples']['recall'] )
  ret_val['out-samples']['avg_accuracy'] = np.mean( ret_val['out-samples']['accuracy'] )
  ret_val['out-samples']['avg_precision'] = np.mean( ret_val['out-samples']['precision'] )

  #transformar dados para poderem ser guardados em JSON
  ret_val['in-samples']['C'] = [ x.tolist() for x in ret_val['in-samples']['C'] ]
  ret_val['out-samples']['C'] = [ x.tolist() for x in ret_val['out-samples']['C'] ]

  #gravar para ficheiro
  filename = './resultados_primal/' + dataset_name + str(num_test) + ".json"
  with open(filename, 'w') as fp:
    json.dump(ret_val, fp, sort_keys=True, indent=2)

  #
  if(verbose >= 0):
    print("Bateria de testes concluida, resultados guardados em",filename)
  #retornar
  return ret_val
  






#corre baterias de testes em todos os datasets
def run_battery_tests_all_datasets(datasets,num_test=10,training_percentage=0.8,learning_rate=0.1,MAX_ITER=10000,verbose=1):
  for dataset_name in datasets:
    run_battery_tests(dataset_name,num_test,training_percentage,learning_rate,MAX_ITER,verbose)









#datasets
#alguns foram omitidos de propósito
datasets = ['CAND','OR',
            'AND3D','OR3D',
            'lin','3linP1',
            'sqr','cubed']



#número de vezes que o modelo será corrido para cálculo de média de resultados
nt = 10
#percentagem de dados de treino 
tp = 0.8
#learning rate inicial do modelo
lr = 0.1
#número máximo de iterações que podem ser feitas por modelo
max_iter = 10000
#verbosidade do processo
#se menor que 0, não avisa nada
#se a 0, apenas avisa quando terminar cada um dos datasets dados
#se a 1 avisa dados de quando inicia testes individuais para certo dataset
#se a 2 mostra tudo excepto os valores dos modelos enquanto estão a correr
#se maior que 2 mostra todas as mensagens
verb = 1
#corre uma bateria de testes em todos os datasets especificados com as variáveis acima definidas
run_battery_tests_all_datasets(datasets,nt,tp,lr,max_iter,verb)












# read the data file
#N,n_row,n_col,data=read_asc_data('./dataset/AND.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/XOR.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/rectangle60.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/rectangle600.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/line600.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/line1500.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/my_digit.txt');np.place(data[:,-1], data[:,-1]!=1, [-1])
#print('find %d images of %d X %d pixels' % (N,n_row,n_col))

#plot_data(10,6,n_row,n_col,data)

#training_percentage = 0.8
#Nt=int(N*training_percentage)
#I=n_row*n_col
#Xt=data[:Nt,:-1];Yt=data[:Nt,-1]
#ew=np.ones([I+1])
#err=[];err.append(cost(Xt,Yt,Nt,ew))

#ew,err=run_stocastic(Xt,Yt,Nt,1,200,ew,err);print("\n")
#ew,err=run_stocastic(Xt,Yt,Nt,0.1,500,ew,err);print("\n")
#ew,err=run_stocastic(Xt,Yt,Nt,0.03,1000,ew,err);print("\n")
#plot_error(err)

#print('in-samples error=%f ' % (cost(Xt,Yt,Nt,ew)))
#C =confusion(Xt,Yt,Nt,ew)
#print("Confusion matrix:")
#print(C)
#print("in-samples confusion matrix evaluations (recall,accuracy,precision) = (",recall(C),",",accuracy(C),",",precision(C),")")

#print("\n\n")

#Ne=N-Nt;Xe=data[Nt:N,:-1];Ye=data[Nt:N,-1]
#print('out-samples error=%f' % (cost(Xe,Ye,Ne,ew)))
#C =confusion(Xe,Ye,Ne,ew)
#print("Confusion matrix:")
#print(C)
#print("out-samples confusion matrix evaluations (recall,accuracy,precision) = (",recall(C),",",accuracy(C),",",precision(C),")")


print('Programa terminado')
