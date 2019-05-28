
#imports
import numpy as np

import pandas as pd

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
from keras import models,layers,metrics,losses
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
import keras.backend as K



#funções auxiliares


#dado um nome de ficheiro csv usa o pandas para ler os dados nele contidos
#para certo separador
#nota: assume que o ficheiro dado está na mesma pasta que o programa atual 
#e que é um csv
def read_from_csv_file(filename,seperator=None):
  #inicializar variável que terá os dados do csv
  csv_file_data = None
  if(not seperator):
  #ler csv usando pandas sem separador definido
    csv_file_data = pd.read_csv(filename)
  #ler csv usando pandas com separador
  else:
    csv_file_data = pd.read_csv(filename,sep=seperator)
  #retornar dados lidos
  return csv_file_data



#separa uma matriz de dados com base numa percentagem
#, retornando 2 matrizes: uma com percentagem % de linhas da matriz
#e a outra com (100-percentagem)% de linhas da matriz
#nota: as linhas são para a mesma percentagem e mesma matriz sempre as mesmas
def split_matrix(matrix,percentile):
  #tamanho da matriz
  matriz_len = len(matrix)
  #determinar quantas linhas temos na primeira matriz
  num_lines = int(percentile * matriz_len)
  #dividir usando np.split
  splits = np.vsplit(matrix, [num_lines])
  first_div = splits[0]
  secound_div = splits[1]
  #returnar ambas as matrizes
  return first_div,secound_div



#trata dos dados, preparando-os para serem usados no programa
#requer como parametros os dados do csv do qual lemos dados para usar no nosso modelo
#e a percentagem de dados que usaremos como dados de treino
def prepare_data(csv_data,split_percentile):

  #nesta função devemos tratar dos dados que temos
  #de modo a que possa ser usado durante o treino do modelo

  #temos portanto que retornar os dados de treino separados nas colunas
  #que estamos a usar para criar a previsão(x_train)
  #e a coluna que queremos prever(y_train)
  #e retornar o equivalente para os dados de teste
  #(x_test e y_test, respetivamente)

  #no nosso caso, queremos prever a coluna "speed_diff" dos dados que nos foram passados
  #assim esta função fará as seguintes coisas:


  #-1 - remover a coluna dateComplete, visto que é-nos inutel 
  #pois temos creation_date e creation_time
  csv_data = csv_data.drop('dateComplete', axis=1)

  #0 - transformar dados não numéricos em classes numéricas

  #para cada coluna
  for col in csv_data:
    #processamos as que têm valores não inteiros
    if(not isinstance(csv_data[col][0],np.int64)):
      #codificamos usando LabelEncoder, que codifica em inteiros
      #únicos para cada classe diferente
      encoder = LabelEncoder()
      transformed = encoder.fit_transform(csv_data[col])
      csv_data[col] = transformed
  
  

  #1- separar os dados em 2:treino e teste
  #,sendo a divisão dada pela split_percentile dos argumentos
  train_data,test_data = split_matrix(csv_data,split_percentile)

  
  #2- obter das matrizes a coluna speed_diff e colocar em y_train e y_test os seus valores
  y_train = train_data['speed_diff']
  y_test = test_data['speed_diff']


  #3 - retirar das matrizes a coluna que obtivemos atrás e colocar o resultado em x_train e x_test
  x_train = np.array(train_data.drop('speed_diff', axis=1))
  x_test = np.array(test_data.drop('speed_diff', axis=1))

  #3.5- alterar a forma para estar de acordo com LSMT
  #(que quer um array (nº elementos total,nº elementos a testar de cada vez, features))
  #como nós queremos colocar os dadis elemento a elemento, temos de colocar 1
  x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
  x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))


  #4- returnar os valores que obtivemos sobre forma de pares
  return (x_train,y_train),(x_test,y_test)


#cria o modelo com base em vários parametros e retorna-o compilado com Adam
#usa como parametros o número de neurónios de entrada,
#a forma do input usado e o valor do learning rate do modelo
#inspiração: https://www.kaggle.com/umerfarooq807/prediction-model-in-keras
def build_model(input_neurons,input_shape,learning_rate):
  #dimensão de retorno
  #como estamos a prever 1 valor numa coluna deve ter valor 1
  out_dim = 1
  #número de neurónios a usar na primeira camada escondida
  hidden_neurons = 500
  #número a usar na segunda
  hidden_neurons_2 = int(hidden_neurons/2)

  #nesta função devemos criar um modelo usando keras
  #e retornar dito modelo após estar compilado usando Adam

  #para fazermos tal começaremos por inicializar o modelo
  #o modelo que iremos usar será sequencial
  model = Sequential()

  #começamos por adicionar a camada de input do modelo
  #esta será uma camada LSTM
  input_layer = LSTM(hidden_neurons,input_shape = (1,input_shape), return_sequences=True)
  model.add(input_layer)

  #em seguida adicionamos uma camada de dropout
  drop_layer_1 = Dropout(0.2)
  model.add(drop_layer_1)


  #a seguir adicionamos 2 camadas escondidas ao modelo
  #1ª escondida
  hidden_layer_1 = LSTM(hidden_neurons_2, return_sequences = True)
  model.add(hidden_layer_1)

  #dropout 2
  drop_layer_2 = Dropout(0.2)
  model.add(drop_layer_2)

  #2ª escondida
  hidden_layer_2 = LSTM(hidden_neurons_2, return_sequences = False)
  model.add(hidden_layer_2)

  #dropout 3
  drop_layer_3 = Dropout(0.2)
  model.add(drop_layer_3)

  #finalmente definimos a camada de saida
  output_layer = Dense(out_dim,input_dim = hidden_neurons_2,activation = 'relu')
  model.add(output_layer)


  #após termos definido o modelo, devemos compilar usando o Adam e o
  #learning rate definido nos argumentos da função
  loss_func = 'mse'
  optimizer_used = tf.train.AdamOptimizer(learning_rate)
  metrics_used = ['accuracy']
  model.compile(loss=loss_func, optimizer=optimizer_used,metrics=metrics_used)

  #finalmente, retornamos o modelo
  return model


#treina o modelo que lhe é dado com base nos dados que temos
#efetivamente corresponde a um fit do modelo com dados apropriados
def train_model(model,data,batch_size,epochs):
  #nesta função queremos treinar o modelo usando os dados que nos foram dados
  
  #para isso começamos por desenpacotar os dados
  (x_train,y_train),(x_test,y_test) = data
  

  #depois disto podemos já fazer o fit do modelo  
  #usando os dados e os argumentos da função, 
  #fazendo ainda shuffle dos dados ao longo do treino
  #nota: estamos a eliminar todas as mensagens que ele coloca
  #normalmente colocando a verbosidade a 0
  #adicioalmente a validation_data serve apenas para calcular as métricas
  #de erro do modelo em cada epoca, e não influencia os dados de treino
  model.fit(x_train, y_train, batch_size=batch_size, \
            epochs=epochs, validation_data=(x_test, y_test), \
            shuffle=True, verbose=1)



#avalia o modelo com base em dados de teste, tendo em conta o tamanho de batch
def evaluate_model(model,test_data,batch_size):
  #nesta função queremos avaliar o modelo dado
  
  #para isso começamos por separar os dados de teste
  x_test,y_test = test_data

  #depois avaliamos o modelo
  finalEval = model.evaluate(x_test,y_test,batch_size=batch_size)
  print(finalEval)








#programa principal
def main():
  #1º passo: preparar dados

  #nome do csv com os dados
  input_csv = "./Guimaraes/tfwi.csv"
  #percentagem de dados que serão para treino
  training_percentile = 0.8
  #ler data do csv
  data = read_from_csv_file(input_csv)
  #preparar dados
  #dataset = (x_train,y_train),(x_test,y_test)
  dataset = prepare_data(data,training_percentile)

  #2º passo: inicializar modelo

  #nº de neurónios de entrada do modelo
  input_neurons = 64
  #forma dos dados de entrada
  input_dim = dataset[0][0].shape[2]
  #learning rate do modelo
  learning_rate = 0.001
  #construir modelo
  model = build_model(input_neurons,input_dim,learning_rate)


  #3ª passo: treinar modelo com os dados

  #batch size
  batch_size = 96
  #nº de épocas que o modelo deve ser treinado
  epochs = 32
  #treinar modelo
  train_model(model,dataset,batch_size,epochs)


  #4º passo: avaliar modelo
  evaluate_model(model,dataset[1],batch_size)

  #terminado
  print("Training terminated")





#correr programa
main()
