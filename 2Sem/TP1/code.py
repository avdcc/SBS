#imports
import gym

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

'''
Pré-processamento(declaração de variáveis necessárias ao modelo)
'''

#criar ambiente gym do jogo
env = gym.make('Breakout-v0')
env.reset()



'''
Criação do modelo usando keras

Compilação do modelo
'''

#TODO: rever esta função mais tarde e adicionar o tipo de camada correto para os dados(quando tivermos definidos)

#cria um modelo linear 
#cujo tamanho dos dados de input é dado por input_size
#e cujo tamanho dos dados de output é dado por output_size
#, sendo que o nº de camadas de input é dado por input_layers
#, o nº de camadas de output é dado por output_layers
#, e a camada intermédia tem  input_layers/2 camadas(TODO:rever mais tarde)
#, retornando o modelo compilado usando otimizador Adam
#com passo step_size, usando sparse_categorical_crossentropy
#e tendo loss accuracy
def build_model(input_size,output_size,input_layers,output_layers,step_size):
  #iniciar modelo sequencial
  model = keras.Sequential()

  #adicionar camada de input  
  #nota: o abaixo assume que vamos usar um dense layer
  #é possivel usar outra, e nesse caso temos de rever isto
  #também assume relu como ativação, rever depois
  input_layer = keras.layers.Dense(input_layers,input_dim=input_size,activation='relu')
  model.add(input_layer)

  #adicionar camada intermédia
  #nota: o abaixo assume que vamos usar um dense layer
  #, mas em principio será assim para a hidden, rever depois
  #também assume relu como ativação, rever depois
  hidden_layer = keras.layers.Dense(input_layers/2,activation='relu')
  model.add(hidden_layer)

  #adicionar camada de output
  #nota: o abaixo assume que vamos usar um dense layer
  #, mas em principio será assim para a output, rever depois
  #também assume softmax como ativação, rever depois
  output_layer = keras.layers.Dense(output_layers,activation='softmax')
  model.add()

  #compilar modelo
  model.compile(optimizer=tf.train.AdamOptimizer(step_size), 
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])

  #retornar modelo
  return model



'''
Fit do modelo(colocar os dados no formato correto)
'''



'''
Avaliação do modelo(calculo de erro)
'''



'''
Fazer previsões(por o modelo a jogar no ambiente)
'''


#temporariamente fazemos que ele simplesmente faça movimentos aleatórios
#TODO: mudar isto para não ser aleatório


#tem 20 episodios
for i_episode in range(20):
  done = False
  steps = 0
  #o reset retorna uma observação inicial do ambiente
  observation = env.reset()
  #para cada episodio vamos fazer 100 steps 
  #ou quando done retornar verdade,
  #a que ocorrer primeiro
  while not done:
    #fazemos render do ambiente
    env.render()
    #criamos accção(aleatória neste momento)
    action = env.action_space.sample()
    #fazemos um passo com a acção que criamos
    #e vemos os dados que  ele retorna
    observation, reward, done, info = env.step(action)
    #incrementar steps
    steps = steps + 1 
    #retorno imediato se o passo retornar done como sendo verdadeiro
    if done:
      #debug info
      print("last action: {}".format(action))
      #print("final observation: {}".format(observation))
      print("final reward: {}".format(reward))
      print("done: {}".format(done))
      print("debug info: {}".format(info))
      print("Episode finished after {} timesteps".format(steps+1))

'''
DEBUG
'''

#espaço de acções
#é um array que
#efetivamente corresponde a acções 
#para mover no jogo
print(env.action_space)

#espaço de observação
#é um array de dimensão (210, 160, 3)
#que corresponde ao nº de pixeis do jogo
print(env.observation_space)
