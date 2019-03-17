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



'''
Criação do modelo usando keras
'''


'''
Compilação do modelo
'''


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
