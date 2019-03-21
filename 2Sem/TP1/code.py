
import gym

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import os
import random
import imageio
from skimage.transform import resize

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import cv2

# ---------------------------------------------------------------------------------
# GLOBAIS

#ambiente
ENV_NAME = "LunarLander-v2" # "Breakout-v0"

#
GAMMA = 0.95
#taxa de aprendizagem
LEARNING_RATE = 0.001

#tamanho da memória
MEMORY_SIZE = 1000000
#tamanho batch 
BATCH_SIZE = 20

#
SAVE_EVERY = 10

#nº de episódios a correr
NUMBER_OF_EPISODES = 2000
#maximo de movimentos que podem ser fetios por episódio
MAX_TIMESTEPS = 1000
 
#
EXPLORATION_MAX = 1.0
#
EXPLORATION_MIN = 0.01
#
EXPLORATION_DECAY = 0.995




# FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string('train_dir', 'tf_train_breakout',"""Directory where to write event logs and checkpoint. """)
# tf.app.flags.DEFINE_string('restore_file_path','./coiso.h5',"""Path of the restore file """)
# tf.app.flags.DEFINE_integer('num_episode', 100000,"""number of epochs of the optimization loop.""")
# # tf.app.flags.DEFINE_integer('observe_step_num', 5000,
# tf.app.flags.DEFINE_integer('observe_step_num', 50000,"""Timesteps to observe before training.""")
# # tf.app.flags.DEFINE_integer('epsilon_step_num', 50000,
# tf.app.flags.DEFINE_integer('epsilon_step_num', 1000000,"""frames over which to anneal epsilon.""")
# tf.app.flags.DEFINE_integer('refresh_target_model_num', 10000,  # update the target Q model every refresh_target_model_num
#               """frames over which to anneal epsilon.""")
# tf.app.flags.DEFINE_integer('replay_memory', 400000,  # takes up to 20 GB to store this amount of history data
#               """number of previous transitions to remember.""")
# tf.app.flags.DEFINE_integer('no_op_steps', 30,"""Number of the steps that runs before script begin.""")
# tf.app.flags.DEFINE_float('regularizer_scale', 0.01,"""L1 regularizer scale.""")
# tf.app.flags.DEFINE_integer('batch_size', 32,"""Size of minibatch to train.""")
# tf.app.flags.DEFINE_float('learning_rate', 0.00025,"""Number of batches to run.""")
# tf.app.flags.DEFINE_float('init_epsilon', 1.0,"""starting value of epsilon.""")
# tf.app.flags.DEFINE_float('final_epsilon', 0.1,"""final value of epsilon.""")
# tf.app.flags.DEFINE_float('gamma', 0.99,"""decay rate of past observations.""")
# tf.app.flags.DEFINE_boolean('resume', False,"""Whether to resume from previous checkpoint.""")
# tf.app.flags.DEFINE_boolean('render', False,"""Whether to display the game.""")

# ATARI_SHAPE = (84, 84, 4)  # input image size to model
# ACTION_SIZE = 3




# ---------------------------------------------------------------------------------
# MODELO

#classe do modelo 
class DQNSolver:
  #inicialização do modelo
  def __init__(self, observation_space, action_space):
    self.exploration_rate = EXPLORATION_MAX

    self.action_space = action_space
    self.memory = deque(maxlen=MEMORY_SIZE)

    self.model = Sequential()
    self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
    self.model.add(Dense(24, activation="relu"))
    self.model.add(Dense(self.action_space, activation="linear"))
    self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

  #
  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  #
  def act(self, state):
    if np.random.rand() < self.exploration_rate:
      return random.randrange(self.action_space)
    q_values = self.model.predict(state)
    return np.argmax(q_values[0])

  #
  def experience_replay(self):
    if len(self.memory) < BATCH_SIZE:
      return
    batch = random.sample(self.memory, BATCH_SIZE)
    for state, action, reward, state_next, terminal in batch:
      q_update = reward
      if not terminal:
        q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
      q_values = self.model.predict(state)
      q_values[0][action] = q_update
      self.model.fit(state, q_values, verbose=0)
    self.exploration_rate *= EXPLORATION_DECAY
    self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)




# ---------------------------------------------------------------------------------
# AUXILIARES

# -----
# Processa a imagem removendo informacao desnecessaria (no breackout)
def processImage(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[17:101,4:80]
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(84,76,1))


# action0 = 0  # do nothing
# observation0, reward0, terminal, info = env.step(action0)
# print("Before processing: " + str(np.array(observation0).shape))
# # plt.imshow(np.array(observation0))
# # plt.show()
# observation0 = preprocess(observation0)
# print("After processing: " + str(np.array(observation0).shape))
# plt.imshow(np.array(np.squeeze(observation0)))
# plt.show()

# brain.setInitState(observation0)
# brain.currentState = np.squeeze(brain.currentState)

# -----
# Guardar o modelo model num certo ficheiro de nome filename na subpasta models
def save_model(model,filename):
  #ficheiro onde iremos guardar: models/filename
  file_saved_to = "models/" + filename
  #guardamos o modelo
  model.save(file_saved_to)

# Carrega o modelo do ficheiro filename da pasta models
def load_model(filename):
  model = None
  #testar se o ficheiro existe
  try:
    model = keras.models.load(filename)
  #se não existir
  except ValueError:
    print("CRITICAL ERROR: model not found in models/",filename,". Please check if it exists")
  return model







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
#e tendo métrica accuracy
def build_model(input_size,output_size,input_layers,output_layers,step_size):
  #iniciar modelo sequencial
  model = keras.Sequential()

  #adicionar camada de input nota: o abaixo assume que vamos usar um dense layer é possivel usar outra, e nesse caso temos de rever isto também assume relu como ativação, rever depois
  input_layer = keras.layers.Dense(input_layers,input_dim=input_size,activation='relu')
  model.add(input_layer)

  #adicionar camada intermédia nota: o abaixo assume que vamos usar um dense layer, mas em principio será assim para a hidden, rever depois também assume relu como ativação, rever depois
  hidden_layer = keras.layers.Dense(input_layers/2,activation='relu')
  model.add(hidden_layer)

  #adicionar camada de output nota: o abaixo assume que vamos usar um dense layer, mas em principio será assim para a output, rever depois também assume softmax como ativação, rever depois
  output_layer = keras.layers.Dense(output_layers,input_dim=output_size,activation='softmax')
  model.add()

  #compilar modelo
  model.compile(optimizer=tf.train.AdamOptimizer(step_size),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])

  #retornar modelo
  return model



'''
Fit do modelo(colocar os dados no formato correto)

Treinar modelo
'''

#explicar porque o modelo faz fit e treino ao mesmo tempo
#deve-se ao facto que a API keras usa o model.fit
#para treinar o dataset
#como tal, o fit de dados e o treino ocorrem simultaneamente


#TODO: completar tarefas abaixo
#coisas a fazer neste sitio:
#definir dados de input/output
#criar modelo com base neles(chamando build_model)
#fazer fit do modelo


#fit do modelo

def fit(model, # Modelo
        gamma, # Penalizacao
        start_states, # Array com os estados inicais
        actions, # Acoes
        rewards, # Reconpensas(Estado Inical + ACOES)
        next_states, # Estado depois (Estado Inicial + ACOES)
        is_terminal): # flag se ganhou ou nao

    #
    next_values = model.predict([ next_states, np.opnes(actions.shape) ])

    #
    next_values[ is_terminal ] = 0

    #
    values = rewards + gama * np.max(next_values,axis=1)


    #fazer fit do modelo
    state = [start_states, actions]
    reward_value = actions * values[:, None]

    #
    model.fit(state,reward_value, epochs=1,batch_size=len(start_states),verbose=0)





# funcao auxiliar
def calc_pos(d,i):
    return np.array(map(lambda x: x[i],d)).reshape(-1, len(d[0][i]))


#treinar modelo
def train_model(trainig_data):
    X,Y = [calc_pos(d,i) for i in range(2)]

    #criar o modelo
    model = build_model(
            len(x[0]),# input_size
            len(y[0]),# output_size
            input_layers,
            output_layers,
            step_size)

    #fit(model,
    #    gamma,
    #    start_states,
    #    actions,
    #    rewards,
    #    next_states,
    #    is_terminal)

    model.fit(X, Y, epochs = 10)
    return model

'''
Avaliação do modelo(calculo de erro)
'''

#nesta parte provavelmente apenas teremos
#de colocar uma mensagem para ver qual a accuracy do modelo













def runner():
  env = gym.make(ENV_NAME)
  env.seed(0)

  scores = []
  scores_window = deque(maxlen=100)  # last 100 scores


  # print('State shape: ', env.observation_space.shape)
  # print('Number of actions: ', env.action_space.n)


  observation_space = env.observation_space.shape[0]
  action_space = env.action_space.n
  dqn_solver = DQNSolver(observation_space, action_space)

  for episode in range(NUMBER_OF_EPISODES):

    state = env.reset()
    state = np.reshape(state, [1, observation_space])
    
    score = 0

    for step in range(MAX_TIMESTEPS):
      # env.render()
      action = dqn_solver.act(state)
      state_next, reward, terminal, info = env.step(action)

      score += reward

      state_next = np.reshape(state_next, [1, observation_space])
      dqn_solver.remember(state, action, reward, state_next, terminal)
      state = state_next
      if terminal:
        print("Episode: " + str(episode) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(score))
        
        # score_logger.add_score(step, run)
        break
      dqn_solver.experience_replay()


if __name__ == "__main__":
  runner()









































'''

# ---------------------------------------------------------------------------------
# Pre-processamento(declaracao de variaveis necessarias ao modelo)

#criar ambiente gym do jogo
env = gym.make(ENV_NAME)
env.reset()

#número de camadas do modelo que criaremos
#TODO: determinar números que melhorem o resultado
input_layers = 64
output_layers = 10

#tamanho de passo usado para o otimizador na compilação
#TODO: determinar o valor otimo para esta variável
step_size = 0.001

#número de epocas que devem ser usadas
ephocs = 10

#nome do ficheiro onde será guardado o modelo
#e depois carregado
save_filename = "test.h5"




#temporariamente fazemos que ele simplesmente faça movimentos aleatórios
#TODO: mudar isto para não ser aleatório


#tem 20 episodios
for i_episode in range(NUMBER_OF_EPISODES):
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


#espaço de acções
#é um array que
#efetivamente corresponde a acções
#para mover no jogo
print(env.action_space)

#espaço de observação
#é um array de dimensão (210, 160, 3)
#que corresponde ao nº de pixeis do jogo
print(env.observation_space)

'''
