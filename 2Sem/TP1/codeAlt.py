
import gym, random, tempfile
import numpy as np

from gym import wrappers
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend as K

import matplotlib.pyplot as plt

# ---------------------------------------------------------

# ENV_NAME = "LunarLander-v2" # "Breakout-v0"
ENV_NAME = "CartPole-v1"

LOAD = False   # Se e para continuar no estado anterior
TRAIN = True   # Se estamos a trreinar o modelo ou não
RENDER = False # Se mostra a imagem do bot a jogar

OWN_LOSS_FUNCTION = False # Nossa propria funcao de loss ou mse
LOSS_FUNCTION = 'mse' # Funcao de loss usada

SAVE_COUNTER = 100 # Nº de episódios para que o modelo seja guardado
EPISODES = 10000 # Nº de episódios
TIMESTEPS = 1000 # Máximo de steps por episódio

SAVED_FILE_LOCATION = "./" + ENV_NAME + ".h5" # Nome do ficheiro onde será guardado o modelo

TIME = 0

# ---------------------------------------------------------
# Criar a nossa funcao de loss
#         ________________
# LOSS = v y_true - y_pred + TIME

def own_loss_function():
  return lambda y_true,y_pred:\
    K.mean((y_pred - y_true)*(y_pred - y_true) + 2*TIME,axis=-1)
    #K.mean(K.square(y_pred - y_true) + TIME ,axis=-1)

# incrementa a variavel global TIME
def inc_time():
  global TIME
  TIME += 1

# reset da varival global TIME
def reset_time():
  global TIME
  TIME = 0

# ---------------------------------------------------------

# Classe do modelo de machine learning
class DDQL:
  # Inicialização de modelo
  def __init__(self, nS, nA):
    self.nS = nS
    self.nA = nA
    #epsilon do modelo
    self.epsilon = 1
    #minimo de epsilon permitido
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.gamma = 0.999
    #learning rate do modelo
    self.learning_rate = 0.0001
    #épocas a usar durante o fit do modelo
    self.epochs = 10
    #controlo da verbosidade do fit do modelo
    self.verbose = 0
    #tamamnho de cada batch usado no modelo
    self.minibatch_size = 30
    self.memory = deque(maxlen=60000)

    # Inicialização do modelo
    self.model = self.create_model()
    #
    self.target_model = self.create_model()

  # Função de criação do modelo
  def create_model(self):
    #inicializar modelo sequencial
    model = Sequential()

    # Add 2 hidden layers with 64 nodes each
    model.add(Dense(64, input_dim=self.nS, activation='relu'))
    model.add(Dense(64, activation='relu'))
    #camada de output
    model.add(Dense(self.nA, activation='linear'))
    #se tiver-mos a usar a nossa função de loss
    if(OWN_LOSS_FUNCTION):
      model.compile(loss=own_loss_function(), optimizer=Adam(lr=self.learning_rate))
    else:
      #compilar modelo com otimizador Adam
      model.compile(loss=LOSS_FUNCTION, optimizer=Adam(lr=self.learning_rate))

    return model

  #adicionar dados à memória ao modelo
  def add_memory(self, s, a, r, s_prime, done):
    self.memory.append((s, a, r, s_prime, done))

  #fazer update do modelo target
  def target_model_update(self):
    self.target_model.set_weights(self.model.get_weights())

  #selecionar ação
  def selectAction(self, s):
    #se o valor de np.random for menor que o epsilon atual
    if np.random.rand() <= self.epsilon:
      #returnar o nA do modelo como escolha
      return np.random.choice(self.nA)
    #caso contrário, prever usando o modelo que temos
    q = self.model.predict(s)
    #e retornar o indice de valor máximo de q[0]
    return np.argmax(q[0])

  def replay(self):
    # Vectorized method for experience replay
    #obter minibatch_size elementos aleatórios dos valores em memory
    minibatch = random.sample(self.memory, self.minibatch_size)
    #criar um array deles
    minibatch = np.array(minibatch)
    #obter todas as linhas cuja quarta coluna tenha valor False(i.e., estados não terminais)
    not_done_indices = np.where(minibatch[:, 4] == False)
    #colocar em y os valores da segunda coluna de cada linha de minibatch
    y = np.copy(minibatch[:, 2])

    # If minibatch contains any non-terminal states, use separate update rule for those states
    if len(not_done_indices[0]) > 0:
      #prever usando o modelo do vertical stack das terceiras colunas de cada linha de minibatch
      predict_sprime = self.model.predict(np.vstack(minibatch[:, 3]))
      #fazer o mesmo para o target model
      predict_sprime_target = self.target_model.predict(np.vstack(minibatch[:, 3]))

      # Non-terminal update rule
      #atualizar os indices não terminais em y
      y[not_done_indices] += np.multiply(self.gamma, \
            predict_sprime_target[not_done_indices, \
            np.argmax(predict_sprime[not_done_indices, :][0], axis=1)][0])

    #obter as acções que correspondem à coluna 1 de cada linha do minibatch
    actions = np.array(minibatch[:, 1], dtype=int)
    #prever usando o modelo sobre o stack vertical da coluna 0 do minibatch
    y_target = self.model.predict(np.vstack(minibatch[:, 0]))
    #atualizar o acima criado para que nas linhas 0 até minibatch_size-1 nas colunas das acções tenham o valor de y
    y_target[range(self.minibatch_size), actions] = y

    #fazer fit do modelo com base no acima calculado
    #usa-se o stack vertical da primeira coluna como data de treino,
    #o y_target como data target, as epocas do modelo e a verbosidade deste também
    self.model.fit(np.vstack(minibatch[:, 0]), y_target, epochs=self.epochs, verbose=self.verbose)


  #similar ao replay, mas sem usar vetores
  #nota adicional: esta função não é usada em lado nenhum, mas penso ser equivalente à replay
  def replayIterative(self):
    # Iterative method - this performs the same function as replay() but is not vectorized
    #iniciar listas
    s_list = []
    y_state_list = []
    #obter minibatch_size elementos aleatórios dos valores em memory
    minibatch = random.sample(self.memory, self.minibatch_size)
    #para cada valor em minibatch
    for s, a, r, s_prime, done in minibatch:
      #adicionar s a s_list
      s_list.append(s)
      #colocar r como y_action
      y_action = r
      #se o modelo não tiver terminado
      if not done:
        #atualizar a y_action usando o r, o gamma
        #e o valor máximo da primeira coluna prevista prlo modelo para s_prime
        y_action = r + self.gamma * np.amax(self.model.predict(s_prime)[0])

      #debug
      print(y_action)

      #colocar em y_state a previsão do modelo para s
      y_state = self.model.predict(s)
      #atualizar o acima de tal modo que na primeira linha na coluna a tenhamos a acção y_action
      y_state[0][a] = y_action
      #atualizar a lista de estados, adicionando o y_state ao fim
      y_state_list.append(y_state)
    #fazer fit do modelo com os dados que criamos
    #tem s_list como data de treino, y_state_list como data target,
    #minibatch_size como tamanho de batch, uma época e verbosidade 0

    #nota: colocar do mesmo modo que o replay o epochs e verbose provelvemente é boa ideia

    self.model.fit(np.squeeze(s_list), np.squeeze(y_state_list), batch_size=self.minibatch_size, epochs=1, verbose=0)

# ---------------------------------------------------------

# Fazer log de um texto para logs.txt
def log(texto):
  with open(ENV_NAME + ".csv", "a") as myfile:
    myfile.write(str(texto))

# def logSettings():
#   with open("logs.txt", "a") as myfile:
#     myfile.write(str(texto))

#     ENV_NAME = "LunarLander-v2" # "Breakout-v0"

#     LOAD = False   # Se e para continuar no estado anterior
#     TRAIN = True   # Se estamos a trreinar o modelo ou não
#     RENDER = False # Se mostra a imagem do bot a jogar

#     SAVE_LOGS = False
#     OWN_LOSS_FUNCTION = False # Nossa propria funcao de loss ou mse
#     LOSS_FUNCTION = 'mse' # Funcao de loss usada

#     SAVE_COUNTER = 100 # Nº de episódios para que o modelo seja guardado
#     EPISODES = 10000 # Nº de episódios
#     TIMESTEPS = 1000 # Máximo de steps por episódio


# Guardar modelo atual(pesos)
def saveProgress(agent, e):
  agent.model.save_weights(SAVED_FILE_LOCATION)
  print("Saved: Episode " + str(e))

# Carregar modelo
def loadProgress(agent):
  try:
    agent.model.load_weights(SAVED_FILE_LOCATION)
  except ValueError:
    print("CRITICAL ERROR: model not found in" + SAVED_FILE_LOCATION + ". Please check if it exists")

# Criar gráfico para mostrar os scores do modelo
def plotScores(scores):

  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.plot(np.arange(len(scores)), scores)
  plt.xlabel('Episodio')
  plt.ylabel('Score')
  plt.show()

# ---------------------------------------------------------

# Programa principal
def main():

  #fazer setup de como os dados do np serão representados no print
  np.set_printoptions(precision=2)
  #criar ficheiro temporário para o ambiente
  tdir = tempfile.mkdtemp()

  #criar ambiente gym
  env = gym.make(ENV_NAME)
  #criar monitor gym sobre o ambiente
  env = wrappers.Monitor(env, tdir, force=True, video_callable=False)

  #obter primeira componente do espaço de observação e colocar em nS
  nS = env.observation_space.shape[0]
  #obter o valor discreto das acções que podemos tomar e colocar em nA
  nA = env.action_space.n

  #inicializar agente
  agent = DDQL(nS, nA)

  #inicializar scores
  scores = []
  scores_window = deque(maxlen=100) # Ultimos 100 scores

  #colocar em ep o número de episódios que vamos querer que ele faça no máximo
  ep = EPISODES

  # continuar o treino
  if (LOAD):
    loadProgress(agent)

  #se não estivermos a treinar o epsilon deve ser 0 de modo a impedir que o modelo seja alterado
  if (not TRAIN):
    agent.epsilon = 0
    ep = 100

  # Cumulative reward
  scores_window = deque(maxlen=100)

  #loop principal do programa
  for e in range(ep):
    #inicializar reward do episódio
    episode_reward = 0
    #fazer reset do ambiente
    s = env.reset()
    # reset variavel global time
    reset_time() if (OWN_LOSS_FUNCTION) else None
    #fazer reshape do array com base em nS
    s = np.reshape(s, [1, nS])

    #se estivermos para guardar e estivermos em modo de treino
    if (e%SAVE_COUNTER == 0 and TRAIN):
      saveProgress(agent, e)

    ultimoGanho = 0

    #loop para os timesteps em cada episódio
    for time in range(TIMESTEPS):
      #incrementar a variável global se tivermos definida a nossa função de perda
      inc_time() if (OWN_LOSS_FUNCTION) else None

      #se tivermos a flag ativa, fazer render do ambiente
      if (RENDER):
        env.render()

      # Query next action from learner and perform action
      #selecionar próxima acção
      a = agent.selectAction(s)
      #fazer ação
      s_prime, r, done, info = env.step(a)

      # Add cumulative reward
      #adicionar a reward do passo à reward do episódio
      episode_reward += r

      ultimoGanho = r

      # Reshape new state
      #fazer reshape de s_prime com base no nS (de modo a ser compativel com o reshape de s)
      s_prime = np.reshape(s_prime, [1, nS])

      # Add experience to memory
      if (TRAIN):
        agent.add_memory(s, a, r, s_prime, done)

      # Set current state to new state
      s = s_prime

      #Perform experience replay if memory length is greater than minibatch length
      if (TRAIN):
        if len(agent.memory) > agent.minibatch_size:
          agent.replay()

      # If episode is done, exit loop
      if done:
        if (TRAIN):
          agent.target_model_update()
        break

    # epsilon decay
    if agent.epsilon > agent.epsilon_min:
      agent.epsilon *= agent.epsilon_decay

    # Save the latest Score
    scores_window.append(episode_reward)
    scores.append(episode_reward)

    texto = 'Episode: ', e, ' Score: ', '%.2f' % episode_reward, ' Avg_Score: ', '%.2f' % np.average(scores_window), ' Frames: ', time, ' Epsilon: ', '%.2f' % agent.epsilon, '\n'
    print(texto)

    csv = str(e) + ";" +  str(episode_reward) + ";" +  str(np.average(scores_window)) + ";" +  str(time) + ";" +  str(agent.epsilon) + ";" +  str(ultimoGanho) +"\n"
    log(csv)


    # Considera-se vencido se tiver média de score superior a 200
    #e estiver em modo de treino(para evitar sair quando deve estar a mostrar)
    # if (np.mean(scores_window)>=200.0 and TRAIN):
    #   end_txt = '\n\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tfinal epsilon:{:.2f}\n\n'.format(e-100, np.mean(scores_window),agent.epsilon)

    #   print(end_txt)
    #   log(end_txt)
    #   saveProgress(agent,e)

    #   plotScores(scores_window)
    #   return scores

  #fechar o ambiente no final
  env.close()



# ---------------------------------------------------------
#correr programa
main()
