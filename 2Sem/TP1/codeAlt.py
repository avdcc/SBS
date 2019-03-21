
import gym, random, tempfile
import numpy as np

from gym import wrappers
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# ---------------------------------------------------------

#nome do ambiente
ENV_NAME = "LunarLander-v2" # "Breakout-v0"

#flags:
#
LOAD = False
#
TRAIN = True
#
RENDER = False
#
SAVE_LOGS = False

#
SAVE_COUNTER = 100
#nº de episódios
EPISODES = 1000
#máximo de steps por episódio
TIMESTEPS = 1000

#nome do ficheiro onde será guardado o modelo
SAVED_FILE_LOCATION = "./" + ENV_NAME + ".h5"



# ---------------------------------------------------------

#classe do modelo de machine learning
class DDQL:
  #inicialização de modelo
  def __init__(self, nS, nA):
    #
    self.nS = nS
    #
    self.nA = nA
    #
    self.epsilon = 1
    #
    self.epsilon_min = 0.01
    #
    self.epsilon_decay = 0.9993
    #
    self.gamma = 0.99
    #
    self.learning_rate = 0.0001
    #
    self.epochs = 1
    #
    self.verbose = 0
    #
    self.minibatch_size = 30
    #
    self.memory = deque(maxlen=5000)
    #inicialização do modelo 
    self.model = self.create_model()
    #
    self.target_model = self.create_model()

  #função de criação do modelo
  def create_model(self):
    #inicializar modelo
    model = Sequential()

    # Add 2 hidden layers with 64 nodes each
    model.add(Dense(64, input_dim=self.nS, activation='relu'))
    model.add(Dense(64, activation='relu'))
    #camada de output
    model.add(Dense(self.nA, activation='linear'))
    #compilar modelo com mse e otimizador Adam
    model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    return model

  #
  def add_memory(self, s, a, r, s_prime, done):
    self.memory.append((s, a, r, s_prime, done))

  #
  def target_model_update(self):
    self.target_model.set_weights(self.model.get_weights())

  #
  def selectAction(self, s):
    #
    if np.random.rand() <= self.epsilon:
      return np.random.choice(self.nA)
    #
    q = self.model.predict(s)
    return np.argmax(q[0])

  #
  def replay(self):
    # Vectorized method for experience replay
    #
    minibatch = random.sample(self.memory, self.minibatch_size)
    #
    minibatch = np.array(minibatch)
    #
    not_done_indices = np.where(minibatch[:, 4] == False)
    #
    y = np.copy(minibatch[:, 2])

    # If minibatch contains any non-terminal states, use separate update rule for those states
    if len(not_done_indices[0]) > 0:
      #
      predict_sprime = self.model.predict(np.vstack(minibatch[:, 3]))
      #
      predict_sprime_target = self.target_model.predict(np.vstack(minibatch[:, 3]))
      
      # Non-terminal update rule
      #
      y[not_done_indices] += np.multiply(self.gamma, \
            predict_sprime_target[not_done_indices, \
            np.argmax(predict_sprime[not_done_indices, :][0], axis=1)][0])

    #
    actions = np.array(minibatch[:, 1], dtype=int)
    #
    y_target = self.model.predict(np.vstack(minibatch[:, 0]))
    #
    y_target[range(self.minibatch_size), actions] = y
    #
    self.model.fit(np.vstack(minibatch[:, 0]), y_target, epochs=self.epochs, verbose=self.verbose)

  #
  def replayIterative(self):
    # Iterative method - this performs the same function as replay() but is not vectorized 
    #
    s_list = []
    y_state_list = []
    #
    minibatch = random.sample(self.memory, self.minibatch_size)
    #
    for s, a, r, s_prime, done in minibatch:
      #
      s_list.append(s)
      #
      y_action = r
      #
      if not done:
        y_action = r + self.gamma * np.amax(self.model.predict(s_prime)[0])

      #
      print(y_action)
      
      #
      y_state = self.model.predict(s)
      #
      y_state[0][a] = y_action
      #
      y_state_list.append(y_state)
    #
    self.model.fit(np.squeeze(s_list), np.squeeze(y_state_list), batch_size=self.minibatch_size, epochs=1, verbose=0)



# ---------------------------------------------------------

#fazer log de um texto para logs.txt
def log(texto):
  with open("logs.txt", "a") as myfile:
    myfile.write(log)


#guardar modelo atual(pesos)
def saveProgress(agent, e):
  agent.model.save_weights(SAVED_FILE_LOCATION)
  print("Saved: Episode " + str(e))

#carregar modelo 
def loadProgress(agent):
  agent.model.load_weights(SAVED_FILE_LOCATION)
  #
  try:
    agent.model.load_weights(SAVED_FILE_LOCATION)
  except ValueError:
    print("CRITICAL ERROR: model not found in" + SAVED_FILE_LOCATION + ". Please check if it exists")

#criar gráfico para mostrar os scores do modelo
def plotScores(scores):

  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.plot(np.arange(len(scores)), scores)
  plt.xlabel('Episodio')
  plt.ylabel('Score')
  plt.show()


# ---------------------------------------------------------

#programa principal
def main():
  
  #
  np.set_printoptions(precision=2)
  #
  tdir = tempfile.mkdtemp()
  #criar ambiente gym
  env = gym.make(ENV_NAME)
  #
  env = wrappers.Monitor(env, tdir, force=True, video_callable=False)

  #
  nS = env.observation_space.shape[0]
  #
  nA = env.action_space.n

  #
  agent = DDQL(nS, nA)

  #
  scores = []
  scores_window = deque(maxlen=100) # Ultimos 100 scores


  ep = EPISODES
  #
  if (not TRAIN): 
    agent.epsilon = 0
    ep = 100

  # Cumulative reward
  scores_window = deque(maxlen=100)

  #loop principal do programa
  for e in range(ep):

    episode_reward = 0
    #
    s = env.reset()
    #
    s = np.reshape(s, [1, nS])

    #
    if (e%SAVE_COUNTER == 0):
      saveProgress(agent, e)

    #
    for time in range(TIMESTEPS):

      #
      if (RENDER):
        env.render()

      # Query next action from learner and perform action
      a = agent.selectAction(s)
      s_prime, r, done, info = env.step(a)

      # Add cumulative reward
      episode_reward += r

      # Reshape new state
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

    #texto de debug 
    texto = 'Episode: ', e, ' Score: ', '%.2f' % episode_reward, ' Avg_Score: ', '%.2f' % np.average(scores_window), ' Frames: ', time, ' Epsilon: ', '%.2f' % agent.epsilon

    print(texto)
    log(texto)
    
    # Considera-se vencido
    if np.mean(scores_window)>=200.0:
      print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e-100, np.mean(scores_window)))
      plotScores(scores)
      return scores
    
    # with open('trained_agent.txt', 'a') as f:
    #   f.write(str(np.average(scores_window)) + '\n')

  env.close()



# ---------------------------------------------------------
#correr programa
main()
