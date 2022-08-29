import gym
import gym_fightingice
from random import randint, random, seed
import numpy as np
from time import sleep
import tflearn
import os
from uuid import uuid4
import csv

# 144

# [0 0 0 0 1/2 0 0 0 0 0 0 0 ]

def main():
	# CHANGABLE PARAMETERS: change these to swap models / modes
	#name = "DIAYN\\Skill 1 Ep 17"
	name = "Final AI\\HA 3 Counter"
	training = True
	train_eval = True
	test_eval = False
	episode_count = 50
	p = "Final AI\\"
	#model_list = [p + 'Skill 1 Combo', p + "Skill 2 Rushdown", p + "Skill 3 Sweeper", p + "HA 1 Aggressive", p + "HA 2 Balanced", p + "HA 3 Counter"]
	model_list = [p + 'Skill 1 Combo', p + "Skill 2 Rushdown", p + "HA 1 Aggressive", p + "HA 2 Balanced", p + "HA 3 Counter"]

	# FIXED PARAMETERS: don't change these please :)

	# handling actions and conversions (a little messy, as I've ported this code between a couple different training environments)
	_actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
	action_strs = _actions.split(" ")
	action_vecs = {}
	create_action_vecs(action_strs, action_vecs) # initialize action_vecs (key = action name, value = onehot of action)
	
	# neural net
	DNN = architecture(name) # initialize architecture, load based on name if possible
	env = None
	EPSILON_START = 0.95
	EPSILON_END = 0.05
	epsilon = EPSILON_START

	# gym code starts below ----------------------------------------------------------------------------- #
	if test_eval:
		for m in model_list:
			env = versus(DNN, env, action_strs, action_vecs, m, epsilon=0)
		exit()
	for episode in range(1, episode_count + 1):
		print("beginning episode", episode)
		
		# determine annealing epsilon
		#anneal = episode_count - episode + 1
		#epsilon = max((anneal / episode_count) - 0.05, EPSILON_END)
		epsilon = 0.05
		print(epsilon)

		if training:
			env = training_episodes(DNN, env, action_strs, action_vecs, epsilon)
			train_DNN(DNN, name, clear=True)
		if train_eval:
			env = evaluate(DNN, env, action_strs, action_vecs, epsilon=0)
		

def training_episodes(DNN, env, action_strs, action_vecs, epsilon):
	# create our environment and initialize
	if not env:
		env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path="",port=4242, freq_restart_java=100000)
		obs = env.reset()
	
	NUM_ROUNDS = 100 # scientific number
	round = 1
	#mean_rewards = []
	# main loop
	while True:
		# round init
		obs = env.reset()
		done = False
		obs = obs[0]
		new_obs = obs
		action = 0

		# training data
		trainX = []
		rewards = []

		print("beginning round", round)
		frame = 0
		tX = []
		reward = 0
		# while round not complete
		while not done:
			frame += 1
			tX.append(frame)

			# getting around a round start bug
			if type(obs) != np.float64:
				action = epsilon_greedy(DNN, obs, action_strs, action_vecs, epsilon=epsilon)
			else:
				print("else activated")
				action = 0
				new_obs, reward, done, _ = env.step(action)
				obs = new_obs
				continue
			
			# collect training data
			trainX.append(np.concatenate((obs, action_vecs[action_strs[action]])))
			rewards.append(reward)
			
			# print("frame", frame, "action", action_strs[action], "reward", reward)
			# input("enter to step")

			# move environment forward
			new_obs, reward, done, _ = env.step(action)
			#print(_)
			# update obs
			obs = new_obs
			

			# round is over
			if done:

				# calculate final reward
				win_loss(obs, rewards, reward)

				# print(rewards, "rewards raw")
				# calculate our trainY and save our training data
				#print(mock_backprop(tX, rewards))
				#input("wait")

				trainY = backprop(DNN, trainX, rewards)

				# round end pause
				# print(rewards)
				# print(trainY)
				# input("waiting..")
				save_training_data(trainX, trainY)
				round += 1

		if round > NUM_ROUNDS:
			print("done training episode")
			# f = open("results.csv", "a")
			# for i in range(len(mean_rewards)):
			# 	mean_rewards[i] = str(mean_rewards[i])
			# f.write("\n")
			# f.write(",".join(mean_rewards))
			# f.close()

			env.reset()
			return env

def create_action_vecs(action_strs, action_vecs):
	"""
	init action_vecs (key = action name, value = onehot of action)
	"""
	for i in range(len(action_strs)):
		v = np.zeros(len(action_strs), dtype=np.float64)
		v[i] = 1
		action_vecs[action_strs[i]] = v


# model architecture
def architecture(name):
	# seed(6841922)
	try:
		# 143 (state) + 56 (action) = 199 input size
		net = tflearn.input_data([None, 199])
		net = tflearn.fully_connected(net, n_units=128, activation="relu")
		net = tflearn.fully_connected(net, n_units=64, activation="relu")
		net = tflearn.fully_connected(net, n_units=32, activation="relu")
		net = tflearn.fully_connected(net, n_units=1, activation="tanh")
		net = tflearn.regression(
			net, optimizer="adam", loss="mean_square", learning_rate=0.000001, batch_size=1)
		model = tflearn.DNN(net, checkpoint_path="C:\\Users\\lalal\\Gym-FightingICE\\models\\" + name + "\\")
	except Exception as e:
		print(e)

	# load model, if it exists
	py_directory = os.getcwd()
	os.chdir(py_directory + "\\models\\" + name)

	try:
		model.load("model.tfl")
	except:
		print("No pre-existing model found, starting from scratch")
		model.save("model.tfl")
		model.load("model.tfl")
	finally:
		os.chdir(py_directory)

	return model

# 0 - 1

# hypothesis -> losing during training so it prioritizes "not losing" because reward is outweighed
# possible solutions:


def greedy(DNN, environment, action_strs, action_vecs):
	# greedily choose the best possible action
	best = 0
	best_val = float("-inf")
	#values = []
	for a in range(len(action_strs)): # 0 , 55
		act_vector = action_vecs[action_strs[a]]
		input = np.concatenate((environment, act_vector))
		pred = DNN.predict([input])
		#if pred > 0:
			#values.append(action_strs[a])
		if pred > best_val:
			best = a
			best_val = pred
	#print(values)
	return best

def epsilon_greedy(DNN, environment, action_strs, action_vecs, epsilon=0.01):
	# if below a certain ep, do a random action, otherwise pick greedily
	if random() < epsilon:
		return randint(0, 55)
	else:
		return greedy(DNN, environment, action_strs, action_vecs)

def win_loss(obs, rewards, final_reward):
	rewards[len(rewards) - 1] += final_reward

	A_LOT = 1000
	if final_reward == A_LOT:
		print("positive win!")
		return 1
	elif final_reward == -A_LOT:
		print("negative loss!")		
		return -1
	return 0
def backprop(DNN, trainX, rewards, alpha=0.5, gamma=0.95, penalty=0.05):
	# calculate trainY's, starting from the end & back propping to the beginning
	trainX.reverse()
	rewards.reverse()
	trainY = []

	for k in range(len(trainX)):
		# get the info required to compute Y
		X = trainX[k]

		# get the current value of Q[state + action] in the table
		# [0][0] is because it returns the value nested in 2 lists -_-
		current_x_pred = DNN.predict([X])[0][0]
		
		# compute Y as follows
		# Q(s_t, a_t) = Q(s_t, a_t) + alpha * [ r_t + gamma * max_a( Q[s_t+1, a]) - Q(s_t, a_t) ] - penalty
		
		# get the "next" Y, which is (max_a(Q[s_{t+1}, a]))
		next_y = None
		if k == 0:
			next_y = 0
		else:
			next_y = trainY[k-1]

		Y = current_x_pred + alpha * (rewards[k] + gamma * (next_y - current_x_pred))

		trainY.append(Y)
	#print(trainY, "trainY")
	return trainY

def mock_backprop(tX, rewards):
	tX.reverse()
	rewards.reverse()
	r_pair = []
	for k in range(len(tX)):
		r_pair.append((tX[k], rewards[k]))
	return r_pair

def save_training_data(trainX, trainY):
	# save the training data collected if we are training!
	# generate unique ID

	#print(trainY, "train Y")
	#print(type(trainY))
	ID = str(uuid4())
	print("round ended, saving data with ID", ID)

	# set up the arrays to be saved
	try:
		final_train_X = np.array(trainX, dtype=np.float64)

		final_train_Y = np.array(trainY, dtype=np.float64)

		print(final_train_X.shape, "train X shape",
				final_train_Y.shape, "train Y shape")
		X_file = "trainX-" + ID
		Y_file = "trainY-" + ID

		# save the files
		cwd = os.getcwd()
		os.chdir(cwd + "\\train_data")

		np.save(X_file, final_train_X, allow_pickle=True)
		np.save(Y_file, final_train_Y, allow_pickle=True)
		os.chdir(cwd)

	except Exception as e:
		print(e)
		raise

def train_DNN(DNN, name, clear=False):
	trainX, trainY = extract_training_data(clear)
	train_model(DNN, name, trainX, trainY)

def extract_training_data(clear=False):
    '''
    Extract the training data from the data files in the train_data folder
    If clear is set to true, delete the data files after extracting info
    (helpful for running a LOT of games)
    '''
    A_LOT = 1000
    trainX_files = []
    trainY_files= []

    py_directory = os.getcwd()
    os.chdir(py_directory + "\\train_data") #TODO: change later
    files = os.listdir()
    print(files)

    for f in files:

        if f.startswith("trainX"):
            trainX_files.append(f)
        elif f.startswith("trainY"): # reminder that it tried to open DS.Store because i just said "else" here before, foolish
            trainY_files.append(f)

    trainX_files.sort()
    trainY_files.sort()

    #trainX = np.empty((0,0))
    #trainY = np.array([], dtype=np.float64)
    x_init = True
    y_init = True
    trainX = None
    trainY = None

    for xfile in trainX_files:
        if x_init:
            trainX = np.load(xfile)
            x_init = False
        else:
            x = np.load(xfile)
            trainX = np.concatenate((trainX, x))
        #trainX.append(np.load(xfile))

    print(trainX.shape)
    print(trainX[0][0])

    for yfile in trainY_files:
        if y_init:
            trainY = np.load(yfile)
            y_init = False
        else:
            y = np.load(yfile)
            trainY = np.concatenate((trainY, y))
        #trainY.append(np.load(xfile))
    print(trainY[0])
    print(trainY.shape)
    trainX = np.split(trainX, np.size(trainX, axis=0), axis=0)
    for i in range(len(trainX)):
        trainX[i] = np.squeeze(trainX[i])
    print(trainX[0].shape, "trainX shape")
    trainY = np.split(trainY, np.size(trainY, axis=0), axis=0)
    print(len(trainY))
    print(trainY[0].shape)

    for Y in trainY:
        Y /= A_LOT

    # cleanup the replays if requested
    if clear:
        for f in files:
            os.remove(f)

    os.chdir(py_directory)
    return trainX, trainY

def train_model(model, name, trainX, trainY):
    cwd = os.getcwd()
    os.chdir(cwd + "\\models\\" + name)
    model.fit(trainX, trainY, n_epoch=5, validation_set=0.1, show_metric=False, batch_size=1, snapshot_epoch=True, snapshot_step=False)
    model.save("model.tfl")
    os.chdir(cwd)


def evaluate(DNN, env, action_strs, action_vecs, epsilon):
	# create our environment and initialize
	if not env:
		env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path="",port=4242, freq_restart_java=100000)
		obs = env.reset()

	EVAL_ROUNDS = 3 #TODO change back to 9
	episode = 1
	mean_rewards = []
	# main loop
	while True:
		# round init
		obs = env.reset()
		done = False
		obs = obs[0]
		new_obs = obs
		action = 0

		# training data
		trainX = []
		rewards = []

		print("beginning evaluation game", episode)
		if episode == 1:
			input("waiting for emily to start recording")
		# while round not complete
		while not done:
			# getting around a round start bug
			if type(obs) != np.float64:
				action = epsilon_greedy(DNN, obs, action_strs, action_vecs, epsilon)
			else:
				print("else activated")
				action = 0
				new_obs, reward, done, _ = env.step(action)
				obs = new_obs
				continue

			# move environment forward
			new_obs, reward, done, _ = env.step(action)

			# collect training data
			trainX.append(np.concatenate((obs, action_vecs[action_strs[action]])))
			rewards.append(reward)
			
			# update obs
			obs = new_obs

			# round is over
			if done:
				# calculate final reward
				win_loss(obs, rewards, reward)
				
				mean_rewards.append(np.mean(rewards))
				episode += 1

		if episode > EVAL_ROUNDS:
			input("all done, waiting for emily to say its okay to restart..")

			# write results to csv
			f = open("results.csv", "a")
			for i in range(len(mean_rewards)):
				mean_rewards[i] = str(mean_rewards[i])
			f.write("\n")
			f.write(",".join(mean_rewards))
			f.close()
			return env

def versus(DNN, env, action_strs, action_vecs, name, epsilon):
	# create our environment and initialize
	if not env:
		env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path="",port=4242, freq_restart_java=100000)
		obs = env.reset()

	EVAL_ROUNDS = 5
	episode = 1
	line = [name + " vs P2", 0, 0, 0]

	# set up for this round
	PATH = "C:\\Users\\lalal\\Gym-FightingICE\\models\\"
	DNN.load(PATH + name + "\\model.tfl")

	# main loop
	while True:
		# round init
		obs = env.reset()
		done = False
		obs = obs[0]
		new_obs = obs
		action = 0

		# training data
		trainX = []
		rewards = []

		print("beginning evaluation game", episode)

		# while round not complete
		while not done:
			# getting around a round start bug
			if type(obs) != np.float64:
				action = epsilon_greedy(DNN, obs, action_strs, action_vecs, epsilon)
			else:
				print("else activated")
				action = 0
				new_obs, reward, done, _ = env.step(action)
				obs = new_obs
				continue

			# move environment forward
			new_obs, reward, done, _ = env.step(action)

			# collect training data
			trainX.append(np.concatenate((obs, action_vecs[action_strs[action]])))
			rewards.append(reward)
			
			# update obs
			obs = new_obs

			# round is over
			if done:
				# calculate final reward
				record = win_loss(obs, rewards, reward)
				if record == 1:
					line[1] += 1
				elif record == -1:
					line[2] += 1
				else:
					line[3] += 1
				episode += 1

		if episode > EVAL_ROUNDS:
			# write results to csv
			f = open("tourney_results.csv", "a")
			for i in range(len(line)):
				line[i] = str(line[i])
			f.write("\n")
			f.write(",".join(line))
			f.close()
			return env


if __name__ == "__main__":
	main()