import gym
import gym_fightingice
from random import randint, random, seed, shuffle
import numpy as np
from time import sleep
import tflearn
import tensorflow as tf
import os
from uuid import uuid4
import csv
import sys

# 144

# [0 0 0 0 1/2 0 0 0 0 0 0 0 ]

if len(sys.argv) != 2:
	print("Usage: python3 diaynexe.py num_skills")
	exit(1)

num_skills = int(sys.argv[1])

def main():
	# CHANGABLE PARAMETERS: change these to swap models / modes
	name = "DIAYN\\Skill "
	names = []
	
	training = True
	episode_count = 50
	
	for i in range(1, num_skills + 1):
		names.append(name + str(i))
	# FIXED PARAMETERS: don't change these please :)

	# handling actions and conversions (a little messy, as I've ported this code between a couple different training environments)
	_actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
	action_strs = _actions.split(" ")
	action_vecs = {}
	create_action_vecs(action_strs, action_vecs) # initialize action_vecs (key = action name, value = onehot of action)
	
	# neural net
	DNN = architecture(names[1]) # initialize architecture, load based on name if possible
	env = None
	EPSILON_START = 0.95
	EPSILON_END = 0.05
	epsilon = EPSILON_START

	# gym code starts below ----------------------------------------------------------------------------- #
	for episode in range(1, episode_count + 1):
		print("beginning episode", episode)
		
		# determine annealing epsilon
		anneal = episode_count - episode + 1
		epsilon = max((anneal / episode_count) - 0.05, EPSILON_END)
		print("value of epsilon", epsilon)

		if training:
			tf.compat.v1.reset_default_graph()
			DNN = architecture(names[1]) # arbitrary chocie, this is later randomized
			env = training_episodes(DNN, env, action_strs, action_vecs, epsilon, names)
			train_DIAYN(name, clear=True)
		#env = evaluate(DNN, env, action_strs, action_vecs, epsilon=0)

def training_episodes(DNN, env, action_strs, action_vecs, epsilon, names):
	# create our environment and initialize
	if not env:
		env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path="",port=4242, freq_restart_java=100000)
		obs = env.reset()
	
	NUM_ROUNDS = 100 # scientific number
	round = 1
	PATH = "C:\\Users\\lalal\\Gym-FightingICE\\models\\"
	DRIVER_ONEHOTS = [[1,0,0], [0,1,0], [0,0,1]]
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

		# select which model is "driving"
		driver = randint(0, len(names) - 1)
		DNN.load(PATH + names[driver] + "\\model.tfl")
		print("Skill", driver + 1, "is driving")
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
			rewards.append(DRIVER_ONEHOTS[driver])
			#mean_rewards.append(np.mean(rewards))
			# print("frame", frame, "action", action_strs[action], "reward", reward)
			# input("enter to step")

			# move environment forward
			new_obs, reward, done, _ = env.step(action)
			#print(_)
			# update obs
			obs = new_obs
			

			# round is over
			if done:
				save_training_data(trainX, rewards)
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
	tf.compat.v1.reset_default_graph()
	# seed(6841922)
	try:
		# 143 (state) + 56 (action) = 199 input size
		net = tflearn.input_data([None, 199])
		net = tflearn.fully_connected(net, n_units=128, activation="relu")
		net = tflearn.fully_connected(net, n_units=64, activation="relu")
		net = tflearn.fully_connected(net, n_units=32, activation="relu")
		net = tflearn.fully_connected(net, n_units=1, activation="tanh")
		net = tflearn.regression(
			net, optimizer="adam", loss="mean_square", learning_rate=0.00001, batch_size=1)
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

def classifier(model_name, load=False):
	tf.compat.v1.reset_default_graph()
	net = tflearn.input_data([None, 199])
	net = tflearn.fully_connected(net, n_units=128, activation="relu")
	net = tflearn.fully_connected(net, n_units=64, activation="relu")
	net = tflearn.fully_connected(net, n_units=32, activation="relu")
	net = tflearn.fully_connected(net, n_units=num_skills, activation="softmax")

	net = tflearn.regression(net, optimizer = "adam", loss = "categorical_crossentropy", learning_rate = 0.00001, batch_size = 1)
	model = tflearn.DNN(net, checkpoint_path= os.getcwd() + "/models/" + model_name + "/")

	# load model, if it exists
	py_directory = os.getcwd()
	os.chdir(py_directory + "\\models\\" + model_name)

	try:
		model.load("model.tfl")
	except:
		print("No pre-existing classifier found, starting from scratch")
		raise
		model.save("model.tfl")
		model.load("model.tfl")
	finally:
		os.chdir(py_directory)
	return model

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

	ID = str(uuid4())
	print("round ended, saving data with ID", ID)

	# set up the arrays to be saved
	try:
		final_train_X = np.array(trainX, dtype=np.float64)

		final_train_Y = np.array(trainY, dtype=np.int32)
		print("skill", final_train_Y[0])
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

def train_DIAYN(name, clear=False):
	# first, load the classifier architecture for generating the reward

	# extract our training data
	trainX, class_trainY = classifier_training_data(True)
	print(len(trainX))
	print(trainX)
	# for each skill, we generate our trainY then train
	for skill in range(num_skills):
		# load classifier & generate reward
		tf.compat.v1.reset_default_graph()
		cl = classifier("DIAYN\\Classifier " + str(num_skills), load=True)
		trainY = generate_reward(cl, skill, trainX)

		# load skill and train
		tf.compat.v1.reset_default_graph()
		n = "DIAYN\\Skill " + str(skill + 1)
		DNN = architecture(n)
		train_model(DNN, n, trainX, trainY)
	
	# train the classifier
	tf.compat.v1.reset_default_graph()
	cl = classifier("DIAYN\\Classifier " + str(num_skills), load=True)

	# shuffle trainX & trainY grouped together
	trainX, class_trainY, testX, class_testY = shuffle_together(trainX, class_trainY)
	
	# c = list(zip(trainX, class_trainY))
	# shuffle(c)
	# trainX, class_trainY = zip(*c)

	# print(trainX)
	# print(class_trainY)

	train_model(cl, "DIAYN\\Classifier " + str(num_skills), trainX, class_trainY)
	eval_classifier(cl, testX, class_testY)

def shuffle_together(tX, tY):
	"""
	shuffle the order of the samples while keeping them together uwu they are friends or lovers or enemies (your interpretation) forever
	"""
	order = list(range(len(tX)))
	shuffle(order)
	trainX = []
	trainY = []
	testX = []
	testY = []
	for i in range(len(tX)):
		if i % 10 == 0:
			testX.append(tX[order[i]])
			testY.append(tY[order[i]])
		else:
			trainX.append(tX[order[i]])
			trainY.append(tY[order[i]])
	return trainX, trainY, testX, testY

def eval_classifier(cl, testX, testY):
	"""
	Calculate % accuracy and write to results csv
	"""
	pred = cl.predict(testX)
	correct = 0
	for i in range(len(pred)):
		if np.argmax(pred[i]) == np.argmax(testY[i]):
			correct += 1
			print(pred[i], testY[i])
	print(correct / len(testY))
	f = open("results.csv", "a")
	f.write("\n")
	f.write(str(correct / len(testY)))
	f.close()
	return

def generate_reward(cl, skill, trainX):
	trainY = []
	pred = cl.predict(trainX)
	for p in pred:
		trainY.append(np.log(p[skill]) - np.log(1/num_skills))
	# data shuffling
	trainY = np.array(trainY)
	trainY = np.expand_dims(trainY, 1)
	return trainY

def classifier_training_data(clear=False):
	'''
	Extract the training data from the data files in the train_data folder
	If clear is set to true, delete the data files after extracting info
	(helpful for running a LOT of games)
	'''
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
		

	print(trainY[0])
	print(trainY.shape)
	trainX = np.split(trainX, np.size(trainX, axis=0), axis=0)
	for i in range(len(trainX)):
		trainX[i] = np.squeeze(trainX[i])
	print(trainX[0].shape, "trainX shape")
	trainY = np.split(trainY, np.size(trainY, axis=0), axis=0)
	for i in range(len(trainY)):
		trainY[i] = np.squeeze(trainY[i])
	print(len(trainY))
	print(trainY[0].shape)

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

	EVAL_ROUNDS = 9
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
			# write results to csv
			f = open("results.csv", "a")
			for i in range(len(mean_rewards)):
				mean_rewards[i] = str(mean_rewards[i])
			f.write("\n")
			f.write(",".join(mean_rewards))
			f.close()
			return env


if __name__ == "__main__":
	main()