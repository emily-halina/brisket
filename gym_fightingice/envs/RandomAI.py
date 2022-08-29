from py4j.java_gateway import get_field
from time import sleep

from random import choice, seed
import tflearn
import os
import numpy as np
import tensorflow as tf


class RandomAI(object):
	def __init__(self, gateway):
		self.gateway = gateway

	def close(self):
		pass

	def getInformation(self, frameData, isControl, nonDelay):
		self.frameData = nonDelay
		self.isControl = isControl
		self.cc.setFrameData(self.frameData, self.player)
		self.nonDelay = nonDelay
		if frameData.getEmptyFlag():
			return
	# please define this method when you use FightingICE version 3.20 or later

	def roundEnd(self, x, y, z):
		print(x)
		print(y)
		print(z)
		self.action_ind = 0

	# please define this method when you use FightingICE version 4.00 or later
	def getScreenData(self, sd):
		pass

	def initialize(self, gameData, player):
		# Initializng the command center, the simulator and some other things
		self.inputKey = self.gateway.jvm.struct.Key()
		self.frameData = self.gateway.jvm.struct.FrameData()
		self.cc = self.gateway.jvm.aiinterface.CommandCenter()

		self.player = player
		self.gameData = gameData
		self.simulator = self.gameData.getSimulator()
		self.actions = ["NEUTRAL", "STAND", "FORWARD_WALK", "DASH", "BACK_STEP", "CROUCH", "JUMP", "FOR_JUMP", "BACK_JUMP", "AIR", "STAND_GUARD", "CROUCH_GUARD", "AIR_GUARD", "STAND_GUARD_RECOV", "CROUCH_GUARD_RECOV", "AIR_GUARD_RECOV", "STAND_RECOV", "CROUCH_RECOV", "AIR_RECOV", "CHANGE_DOWN", "DOWN", "RISE", "LANDING", "THROW_A", "THROW_B", "THROW_HIT", "THROW_SUFFER", "STAND_A",
						"STAND_B", "CROUCH_A", "CROUCH_B", "AIR_A", "AIR_B", "AIR_DA", "AIR_DB", "STAND_FA", "STAND_FB", "CROUCH_FA", "CROUCH_FB", "AIR_FA", "AIR_FB", "AIR_UA", "AIR_UB", "STAND_D_DF_FA", "STAND_D_DF_FB", "STAND_F_D_DFA", "STAND_F_D_DFB", "STAND_D_DB_BA", "STAND_D_DB_BB", "AIR_D_DF_FA", "AIR_D_DF_FB", "AIR_F_D_DFA", "AIR_F_D_DFB", "AIR_D_DB_BA", "AIR_D_DB_BB," "STAND_D_DF_FC"]
		
		self.tournament = False
		self.random_actions = []
		self.curr = 0
		if self.tournament:
			self.DNN = self.architecture("Final AI\\Skill 3 Sweeper")
			_actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
			self.action_strs = _actions.split(" ")
			self.action_vecs = {}
			self.create_action_vecs(self.action_strs, self.action_vecs)
		else:
			seed(31415)
			for i in range(10000):
				self.random_actions.append(choice(self.actions))
		print("hi")
		return 0

	def input(self):
		# Return the input for the current frame
		return self.inputKey

	def processing(self):
		#print("i'm tryin!")
		try:
			# Just compute the input for the current frame
			if self.frameData.getEmptyFlag() or self.frameData.getRemainingFramesNumber() <= 0:
				self.isGameJustStarted = True
				return

			if self.cc.getSkillFlag():
				self.inputKey = self.cc.getSkillKey()
				return

			self.inputKey.empty()
			self.cc.skillCancel()

			a = None
			if self.tournament:
				a = self.greedy(self.DNN, self.get_obs(), self.action_strs, self.action_vecs)
				#print(type(a))
				a = self.action_strs[a]
			else:
				a = self.random_actions[self.curr]
				self.curr += 1
			self.cc.commandCall(a)
			
		except Exception as e:
			print("ERROR IN RANDOM AI")
			print(e.args)
			return
	
	def architecture(self, name):
		# seed(6841922)
		tf.compat.v1.reset_default_graph()
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
		
	def greedy(self, DNN, environment, action_strs, action_vecs):
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
	
	def create_action_vecs(self, action_strs, action_vecs):
		"""
		init action_vecs (key = action name, value = onehot of action)
		"""
		for i in range(len(action_strs)):
			v = np.zeros(len(action_strs), dtype=np.float64)
			v[i] = 1
			action_vecs[action_strs[i]] = v
	
	def get_obs(self):
		my = self.frameData.getCharacter(self.player)
		opp = self.frameData.getCharacter(not self.player)

		# my information
		myHp = abs(my.getHp() / 100)
		myEnergy = my.getEnergy() / 300
		myX = ((my.getLeft() + my.getRight()) / 2) / 960
		myY = ((my.getBottom() + my.getTop()) / 2) / 640
		mySpeedX = my.getSpeedX() / 15
		mySpeedY = my.getSpeedY() / 28
		myState = my.getAction().ordinal()
		myRemainingFrame = my.getRemainingFrame() / 70

		# opp information
		oppHp = abs(opp.getHp() / 100)
		oppEnergy = opp.getEnergy() / 300
		oppX = ((opp.getLeft() + opp.getRight()) / 2) / 960
		oppY = ((opp.getBottom() + opp.getTop()) / 2) / 640
		oppSpeedX = opp.getSpeedX() / 15
		oppSpeedY = opp.getSpeedY() / 28
		oppState = opp.getAction().ordinal()
		oppRemainingFrame = opp.getRemainingFrame() / 70

		# time information
		game_frame_num = self.frameData.getFramesNumber() / 3600

		observation = []

		# my information
		observation.append(myHp)
		observation.append(myEnergy)
		observation.append(myX)
		observation.append(myY)
		if mySpeedX < 0:
			observation.append(0)
		else:
			observation.append(1)
		observation.append(abs(mySpeedX))
		if mySpeedY < 0:
			observation.append(0)
		else:
			observation.append(1)
		observation.append(abs(mySpeedY))
		for i in range(56):
			if i == myState:
				observation.append(1)
			else:
				observation.append(0)
		observation.append(myRemainingFrame)

		# opp information
		observation.append(oppHp)
		observation.append(oppEnergy)
		observation.append(oppX)
		observation.append(oppY)
		if oppSpeedX < 0:
			observation.append(0)
		else:
			observation.append(1)
		observation.append(abs(oppSpeedX))
		if oppSpeedY < 0:
			observation.append(0)
		else:
			observation.append(1)
		observation.append(abs(oppSpeedY))
		for i in range(56):
			if i == oppState:
				observation.append(1)
			else:
				observation.append(0)
		observation.append(oppRemainingFrame)

		# time information
		observation.append(game_frame_num)

		myProjectiles = self.frameData.getProjectilesByP1()
		oppProjectiles = self.frameData.getProjectilesByP2()

		if len(myProjectiles) == 2:
			myHitDamage = myProjectiles[0].getHitDamage() / 200.0
			myHitAreaNowX = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[
				0].getCurrentHitArea().getRight()) / 2) / 960.0
			myHitAreaNowY = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[
				0].getCurrentHitArea().getBottom()) / 2) / 640.0
			observation.append(myHitDamage)
			observation.append(myHitAreaNowX)
			observation.append(myHitAreaNowY)
			myHitDamage = myProjectiles[1].getHitDamage() / 200.0
			myHitAreaNowX = ((myProjectiles[1].getCurrentHitArea().getLeft() + myProjectiles[
				1].getCurrentHitArea().getRight()) / 2) / 960.0
			myHitAreaNowY = ((myProjectiles[1].getCurrentHitArea().getTop() + myProjectiles[
				1].getCurrentHitArea().getBottom()) / 2) / 640.0
			observation.append(myHitDamage)
			observation.append(myHitAreaNowX)
			observation.append(myHitAreaNowY)
		elif len(myProjectiles) == 1:
			myHitDamage = myProjectiles[0].getHitDamage() / 200.0
			myHitAreaNowX = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[
				0].getCurrentHitArea().getRight()) / 2) / 960.0
			myHitAreaNowY = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[
				0].getCurrentHitArea().getBottom()) / 2) / 640.0
			observation.append(myHitDamage)
			observation.append(myHitAreaNowX)
			observation.append(myHitAreaNowY)
			for t in range(3):
				observation.append(0.0)
		else:
			for t in range(6):
				observation.append(0.0)

		if len(oppProjectiles) == 2:
			oppHitDamage = oppProjectiles[0].getHitDamage() / 200.0
			oppHitAreaNowX = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[
				0].getCurrentHitArea().getRight()) / 2) / 960.0
			oppHitAreaNowY = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[
				0].getCurrentHitArea().getBottom()) / 2) / 640.0
			observation.append(oppHitDamage)
			observation.append(oppHitAreaNowX)
			observation.append(oppHitAreaNowY)
			oppHitDamage = oppProjectiles[1].getHitDamage() / 200.0
			oppHitAreaNowX = ((oppProjectiles[1].getCurrentHitArea().getLeft() + oppProjectiles[
				1].getCurrentHitArea().getRight()) / 2) / 960.0
			oppHitAreaNowY = ((oppProjectiles[1].getCurrentHitArea().getTop() + oppProjectiles[
				1].getCurrentHitArea().getBottom()) / 2) / 640.0
			observation.append(oppHitDamage)
			observation.append(oppHitAreaNowX)
			observation.append(oppHitAreaNowY)
		elif len(oppProjectiles) == 1:
			oppHitDamage = oppProjectiles[0].getHitDamage() / 200.0
			oppHitAreaNowX = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[
				0].getCurrentHitArea().getRight()) / 2) / 960.0
			oppHitAreaNowY = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[
				0].getCurrentHitArea().getBottom()) / 2) / 640.0
			observation.append(oppHitDamage)
			observation.append(oppHitAreaNowX)
			observation.append(oppHitAreaNowY)
			for t in range(3):
				observation.append(0.0)
		else:
			for t in range(6):
				observation.append(0.0)

		observation = np.array(observation, dtype=np.float64)
		observation = np.clip(observation, 0, 1)
		return observation

		
	# This part is mandatory
	class Java:
		implements = ["aiinterface.AIInterface"]
