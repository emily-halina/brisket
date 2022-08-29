import os
import sys
import numpy as np
import tflearn
import tensorflow as tf

# change this flag to test your models init'd correctly (sanity check)
testing = False

def architecture(model_name):
    net = tflearn.input_data([None, 199])
    net = tflearn.fully_connected(net, n_units=128, activation="relu")
    net = tflearn.fully_connected(net, n_units=64, activation="relu")
    net = tflearn.fully_connected(net, n_units=32, activation="relu")
    net = tflearn.fully_connected(net, n_units=1, activation="tanh")
    net = tflearn.regression(
        net, optimizer="adam", loss="mean_square", learning_rate=0.00001, batch_size=1)

    net = tflearn.regression(net, optimizer = "adam", loss = "mean_square", learning_rate = 0.00001, batch_size = 1)
    model = tflearn.DNN(net, checkpoint_path= os.getcwd() + "/models/" + model_name + "/")

    model.save("model.tfl")
    return model

def classifier(model_name, load=False):
    net = tflearn.input_data([None, 199])
    net = tflearn.fully_connected(net, n_units=128, activation="relu")
    net = tflearn.fully_connected(net, n_units=64, activation="relu")
    net = tflearn.fully_connected(net, n_units=32, activation="relu")
    net = tflearn.fully_connected(net, n_units=int(sys.argv[1]), activation="softmax")

    net = tflearn.regression(net, optimizer = "adam", loss = "categorical_crossentropy", learning_rate = 0.0001, batch_size = 1)
    model = tflearn.DNN(net, checkpoint_path= os.getcwd() + "/models/" + model_name + "/")

    if not load:
        model.save("model.tfl")
    return model

def test():
    net = tflearn.input_data([None, 199])
    net = tflearn.fully_connected(net, n_units=128, activation="relu")
    net = tflearn.fully_connected(net, n_units=64, activation="relu")
    net = tflearn.fully_connected(net, n_units=32, activation="relu")
    net = tflearn.fully_connected(net, n_units=1, activation="tanh")
    net = tflearn.regression(
        net, optimizer="adam", loss="mean_square", learning_rate=0.00001, batch_size=1)

    net = tflearn.regression(net, optimizer = "adam", loss = "mean_square", learning_rate = 0.0001, batch_size = 1)
    model = tflearn.DNN(net)

    cwd = os.getcwd()
    model.load(cwd + "/models/DIAYN/Skill 1/model.tfl")

    a = np.random.rand(199)
    print(model.predict([a]))
    model.load(cwd + "/models/DIAYN/Skill 2/model.tfl")
    print(model.predict([a]))
    model.load(cwd + "/models/DIAYN/Skill 3/model.tfl")
    print(model.predict([a]))

    tf.compat.v1.reset_default_graph()
    c = classifier("Classifier 3", True)
    c.load(cwd + "/models/DIAYN/Classifier 3/model.tfl")
    print(c.predict([a]))
    

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 initdiayn.py num_skills")
        return
    
    if testing:
        test()
        return

    num_skills = -1
    try:
        num_skills = int(sys.argv[1])
    except:
        print("Error: num_skills should be an integer.\nUsage: python3 initdiayn.py num_skills")
        exit(1)

    py_dir = os.getcwd()
    diayn_dir = py_dir + "/models/DIAYN/"
    os.chdir(diayn_dir)
    model_dir_list = os.listdir()

    # init the given amount of models
    for skill in range(1, num_skills + 1):
        tf.compat.v1.reset_default_graph()
        # if a model exists, skip it: delete stuff manually, we don't want to overwrite
        name = "Skill " + str(skill)
        if name in model_dir_list:
            continue

        # make the dir and get into the right place
        os.mkdir(name)
        os.chdir(name)

        # define the architecture and save it (if it doesn't exist)
        model = architecture(name)

        # get back to the correct place for next iter
        os.chdir(diayn_dir)
        print(name, "created successfully!")

    # now we init the classifier
    tf.compat.v1.reset_default_graph()
    class_name = "Classifier " + str(num_skills)
    os.mkdir(class_name)
    os.chdir(class_name)
    classifier(class_name)

    print("classifier initialized!")
main()
