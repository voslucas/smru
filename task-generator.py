# Generate tasks for the task manager.
import os
import subprocess
import datetime

from task-manager import TaskManager 
taskManager = TaskManager("queue1")


models = ["smru2"]
rates = [0.01]
hiddens = [8]
layers = [1]
seeds = [1,2,3,4,5,6,7,8,9,10]
clipping = [0.0]
sequence = [56]
bmodes = ["bk"]
wmodes = ["nn","xn","xu","id"]
epochs = 15

for rate in rates:  
    for hidden in hiddens:
        for layer in layers:
            for clip in clipping:
                for seed in seeds:
                    for model in models:
                        for seq in sequence:
                            for bmode in bmodes:
                                for wmode in wmodes:
                                    task = "mnist" + str(seq)
                                    name = "{};{};{};{};{};{};{};{};{};{}".format(task,model,rate,hidden,layer,clip,seed,seq,bmode,wmode)
                                    fname = name + ".csv"
                                    if os.path.exists(fname)==False:
                                        command = "python mnist_test.py --modelname {} --rate {} --hidden {} --layers {} --clipping {} --seed {} --sequence {} --epochs {} --bmode {} --wmode {}".format(model,rate,hidden,layer,clip,seed,seq, epochs,bmode,wmode)
                            
                                        print("Adding task with name {} at {} by executing {}.".format(name,datetime.datetime.now(),command))
                                        taskManager.addTask(name,command)
