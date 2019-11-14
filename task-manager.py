
# Class to support remote tasks in Azure Storage

import os
import time
import datetime
import subprocess
import random

from azure.storage.common import CloudStorageAccount
from subprocess import PIPE
from azure.common import (
    AzureConflictHttpError,
    AzureMissingResourceHttpError,
)



# Hardcoded SAS key to the storage account
SAS = '#PUT YOUR AZURE STORAGE ACCOUNT SAS KEY HERE'

TASKS_DIR = 'tasks'
RESULTS_DIR = 'results'
# Tasks 
# - are 'text files' stored in folder 'tasks' containing commandline actions
# - have extension .tsk
# - should be removed as soon as a worker is executing the command
# - result of task is captured and stored in 'results' 

class TaskManager():
    def __init__(self, queue):
        #read config
        self.account = CloudStorageAccount(account_name="pythontasks", sas_token=SAS)
        self.service = self.account.create_file_service()
        self.share = queue
        self.service.create_share(self.share, fail_on_exist=False)
        self.service.create_directory(self.share, TASKS_DIR)  # True
        self.service.create_directory(self.share, RESULTS_DIR)  # True


    def addTask(self, name, command):
        file_name = name + ".tsk"
        self.service.create_file_from_text(self.share, TASKS_DIR, file_name, command)


    def findTask(self):
        dir1 = list(self.service.list_directories_and_files(self.share, TASKS_DIR))
        candidates = []
        for res in dir1: 
            #split res.name in 
            base = os.path.basename(res.name)     # strip paths / etc.
            name = os.path.splitext(base)[0]  
            ext = os.path.splitext(base)[1]
            if ext==".tsk":
                print("Found a task.")
                exists = self.service.exists(self.share, TASKS_DIR, name + ".lck")
                if (exists==False):
                    candidates.append(name)
        if len(candidates)>0:
            return random.choice(candidates)
        else:
            return ""

    def executeTask(self, name):
        task_file_name = name + ".tsk"
        result_file_name = name + ".res"
        
        # make lck file
        self.service.create_file_from_text(self.share,TASKS_DIR,name + ".lck", "locked")

        # read commandline from task file
        command = self.service.get_file_to_text(self.share, TASKS_DIR, task_file_name)
        # run the task
        print("Starting with taskname {} at {} by executing {}.".format(name,datetime.datetime.now(),command.content))
        result=subprocess.run(command.content, shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True)
        if (result.returncode==0):
            self.service.create_file_from_text(self.share,RESULTS_DIR,result_file_name, result.stdout)
        else:
            self.service.create_file_from_text(self.share, RESULTS_DIR,result_file_name, str(result))
        # remove this task and lock
        self.service.delete_file(self.share,TASKS_DIR,task_file_name,timeout=None)
        self.service.delete_file(self.share,TASKS_DIR,name + ".lck",timeout=None)


    def monitor(self):
        while (True):
            name = self.findTask()
            if (name==""):
                print("No task found. Sleeping for one minute.")
                time.sleep(60)
            else:
                self.executeTask(name)


