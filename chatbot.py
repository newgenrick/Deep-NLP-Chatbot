# A simple deep learning chatbot built using seq2seq model 
# written in python with tensorflow api
# Author : Abhishek Awasthi 

# importing essenstial libraries

import tensorflow as tf
import numpy as np
import re
import time

# loading data
# the data used here is from cornell movie dataset 

lines = open("movie_lines.txt",encoding = "utf-8",errors = "ignore" ).read().split("\n")
conversations = open("movie_conversations.txt",encoding = "utf-8",errors = "ignore" ).read().split("\n")

#creating a dictionary of the line codes for lines list

id2line = {}

for _line in lines:
    _line = _line.split(" +++$+++ ")
    if(len(_line) == 5):
        id2line[_line[0]] = _line[4]

# creating a list of all conversations 

conversations_ids = []

for _conversation in conversations:
    _conversation = _conversation.split(" +++$+++ ")[-1][1:-1].replace("'","").replace(" ","").split(",")
    conversations_ids.append(_conversation)

    
questions = []
answers = []

for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
    