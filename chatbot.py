# A simple deep learning chatbot built using seq2seq model 
# written in python with tensorflow api
# Author : Abhishek Awasthi 

# importing essenstial libraries

import tensorflow as tf
import numpy as np
import re
import time
from collections import Counter
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
    
        
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm","i am",text)
    text = re.sub(r"he's","he is",text)
    text = re.sub(r"she's","she is",text)
    text = re.sub(r"that's","that is",text)
    text = re.sub(r"what's","what is",text)
    text = re.sub(r"where's","where is",text)
    text = re.sub(r"\'ll","will",text)
    text = re.sub(r"\'ve","have",text)
    text = re.sub(r"\'re","are",text)
    text = re.sub(r"\'d","would",text)
    text = re.sub(r"won't","will not",text)
    text = re.sub(r"can't","cannot",text)
    text = re.sub(r"[-()\*<>;:!@#$%^&*\".,]","",text)
    return text
    
cleaned_questions = []

for question in questions:
    cleaned_questions.append(clean_text(question))
    
cleaned_answers = []
        
for answer in answers:
    cleaned_answers.append(clean_text(answer))
    
all_questions = " ".join(cleaned_questions)
all_answers = " ".join(cleaned_answers)

question_words = all_questions.split()
answer_words = all_questions.split()

qword_count = Counter(question_words).most_common()
aword_count = Counter(answer_words).most_common()

threshold = 20
answerword2int = dict()
questionword2int = dict()

for i,(key,count) in enumerate(aword_count):
    if count>threshold:
        answerword2int[key] = i
        
for i,(key,count) in enumerate(qword_count):
    if count>threshold:
        questionword2int[key] = i       
        
        
        
tokens  = ["<EOS>","<PAD>","<OUT>","<SOS>"]

for token in tokens:
    questionword2int[token] = len(questionword2int) +1
    
for token in tokens:
    answerword2int[token] = len(answerword2int) +1
    
answerint2word = {w_i:w for w,w_i in answerword2int.items()}        
questionint2word = {w_i:w for w,w_i in questionword2int.items()}        

for i in range(len(cleaned_answers)):
    cleaned_answers[i] += ' <EOS>'
        
question_to_int = []
for question in cleaned_questions:
    ints = []
    for word in question.split():
        if word not in questionword2int:
            ints.append(questionword2int['<OUT>'])
        else:
            ints.append(questionword2int[word])
    question_to_int.append(ints)
    
    
        
answer_to_int = []
for answer in cleaned_answers:
    ints = []
    for word in answer.split():
        if word not in answerword2int:
            ints.append(answerword2int['<OUT>'])
        else:
            ints.append(answerword2int[word])
    answer_to_int.append(ints)
    

enumerate_questions_sorted = sorted(enumerate(question_to_int),key=lambda x: len(x[1]) )
k = True      
for i in range(len(enumerate_questions_sorted)):
    if(k and len(enumerate_questions_sorted[i][1])>0):
        start = i
        k = False
    if(len(enumerate_questions_sorted[i][1])>25):
        stop = i
        break
    
sorted_questions = [x[1] for x in enumerate_questions_sorted[start:stop]]
sorted_answers = [answer_to_int[x[0]] for x in enumerate_questions_sorted[start:stop]]


# seq2seq start ----------------------------------------------------------------->>

def create_placeholders():
    inputs = tf.placeholder(tf.int32,[None,None],name = 'inputs')
    targets = tf.placeholder(tf.int32,[None,None],name = 'targets')
    lr = tf.placeholder(tf.float32,name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32,name = 'keep_prob')
    return inputs,targets,lr,keep_prob


def create_batches(targets,word2int,batch_size):
    left_side = tf.fill([batch_size,1],word2int['<SOS>'])
    right_side = tf.strided_slice(targets,[0,0],[batch_size,-1],[1,1])
    batch = tf.concat([left_side,right_side],1)
    return batch

def encoder_rnn_layer(rnn_inputs,rnn_size,num_layers,keep_prob,sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
    _,encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                      cell_bw = encoder_cell,
                                                      sequence_length = sequence_length,
                                                      inputs = rnn_inputs,
                                                      dtype = tf.float32)
    return encoder_state

