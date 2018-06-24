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
    text = re.sub(r"[-()\*<>;:!@#$%^&*.,]","",text)
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
        
        
        
        
        
        
        
        
        