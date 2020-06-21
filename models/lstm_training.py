#Importing requried packages
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import numpy as np
import re
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize,sent_tokenize
#Loading the files
with open("dataset/train.context", "rb") as cfile:
    context = cfile.read().decode('utf-8').split("\n")
with open("dataset/train.question", "rb") as cfile:
    questions = cfile.read().decode('utf-8').split("\n")
with open("dataset/train.span", "rb") as cfile:
    span = cfile.read().decode('utf-8').split("\n")
with open("dataset/train.answer", "rb") as cfile:
    the_answers = cfile.read().decode('utf-8').split("\n")

#New lists for holding the results after anaphora resolution
res_contexts = []
res_questions = []
res_spans = []
res_answers = []

#Code for anaphora (coreference) resolution

import spacy
nlp = spacy.load('en')
import neuralcoref
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')

#Resolving the contexts, questions and answers and updating the original variables

for i in range(len(context)):
    c = context[i]
    q = questions[i]
    a = the_answers[i]
    res_contexts.append (nlp(c)._.coref_resolved)
    res_questions.append(nlp(q)._.coref_resolved)
    res_answers.append (nlp(a)._.coref_resolved)

context = res_contexts
questions = res_questions
the_answers = res_answers

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
lemmatizer=WordNetLemmatizer()
import string
stop_words = set(stopwords.words('english'))
punctuations=string.punctuation
#To tokenize given sentence into words
def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
#Function to preprocess a given context
#Involves: converting to lower case, tokenization, lemmatization and update the mappings(span)
def preprocess(context):
    actual=context
    tok=word_tokenize(context)
    mappings=[-1]*len(tok)
    context=(' ').join(tok)
    process=context.lower()
    lemmatized=[]
    tokens=word_tokenize(process)
    for word in tokens:
        lemmatized.append(lemmatizer.lemmatize(word))  
    final=""
    cnt=0
    for i in range(len(mappings)):
        if(lemmatized[i] not in stop_words and lemmatized[i] not in punctuations):
            final+=lemmatized[i]+" "
            mappings[i]=cnt
            cnt+=1
    final=final.strip()
    return final,mappings
#Maps original span to new span
def get_span(span,mapping):
    y=0
    z=0
    x=[int(i) for i in span.split(' ')]
    for i in range(x[0],x[1]+1):
        if(mapping[i]!=-1):
            y=mapping[i]
            break
    for j in range(x[1],x[0]-1,-1):
        if(mapping[j]!=-1):
            z=mapping[j]
            break
    if(y>z): 
        z=y
    new_span=str(y)+' '+str(z)
    return new_span
    
max_context = []
for a in context:
    max_context.append(len(tokenize(a)))
    
max_question = []
for a in questions:
    max_question.append(len(tokenize(a)))
    
max_span = []
for a in span:
    max_span.append(len(tokenize(a)))
#Considering 500 words for context and 36 for question
#Preprocessing each context and question
context_len = 500
question_len = 36
train_context = []
train_question = []
short_answers = []
for a in range(len(context)-1):
    if max_context[a]<=context_len:
        pre=preprocess(context[a])
        train_context.append(pre[0])
        train_question.append(preprocess(questions[a])[0])
        x,y=[int(i) for i in span[a].split(' ')]
        q,w=[int(i) for i in get_span(span[a],pre[1]).split(' ')]
        short_answers.append(get_span(span[a],pre[1]))
#Creating vocabulary for embedding generation
vocabulary = dict()
vocabulary[""] = 0
count = 1
for para in context:
    words = preprocess(para)[0].split()
    for word in words:
        if word.isalpha() and word not in vocabulary.keys():
            vocabulary[word] = count
            count+=1
vocab_size = len(vocabulary.keys())
#Generating embeddings for context and questions
def to_vector(sentence, max_len):
    vector = np.zeros(max_len)
    tokens = tokenize(sentence)
    for ind in range(len(tokens)):
        if(ind>=max_len):
            break
        if tokens[ind] in vocabulary.keys():
            vector[ind] = vocabulary[tokens[ind]]
    return vector
    
context_vectors = np.asarray([to_vector(a, context_len) for a in train_context])
question_vectors = np.asarray([to_vector(a, question_len) for a in train_question])

start_point = np.zeros((len(short_answers), context_len))
answer_length = np.zeros(len(short_answers))
for ind in range(len(short_answers)):
    the_split = short_answers[ind].split()
    b = int(the_split[0])
    start_point[ind][b] = 1
    answer_length[ind] = int(the_split[1])-int(the_split[0])+1
#Train-test split
train_cv,test_cv,train_qv,test_qv,train_an,test_an=train_test_split(context_vectors,question_vectors,start_point,test_size=0.5)

# placeholders
input_sequence = Input((context_len,))
question = Input((question_len,))

# encoders
# embed the input sequence into a sequence of vectors
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,
                              output_dim=64))
input_encoder_m.add(Dropout(0.3))
# output: (samples, story_maxlen, embedding_dim)

# embed the input into a sequence of vectors of size query_maxlen
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,
                              output_dim=question_len))
input_encoder_c.add(Dropout(0.3))
# output: (samples, story_maxlen, query_maxlen)

# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=question_len))
question_encoder.add(Dropout(0.3))
# output: (samples, query_maxlen, embedding_dim)

# encode input sequence and questions (which are indices)
# to sequences of dense vectors
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

# compute a 'match' between the first input vector sequence
# and the question vector sequence
# shape: `(samples, story_maxlen, query_maxlen)`
match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)

# add the match matrix with the second input vector sequence
response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

# concatenate the match matrix with the question vector sequence
answer = concatenate([response, question_encoded])

# Usage of RNN
answer = LSTM(32)(answer)  # (samples, 32)

# one regularization layer -- more would probably be needed.
answer = Dropout(0.3)(answer)
answer = Dense(context_len)(answer)  # (samples, vocab_size)
# we output a probability distribution over the vocabulary
answer = Activation('softmax')(answer)

# build the final model
model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
              
model.fit([train_cv, train_qv], train_an,
           batch_size=32,
           epochs=200)
#To predict answer, given question, the trained model and context      
def predict_answer(c, q, m):
    act=(' ').join(word_tokenize(c)[:500])
    proc=preprocess(act)
    cv = np.asarray([to_vector(proc[0], context_len)])
    qv = np.asarray([to_vector(preprocess(q)[0], question_len)])
    ind =  np.argmax(m.predict([cv, qv]))#axis=1?
    new_ind=proc[1].index(ind)
    final=c.split(" ")[new_ind:]
    strr=""
    for i in final:
        if(i not in "[!#$%&\(),.:;<>?@[\\]{|}]"):
            strr+=i+' '
        else:
            strr.strip()
            break
    return strr
