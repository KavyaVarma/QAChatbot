from flask import Flask, render_template, request
from keras.models import load_model
from models import question_processed
from models import DR_MODEL
from tensorflow import get_default_graph
graph=get_default_graph()
load=load_model('models/learned.h5')

import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
lemmatizer=WordNetLemmatizer()
import string
import json
import re
stop_words = set(stopwords.words('english'))
punctuations=string.punctuation

vocabulary = json.load(open('models/vocab.json','r'))
#print(len(vocabulary))
#To tokenize given sentence into words
def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
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
#To predict answer, given question, the trained model and context 
def predict_answer(c, q, m):
    act=(' ').join(word_tokenize(c)[:500])
    proc=preprocess(act)
    cv = np.asarray([to_vector(proc[0], 500)])
    qv = np.asarray([to_vector(preprocess(q)[0], 36)])
    ind =  np.argmax(m.predict([cv, qv]))#axis=1?
    new_ind=proc[1].index(ind)
    final=word_tokenize((' ').join(act.split(" ")[new_ind:]))
    strr=""
    for i in final:
        if(i not in "[!#$%&\(),.:;<>?@[\\]{|}]"):
            strr+=i+' '
        else:
            strr.strip()
            break
    return strr

#Flask functions, used to render the output on the UI (webpage)
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html',result={})

@app.route('/result',methods = ['POST', 'GET'])
def result():
	if request.method == 'POST':
		result = dict(request.form)
		#print(result["Document"])
		paragraphs = result["Document"][0].split('\n')
		while("" in paragraphs) : 
			paragraphs.remove("") 
		drm = DR_MODEL.DR_MODEL(paragraphs,True,True)
		userQuery = result["Question"][0]
		pq = question_processed.ProcessedQuestion(userQuery,True,True,True)
		drresponse = drm.query(pq)
		try:
			global graph
			with graph.as_default():
				result["Answer Predicted by LSTM Model"]=predict_answer(result["Document"][0],result["Question"][0],load)
		except:
			result["Answer Predicted by LSTM Model"]="Oops! The answer was not found"
		result["Answer Predicted by DR Based Model"]=drresponse
		return render_template("home.html",result = result)

if __name__ == '__main__':
	app.run(debug = True)
