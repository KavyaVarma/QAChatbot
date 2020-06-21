from keras.models import load_model
from models import question_processed
from models import DR_MODEL
import warnings
warnings.filterwarnings('ignore')
load=load_model('models/learned.h5')

import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
lemmatizer=WordNetLemmatizer()
import string
import json
import re
import pandas as pd
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

#The main function
#If option 1 is selected, then a context is loaded and the user gets to ask a question to get an answer
#If option 0 is selected, then the program runs on 5 different sets of context and question to generate answers
if __name__ == '__main__':
	result = {}
	print('\n\n\n\nNOTE : LSTM considers only first 500 words of the context and first 36 words of the question. But DRM considers all.')
	custom=int(input('\n\nDo you want to run it on existing test cases (0) or on a custom input(1) ?(0/1)\t'))
	if(custom==1):
		with open('testing_doc.txt') as f:
			result["Document"]= f.read()
		print("\n\033[1;32;40m Context :\033[0m \n",result["Document"],"\n\n")
		choice=1
		while(choice):
			paragraphs = result["Document"].split('\n')
			while("" in paragraphs) : 
				paragraphs.remove("") 
			drm = DR_MODEL.DR_MODEL(paragraphs,True,True)
			result["Question"]=input("\nEnter the question : ")
			userQuery = result["Question"]
			pq = question_processed.ProcessedQuestion(userQuery,True,True,True)
			drresponse = drm.query(pq)
			try:
				result["Answer Predicted by LSTM Model"]=predict_answer(result["Document"],result["Question"],load)
			except:
				result["Answer Predicted by LSTM Model"]="Oops! The answer was not found"
			result["Answer Predicted by DR Based Model"]=drresponse
			print("\n\033[2;37;40m Answer Predicted by LSTM Model :\033[0m",result["Answer Predicted by LSTM Model"],sep='\n')
			print("\n\033[2;37;40m Answer Predicted by DR Based Model :\033[0m",result["Answer Predicted by DR Based Model"],sep='\n')
			choice=int(input("\nDo you want to ask another question? 0/1 : "))
	else:
		df=pd.read_csv("tests.csv")
		leng=df.shape[0]
		for i in range(leng):
			paragraphs=str(df['Document'][i]).strip()
			print("\n\033[1;32;40m Context :\033[0m \n",paragraphs,"\n")
			drm = DR_MODEL.DR_MODEL([paragraphs],True,True)
			result["Question"]=str(df['Question'][i]).strip()
			print("\033[1;31;40m Question :\033[0m\n",result["Question"],"\n")
			userQuery = result["Question"]
			pq = question_processed.ProcessedQuestion(userQuery,True,True,True)
			drresponse = drm.query(pq)
			try:
				result["Answer Predicted by LSTM Model"]=predict_answer(str(df['Document'][i]).strip(),result["Question"],load)
			except:
				result["Answer Predicted by LSTM Model"]="Oops! The answer was not found"
			result["Answer Predicted by DR Based Model"]=drresponse
			print("\n\033[2;37;40m Answer Predicted by LSTM Model :\033[0m",result["Answer Predicted by LSTM Model"],sep='\n')
			print("\n\033[2;37;40m Answer Predicted by DR Based Model :\033[0m",result["Answer Predicted by DR Based Model"],sep='\n')
