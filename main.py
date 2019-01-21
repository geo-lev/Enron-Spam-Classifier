import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.sparse as sp
import LogisticRegression

NEWLINE = '\n'
SKIP_FILES = {'cmds'}

def read_files(path):
    for root,dir_names,file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root,path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path =os.path.join(root,file_name)
                if os.path.isfile(file_path):
                    past_header,lines= False,[]
                    f = open(file_path,encoding='latin-1')
                    for line in f :
                        if past_header:
                            lines.append(line)
                        elif line == NEWLINE:
                            past_header = True
                    f.close()
                    content = NEWLINE.join(lines)
                    yield file_path , content
                    
def build_data_frame(path , labeling):
    rows= []
    index = []
    for file_name , text in read_files(path):
        rows.append({'text' : text , 'class' : labeling})
        index.append(file_name)
    df = pd.DataFrame(rows,index=index)
    return df

HAM = 0
SPAM = 1

SOURCES = [('beck-s', HAM),
           ('farmer-d' , HAM),
           ('kaminski-v', HAM),
           ('kitchen-l', HAM),
           ('lokay-m', HAM),
           ('williams-w3', HAM),
           ('BG' , SPAM),
           ('GP' , SPAM),
           ('SH' , SPAM)]

df = pd.DataFrame({'text': [] ,'class': []})
for path, labeling in SOURCES:
    df = df.append(build_data_frame(path , labeling))
df = df.reindex(np.random.permutation(df.index))

from sklearn.feature_extraction.text import CountVectorizer
word_count= CountVectorizer(min_df=0.1)
X = word_count.fit_transform(df['text'])
X = sp.csc_matrix.todense(X)
y=df['class']
y = y[:,np.newaxis]

clf = LogisticRegression.LogisticRegression()
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
i=0
for train_index,test_index in skf.split(X,y):
    i += 1
    print('Training ',i,'fold of ',skf.n_splits) 
    X_train,X_test = X[train_index],X[test_index]
    y_train , y_test = y[train_index] , y[test_index]
    weight , log_likelihood = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test , weight)
    print('Accuracy is ' , (y_test==y_pred).mean())
    from sklearn.metrics import f1_score
    print('f1_micro is ', f1_score(y_test,y_pred))
    plt.plot(log_likelihood)
