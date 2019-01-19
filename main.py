import os 
import numpy as np 
import pandas as pd 

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
        rows.append({'text' : text , 'label' : labeling})
        index.append(file_name)
    df = pd.DataFrame(rows,index=index)
    return df


HAM = 'ham'
SPAM = 'spam'

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
