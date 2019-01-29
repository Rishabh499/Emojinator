import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

import pandas as pd
file_name=[]
a=[]

df1=pd.DataFrame(np.nan,columns=np.arange(2501),index=np.arange(7200))
for dir,subdir,file in os.walk("project1/captured_images"):
    
    a.append(subdir)
    
    for names in file:
        if names.endswith(".jpg"):
                fullname=os.path.join(dir,names)
                file_name.append(fullname)



for i,file in enumerate(file_name):
    f1=str(file)
    
    
    
    img=Image.open(file)
    
    width,height=img.size
    
    value = np.asarray(img.getdata(), dtype=np.int).reshape((img.size[1], img.size[0]))
    
    value = value.flatten()/float(255)
    
    
    
    




    df = pd.DataFrame(value).T
    df.ix[0,2500]=int(f1[14])
    
    df1.ix[i,:]=np.array(df)

df1=df1.sample(frac=1).reset_index(drop=True)



with open('project1/images.csv', 'a') as dataset:
    df1.to_csv(dataset)



