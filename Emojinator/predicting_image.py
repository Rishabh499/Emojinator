import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image
from sklearn.externals import joblib
a=Image.open("abc1/3/1.jpg")
img=np.asarray(a.getdata()).reshape(a.size[1],a.size[0])
img=img.flatten()

img= pd.DataFrame(img).T
x=joblib.load('save.sav')

print (x.predict(img))
