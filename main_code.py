import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import zipfile
import shutil
import random
import pandas as pd
import csv
import os
import json
from google.colab.patches import cv2_imshow
import math
import pandas as pd
import scipy
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='../source.jpg', help='path to image')
    opt = parser.parse_args()
    
if __name__ == "__main__":
  opt = parse_opt()
  org_img=cv2.imread(opt.source) 
def round_decimals_down(number:float, decimals:int=2):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor

def round_decimals_up(number:float, decimals:int=2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor

shape=(32,32,1)

with open('./thresh.json') as f:
  thresh=json.load(f)

cropped_data=[]
conf=[]
with open('../results.csv','r',newline='') as f:
  csvr=csv.reader(f)
  for row in csvr:
    cropped_data.append(row)
    conf.append(float(row[-1]))
selected_region=cropped_data[np.argmax(conf)]
x1=int(selected_region[1])
y1=int(selected_region[2])
x2=int(selected_region[3])
y2=int(selected_region[4])


cropped_img=org_img[y1:y2,x1:x2,:]
from google.colab.patches import cv2_imshow
print('Cropped image')
cv2_imshow(cropped_img)




pred_data={}
pred_temp={}

w=cropped_img.shape[1]
h=cropped_img.shape[0]
for pn in range(104):
  keras.backend.clear_session()
  model=keras.models.load_model('./models/skeleton_{}.hdf5'.format(pn))
  reg=thresh[str(pn)]
  min_x=round_decimals_down(reg[0])
  min_y=round_decimals_down(reg[1])
  max_x=round_decimals_up(reg[2])
  max_y=round_decimals_up(reg[3])
  part_img=cropped_img[int(min_y*h):int(max_y*h),int(min_x*w):int(max_x*w),:]

  img=cv2.resize(part_img,(shape[0],shape[1]))
  img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  img=img.astype(np.float32)
  img=img/255
  img=np.expand_dims(img,axis=2)
  img=np.expand_dims(img,axis=0)
  preds=model.predict(img)[0]
  pred_x=preds[0]
  pred_y=preds[1]

  x_=pred_x*part_img.shape[1]
  y_=pred_y*part_img.shape[0]

  ref_x=x_+int(min_x*w)
  ref_y=y_+int(min_y*h)

  pred_temp[pn]=preds
  pred_data[pn]=[ref_x,ref_y]

assign={29: 'ANS',
35: 'PNS',
41: 'UIT',
42: 'UIA',
44: 'A',
47: 'B',
49: 'Pog',
50: 'Gn',
51: 'Me',
56: 'LIT',
57: 'LIA',
66: 'Go',
69: 'Ar',
71: 'Co',
74: 'Pt',
75: 'S',
76: 'Or',
77: 'N',
}
painted_img=cropped_img.copy()
for key in pred_data:
  xx=int(pred_data[key][0])
  yy=int(pred_data[key][1])
  for i in range(-2,2):
      for j in range(-2,2):
        try:
          painted_img[int(yy+i),int(xx+j),:]=(0,0,255)
        except:
          pass
  if int(key) in assign:
      img=cv2.putText(painted_img, '{}({})'.format(key,assign[int(key)]), (xx+1,yy+1), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,255), 1, cv2.LINE_AA) 
  else:
      img=cv2.putText(painted_img, '{}'.format(key), (xx+1,yy+1), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,255), 1, cv2.LINE_AA) 

cv2_imshow(img)
cv2.imwrite('../output.bmp',img)


nose=[i for i in range(0,13)]
upper_lip=[i for i in range(13,18)]
lower_lip=[i for i in range(18,29)]
upper_centeral_tooth=[i for i in range(40,44)]
lower_centeral_tooth=[55,56,45,57]
maxilla=[i for i in range(29,41)]
maxilla.extend([43,44])
symphasis=[i for i in range(45,56)]
mandible=[i for i in range(62,74)]
c4=[i for i in range(78,83)]
c3=[i for i in range(83,88)]
c2=[i for i in range(88,93)]
c1=[i for i in range(93,102)]
other=[102,103]
lower_molar=[58,59]
upper_molar=[60,61]
anatomic_landmarks=[i for i in range(74,78)]

cats=['nose','upper_lip','lower_lip','upper_centeral_tooth','lower_centeral_tooth',
      'maxilla','symphasis','mandible','c4','c3','c2','c1','lower_molar','upper_molar','anatomic_landmarks','other']

categories={}
for ca in cats:
  for index in globals()['{}'.format(ca)]:
    categories[index]=ca

with open('../output.json','w') as f:
  output={}
  output['image_path']=img_name
  output['parts']=[]
  for key in pred_data:
    output['parts'].append({'type':categories[key],'points':[{'x':int(pred_data[key][0]),'y':int(pred_data[key][1]),'label':assign[key] if key in assign else key}],'unclose':True})
  json.dump(output,f)
