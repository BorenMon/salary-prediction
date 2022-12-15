# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# Loading the saved model
loaded_model = pickle.load(open('C:/Users/Boren/Desktop/phyton/trained-model.sav', 'rb'))

input_data = (20, 2, 0, 13, 0, 4, 4, 2, 0, 0, 0, 40, 17)

input_data_np = np.asarray(input_data).reshape(1, -1)

prediction = loaded_model.predict(input_data_np)

if(prediction[0] == 0):
  print('<=50K')
else:
  print('>50K')
