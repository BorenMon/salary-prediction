# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 23:43:34 2022

@author: Boren
"""

import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('trained-model.sav', 'rb'))

# Creating a function for prediction
def salary_prediction(input_data):

    input_data_np = np.asarray(input_data).reshape(1, -1)

    prediction = loaded_model.predict(input_data_np)
    
    sex = 'His ' if input_data[8] == 0 else 'Her '

    if(prediction[0] == 0):
      return sex + 'salary is less than or equal to $50K.'
    else:
      return sex + 'Salary is greater than $50K.'
  
def main():
    
    # Giving Title
    st.title('Salary Prediction Web App')
    
    # Getting the input data from user
    
    age = st.number_input(label='Age', min_value=0, step=1, value=18)
    st.write(age)
    
    
    
    workclass_ops = ['State-gov', 'Self-emp-not-inc', 
                    'Private', 'Federal-gov', 'Local-gov',
                    'Self-emp-inc', 'Without-pay', 
                    'Never-worked']   
    workclass = st.selectbox('Workclass', options=workclass_ops) 
    workclass_enc = workclass_ops.index(workclass)
    st.write(workclass_enc)
    
    
    
    education_ops = ['Bachelors', 'HS-grad', '11th', 
                    'Masters', '9th', 'Some-college', 
                    'Assoc-acdm','Assoc-voc', '7th-8th', 
                    'Doctorate', 'Prof-school', '5th-6th', 
                    '10th', '1st-4th', 'Preschool', '12th']
    education = st.selectbox('Education', options=education_ops)    
    education_enc = education_ops.index(education)
    st.write(education_enc)
    
    

    educationNum = st.selectbox('Education Number', options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    st.write(educationNum)
    
    
    
    materialStatus_ops = ['Never-married',
                        'Married-civ-spouse',
                        'Divorced',
                        'Married-spouse-absent',
                        'Separated' 'Married-AF-spouse' 'Widowed']
    materialStatus = st.selectbox('Material Status', options=materialStatus_ops)
    materialStatus_enc = materialStatus_ops.index(materialStatus)
    st.write(materialStatus_enc)
    
    
    
    occupation_ops = ['Adm-clerical', 'Exec-managerial',
                      'Handlers-cleaners', 'Prof-specialty',
                    'Other-service', 'Sales', 'Craft-repair', 'Transport-moving',
                    'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Protective-serv',
                    'Armed-Forces', 'Priv-house-serv']
    occupation = st.selectbox('Occupation', options=occupation_ops)
    occupation_enc = occupation_ops.index(occupation)
    st.write(occupation_enc)
    
    
    
    relationship_ops = ['Not-in-family', 'Husband',
                        'Wife', 'Own-child', 'Unmarried',
                        'Other-relative']
    relationship = st.selectbox('Relationship', options=relationship_ops)
    relationship_enc = relationship_ops.index(relationship)
    st.write(relationship_enc)
    
    
    
    race_ops = ['White', 'Black', 'Asian-Pac-Islander',
                'Amer-Indian-Eskimo', 'Other']
    race = st.selectbox('Race', options=race_ops)
    race_enc = race_ops.index(race)
    st.write(race_enc)
    
    
    
    sex_ops = ['Male', 'Female']
    sex = st.selectbox('Sex', options=sex_ops)
    sex_enc = sex_ops.index(sex)
    st.write(sex_enc)
    
    
    hoursPerWeek = st.number_input(label='Hours Per Week', min_value=0.0, value=0.0)
    st.write(hoursPerWeek)
    
    # Code for prediction
    result = ''
    
    # Creating a button for prediction
    if st.button('Predict Annual Salary'):
        result = salary_prediction([age, workclass_enc, education_enc, 
                                   educationNum, materialStatus_enc, occupation_enc, 
                                   relationship_enc, race_enc, sex_enc, hoursPerWeek])
        
    st.success(result)
    

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    