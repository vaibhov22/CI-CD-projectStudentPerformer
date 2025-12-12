from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,Predict_pipeline

application = Flask(__name__)

app = application
## route for the home page
@app.route('/')
 
def index():
    return render_template('index.html')

@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
     gender = request.form.get("gender")
     race_ethnicity = request.form.get("race_ethnicity")
     parental_level = request.form.get("parental_level_of_education")
     lunch = request.form.get("lunch")
     test_prep = request.form.get("test_preparation_course")
     reading_score = request.form.get("reading_score")
     writing_score = request.form.get("writing_score")
 
     # Input validation
     if (not gender or not race_ethnicity or not parental_level or not lunch or
         not test_prep or not reading_score or not writing_score):
         return render_template('home.html', results="⚠ Please fill all fields before predicting!")
 
     data = CustomData(
         gender=gender,
         race_ethnicity=race_ethnicity,
         parental_level_of_education=parental_level,
         lunch=lunch,
         test_preparation_course=test_prep,
         reading_score=int(reading_score),
         writing_score=int(writing_score)
     )
 
     pred_df = data.get_data_as_frame()
     predict_pipeline = Predict_pipeline()
     results = predict_pipeline.predict(pred_df)
 
     return render_template('home.html', results=round(results[0], 2))


if __name__== "__main__":
   app.run(host="0.0.0.0")