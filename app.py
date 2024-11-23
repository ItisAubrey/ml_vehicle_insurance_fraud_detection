from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application=Flask(__name__)
app=application

##Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Get data from the form
        data = CustomData(
            Month=request.form.get('Month'),
            DayOfWeek=request.form.get('DayOfWeek'),
            Make=request.form.get('Make'),
            AccidentArea=request.form.get('AccidentArea'),
            DayOfWeekClaimed=request.form.get('DayOfWeekClaimed'),
            MonthClaimed=request.form.get('MonthClaimed'),
            Sex=request.form.get('Sex'),
            MaritalStatus=request.form.get('MaritalStatus'),
            Fault=request.form.get('Fault'),
            PolicyType=request.form.get('PolicyType'),
            VehicleCategory=request.form.get('VehicleCategory'),
            VehiclePrice=request.form.get('VehiclePrice'),
            Days_Policy_Accident=request.form.get('Days_Policy_Accident'),
            Days_Policy_Claim=request.form.get('Days_Policy_Claim'),
            PastNumberOfClaims=request.form.get('PastNumberOfClaims'),
            AgeOfVehicle=request.form.get('AgeOfVehicle'),
            AgeOfPolicyHolder=request.form.get('AgeOfPolicyHolder'),
            PoliceReportFiled=request.form.get('PoliceReportFiled'),
            WitnessPresent=request.form.get('WitnessPresent'),
            AgentType=request.form.get('AgentType'),
            NumberOfSuppliments=request.form.get('NumberOfSuppliments'),
            AddressChange_Claim=request.form.get('AddressChange_Claim'),
            NumberOfCars=request.form.get('NumberOfCars'),
            BasePolicy=request.form.get('BasePolicy'),
            WeekOfMonth=int(request.form.get('WeekOfMonth')),
            WeekOfMonthClaimed=int(request.form.get('WeekOfMonthClaimed')),
            Age=int(request.form.get('Age')),
            PolicyNumber=int(request.form.get('PolicyNumber')),
            RepNumber=int(request.form.get('RepNumber')),
            Deductible=int(request.form.get('Deductible')),
            DriverRating=int(request.form.get('DriverRating')),
            Year=int(request.form.get('Year'))
        )

        # Get data in DataFrame format
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # Instantiate the prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")

        # Make predictions
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        print("Prediction Result:", results)

        # Render the result in the HTML page
        return render_template('home.html', results=f"{results[0]:.2f}%")

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)


