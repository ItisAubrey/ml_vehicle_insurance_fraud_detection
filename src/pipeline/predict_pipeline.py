import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object, label_encode_binary_columns, map_ordinal_features, change_dtype_to_string, \
    one_hot_encode
import os
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            selected_features_path = os.path.join("artifacts", "selected_features.pickle")
            model_path = os.path.join("artifacts", "model.pkl")
            print("Before Loading")
            model = load_object(file_path=model_path)
            selected_features = load_object(file_path=selected_features_path)
            print("After Loading")
            binary_columns=['AccidentArea', 'Sex', 'Fault', 'PoliceReportFiled', 'WitnessPresent', 'AgentType']
            data_scaled = label_encode_binary_columns(features, binary_columns)
            # Ordinal Feature Mapping
            ordinal_mappings = {
                'VehiclePrice': {'more than 69000': 1, '20000 to 29000': 0, '30000 to 39000': 0, 'less than 20000': 1,
                                 '40000 to 59000': 1, '60000 to 69000': 0},
                'AgeOfVehicle': {'new': 2, '2 years': 0, '3 years': 2, '4 years': 2, '5 years': 1, '6 years': 1,
                                 '7 years': 0, 'more than 7': 0},
                'BasePolicy': {'Liability': 0, 'Collision': 1, 'All Perils': 2}
            }
            data_scaled = map_ordinal_features(data_scaled, ordinal_mappings)
            dtype_change_string = ['RepNumber', 'Deductible', 'Year']
            data_scaled = change_dtype_to_string(data_scaled, dtype_change_string)
            onehot_encoding_columns = ['Make', 'MonthClaimed', 'MaritalStatus', 'PolicyType', 'VehicleCategory',
                                       'RepNumber', 'Deductible', 'Days_Policy_Accident', 'Days_Policy_Claim',
                                       'PastNumberOfClaims', 'AgeOfPolicyHolder', 'NumberOfSuppliments',
                                       'AddressChange_Claim', 'NumberOfCars', 'Year']
            data_scaled = one_hot_encode(data_scaled, onehot_encoding_columns)
            # Ensure all expected columns are present in the prediction data
            missing_cols = [col for col in selected_features if col not in data_scaled.columns]
            for col in missing_cols:
                data_scaled[col] = 0  # Set missing dummy variables to zero

            # Select columns in the correct order
            data_scaled = data_scaled[selected_features]
            preds = model.predict_proba(data_scaled)
            # Probability of the positive class (class 1)
            positive_class_prob = preds[:, 1] * 100  # Multiply by 100 to convert to percentage
            return positive_class_prob

        except Exception as e:
            raise CustomException(e, sys)



import pandas as pd
import sys

class CustomData:
    def __init__(self,
                 Month: str,
                 DayOfWeek: str,
                 Make: str,
                 AccidentArea: str,
                 DayOfWeekClaimed: str,
                 MonthClaimed: str,
                 Sex: str,
                 MaritalStatus: str,
                 Fault: str,
                 PolicyType: str,
                 VehicleCategory: str,
                 VehiclePrice: str,
                 Days_Policy_Accident: str,
                 Days_Policy_Claim: str,
                 PastNumberOfClaims: str,
                 AgeOfVehicle: str,
                 AgeOfPolicyHolder: str,
                 PoliceReportFiled: str,
                 WitnessPresent: str,
                 AgentType: str,
                 NumberOfSuppliments: str,
                 AddressChange_Claim: str,
                 NumberOfCars: str,
                 BasePolicy: str,
                 WeekOfMonth: int,
                 WeekOfMonthClaimed: int,
                 Age: int,
                 PolicyNumber: int,
                 RepNumber: int,
                 Deductible: int,
                 DriverRating: int,
                 Year: int):
        # Initializing all the attributes with values received from the form
        self.Month = Month
        self.DayOfWeek = DayOfWeek
        self.Make = Make
        self.AccidentArea = AccidentArea
        self.DayOfWeekClaimed = DayOfWeekClaimed
        self.MonthClaimed = MonthClaimed
        self.Sex = Sex
        self.MaritalStatus = MaritalStatus
        self.Fault = Fault
        self.PolicyType = PolicyType
        self.VehicleCategory = VehicleCategory
        self.VehiclePrice = VehiclePrice
        self.Days_Policy_Accident = Days_Policy_Accident
        self.Days_Policy_Claim = Days_Policy_Claim
        self.PastNumberOfClaims = PastNumberOfClaims
        self.AgeOfVehicle = AgeOfVehicle
        self.AgeOfPolicyHolder = AgeOfPolicyHolder
        self.PoliceReportFiled = PoliceReportFiled
        self.WitnessPresent = WitnessPresent
        self.AgentType = AgentType
        self.NumberOfSuppliments = NumberOfSuppliments
        self.AddressChange_Claim = AddressChange_Claim
        self.NumberOfCars = NumberOfCars
        self.BasePolicy = BasePolicy
        self.WeekOfMonth = WeekOfMonth
        self.WeekOfMonthClaimed = WeekOfMonthClaimed
        self.Age = Age
        self.PolicyNumber = PolicyNumber
        self.RepNumber = RepNumber
        self.Deductible = Deductible
        self.DriverRating = DriverRating
        self.Year = Year

    def get_data_as_data_frame(self):
        try:
            # Construct a dictionary to match the required structure for the prediction
            custom_data_input_dict = {
                "Month": [self.Month],
                "DayOfWeek": [self.DayOfWeek],
                "Make": [self.Make],
                "AccidentArea": [self.AccidentArea],
                "DayOfWeekClaimed": [self.DayOfWeekClaimed],
                "MonthClaimed": [self.MonthClaimed],
                "Sex": [self.Sex],
                "MaritalStatus": [self.MaritalStatus],
                "Fault": [self.Fault],
                "PolicyType": [self.PolicyType],
                "VehicleCategory": [self.VehicleCategory],
                "VehiclePrice": [self.VehiclePrice],
                "Days_Policy_Accident": [self.Days_Policy_Accident],
                "Days_Policy_Claim": [self.Days_Policy_Claim],
                "PastNumberOfClaims": [self.PastNumberOfClaims],
                "AgeOfVehicle": [self.AgeOfVehicle],
                "AgeOfPolicyHolder": [self.AgeOfPolicyHolder],
                "PoliceReportFiled": [self.PoliceReportFiled],
                "WitnessPresent": [self.WitnessPresent],
                "AgentType": [self.AgentType],
                "NumberOfSuppliments": [self.NumberOfSuppliments],
                "AddressChange_Claim": [self.AddressChange_Claim],
                "NumberOfCars": [self.NumberOfCars],
                "BasePolicy": [self.BasePolicy],
                "WeekOfMonth": [self.WeekOfMonth],
                "WeekOfMonthClaimed": [self.WeekOfMonthClaimed],
                "Age": [self.Age],
                "PolicyNumber": [self.PolicyNumber],
                "RepNumber": [self.RepNumber],
                "Deductible": [self.Deductible],
                "DriverRating": [self.DriverRating],
                "Year": [self.Year]
            }

            # Return the data as a DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

