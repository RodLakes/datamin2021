import math
#import tensorflow as tf
from collections import defaultdict
import numpy as np
from numpy import unique
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OrdinalEncoder
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pickle
import json
import urllib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

def train():
    data =pd.read_csv("weatherAUS.csv")

    df = data.copy()
    #datatime
    df['month'] = pd.to_datetime(df['Mes'])
    '''
    #Geocode by address
    locator = Nominatim(user_agent="myGeocoder")

    # 1 - convenient function to delay between geocoding calls
    geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
    # 2- - create location column
    df['location'] = df['address'].apply(geocode)
    print("step 2")
    # 3 - create longitude, laatitude and altitude from location column (returns tuple)
    df['point'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
    print("step 3")
    # 4 - split point column into latitude, longitude and altitude columns
    df[['latitude', 'longitude', 'altitude']] = pd.df(df['point'].tolist(), index=df.index)
    print("step 4")
    '''
    #Geocode by town (Singapore is so small that geocoding by addresses might not make much difference compared to geocoding to town)
    Location = [x for x in df['Location'].unique().tolist() 
                if type(x) == str]
    latitude = []
    longitude =  []
    for i in range(0, len(town)):
        # remove things that does not seem usefull here
        try:
            geolocator = Nominatim(user_agent="ny_explorer")
            loc = geolocator.geocode(town[i])
            latitude.append(loc.latitude)
            longitude.append(loc.longitude)
            #print('The geographical coordinate of location are {}, {}.'.format(loc.latitude, loc.longitude))
        except:
            # in the case the geolocator does not work, then add nan element to list
            # to keep the right size
            latitude.append(np.nan)
            longitude.append(np.nan)
    # create a df with the locatio, latitude and longitude
    df_ = pd.df({'Location':Location, 
                        'latitude': latitude,
                        'longitude':longitude})
    # merge on Location with rest_df to get the column 
    df = df.merge(df_, on='Location', how='left')

    Ffill = df.copy()
    Ffill=Ffill.fillna(method="ffill")
    df = Ffill.copy()

    LocationDic = {'Canberra': 1,'Sydney': 2,'Perth': 3, 'Darwin': 4,'Hobart': 5,'Brisbane': 6, 'Adelaide': 7, 'Bendigo': 8, 'Townsville': 9, 'AliceSprings': 10, 'MountGambier': 11, 'Launceston': 12, 'Ballarat': 13, 'Albany': 14, 'Albury': 15, 'MelbourneAirport': 16, 'PerthAirport': 17, 'Mildura': 18, 'SydneyAirport': 19, 'Nuriootpa': 20, 'Sale': 21, 'Watsonia ': 22, 'Tuggeranong': 23, 'Portland': 24, ' Woomera': 25, 'Cobar': 26, 'Cairns': 27, 'Wollongong': 28, 'GoldCoast': 29, 'WaggaWagga': 30, 'Penrith': 31, 'NorfolkIsland': 32, 'Newcastle': 33, 'SalmonGums': 34, 'CoffsHarbour': 35, 'Witchcliffe': 36, 'Richmond': 37, 'Dartmoor': 38, 'NorahHead': 39, 'BadgerysCreek': 40, 'MountGinini': 41, 'Moree': 42, 'Walpole': 43, 'PearceRAAF': 44, 'Williamtown': 45, 'Melbourne': 46, 'Nhil': 47, 'Katherine': 48, 'Uluru': 49,}
    WindGustDirDic = {'N' : 7, 'SE' : 1, 'E' : 10, 'SSE' : 5, 'NW' : 8, 'S' : 3, 'W' : 1, 'SW' : 2, 'NNE' : 15, 'NNW' : 13, 'ENE' : 14, 'ESE' : 9, 'NE' : 11, 'SSW' : 12, 'WNW' : 6, 'WSW' : 4,}
    RainTodayDic = {'No':1, 'Yes':0,}
    RainTomorrowDic = {'No':1, 'Yes':0,}
    WindDir9amDic = {'N' : 7, 'SE' : 1, 'E' : 10, 'SSE' : 5, 'NW' : 8, 'S' : 3, 'W' : 1, 'SW' : 2, 'NNE' : 15, 'NNW' : 13, 'ENE' : 14, 'ESE' : 9, 'NE' : 11, 'SSW' : 12, 'WNW' : 6, 'WSW' : 4,}
    WindDir3pmDic = {'N' : 7, 'SE' : 1, 'E' : 10, 'SSE' : 5, 'NW' : 8, 'S' : 3, 'W' : 1, 'SW' : 2, 'NNE' : 15, 'NNW' : 13, 'ENE' : 14, 'ESE' : 9, 'NE' : 11, 'SSW' : 12, 'WNW' : 6, 'WSW' : 4,}

    df['LocationCod'] = df['LocationCod'].replace(LocationDic, regex=True)
    df['WindGustDirCod'] = df['WindGustDirCod'].replace(WindGustDirDic, regex=True)
    df['RainTodayCod'] = df['RainTodayCod'].replace(RainTodayDic, regex=True)
    df['RainTomorrowCod'] = df['RainTomorrowCod'].replace(RainTomorrowDic, regex=True) 
    df['WindDir9amCod'] = df['WindDir9amCod'].replace(WindDir9amDic, regex=True)
    df['WindDir3pmCod'] = df['WindDir3pmCod'].replace(WindDir3pmDic, regex=True)     
 

    # drop some unnecessary columns
    df.drop(["Sunshine", "Evaporation", "Cloud3pm", "Cloud9am"], axis = 'columns', inplace=True)
    columns_desired=['Mes',
 'LocationCod',
 'MinTemp',
 'MaxTemp',
 'Rainfall',
 'WindGustDirCod',
 'WindGustSpeed',
 'WindDir9amCod',
 'WindDir3pmCod',
 'WindSpeed9am',
 'WindSpeed3pm',
 'Humidity9am',
 'Humidity3pm',
 'Pressure9am',
 'Pressure3pm',
 'Temp9am',
 'Temp3pm',
 'RainTodayCod',
 'RISK_MM']  

    X = df[columns_desired] # Features
    y = df["RainTomorrow"] # Target variable
   
   
    X=X.values
    y=y.values
    #splitting Train and Test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    x = X_train
    xx = X_test
    y = y_train
    yy = y_test

    RFC = RandomForestClassifier(n_estimators=3, random_state=1)
    RFC.fit(x, y)

    y_predition = RFC.predict(X_test)
  
  
  
  

    #standardization scaler - fit&transform on train, fit only on test

    s_scaler = StandardScaler()
    X_train = s_scaler.fit_transform(X_train.astype(np.float))
    X_test = s_scaler.transform(X_test.astype(np.float))

   #save model
    filename = 'model.sav'
    scalername = 'scaler.sav'
    pickle.dump(RFC, open(filename, 'wb'))
    pickle.dump(s_scaler, open(scalername, 'wb'))

    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, y_test)
    print(result)
    return result


