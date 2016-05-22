import matplotlib
matplotlib.use('Agg')
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import FloatType
from pyspark.sql.functions import lit
from optimize import func_to_optimize
import scipy.optimize as optimize
import pandas as pd
import numpy as np
import utils

#spark configuration
conf = (SparkConf()
         .setMaster("local[16]")
         .setAppName("local fare")
         .set("spark.executor.memory", "1g")
         .set("spark.driver.memory", "6g")
        )
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

#function to optimize to extract fare model parameters
def func_to_optimize(params, Dist, Time, Y):
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
    tot = 0.0
    for n in range(len(Dist)):
        dist = Dist[n]
        time = Time[n]
        y = a*5*dist + b*c + d
        tot = tot + (Y[n] - y)*(Y[n] - y)
    return tot

#function to call for each ride we want to predict
#finds closest historical taxi rides and fits fare model
def  make_prediction(event, df):
    event_timestamp, event_dayofweek, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, event_passenger_count = event[0], event[1], event[2], event[3], event[4], event[5], event[6]
    udf_diff_timeofday=udf(utils.diff_timeofday, IntegerType())
    udf_shortest_distance=udf(utils.shortest_distance, FloatType())
    df = df.withColumn("diff_timeofday", udf_diff_timeofday(df.pickup, lit(event_timestamp))).filter("`diff_timeofday` < 30")
    df = df.withColumn("event_sum_distance",
        udf_shortest_distance(df.pick_lat, df.pick_lon, lit(pickup_lat), lit(pickup_lon))+udf_shortest_distance(df.drop_lat, df.drop_lon, lit(dropoff_lat), lit(dropoff_lon))).filter("`event_sum_distance` < 2")
    df = df.sort('event_sum_distance')
    if df.count() < 10:
        return [0,0]
    a = pd.DataFrame(df.take(50))
    a.columns = df.columns

    speed_array = a.as_matrix(["avg_speed"])
    dist_sf_array = a.as_matrix(["dist_sf"])
    distance_array = a["trip_distance"].tolist()
    fare_array = a["total_notip"].tolist()
    time_array = a["trip_time_in_secs"].tolist()

    #set initial parameter values
    x0 = [0.5, 0.5, 3.0, 3.0]
    bnds = ((0.25, 0.75), (0.25, 0.75), (0.1,20), (0,10))
    
    #perform the fit
    res = optimize.minimize(func_to_optimize, x0, args=(distance_array, time_array, fare_array), method='TNC', bounds=bnds)
    grid_dist = utils.grid_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

    #get the predictions
    time_pred = utils.time_prediction(speed_array.mean(), grid_dist, dist_sf_array.mean())
    fare_pred = utils.fare_prediction(res.x[0], grid_dist, dist_sf_array.mean(), res.x[1], res.x[2], res.x[3])
    if res.success == True:
        return [fare_pred, time_pred]
    else:
        return [0,0]


#read in dataframe and cache it
df1 = utils.get_df(sqlContext, 1, 1)
df1.cache()

#get random sample for prediction
test_sample=df1.sample(False, 0.1, seed=42).limit(500).toPandas()
test_sample.columns = df1.columns
test_fare = test_sample["total_notip"].tolist()
test_time = test_sample["trip_time_in_secs"].tolist()
pred_fare = []
pred_time = []

#get prediction for each event
for index, row in test_sample.iterrows():
    print 'Processing event '+str(index)
    event = [row['pickup'],utils.dayofweek(row['pickup']),row['pick_lat'],row['pick_lon'],row['drop_lat'],row['drop_lon'],row['pc']]
    prediction = make_prediction(event, df1) 
    if prediction[0] < 2 or prediction[0] > 200:
        continue
    if prediction[1] < 30 or prediction[1] > 5000:
        continue
    pred_fare.append(prediction[0])
    pred_time.append(prediction[1])

print test_fare
print pred_fare
print " "
print test_time
print pred_time
diff_fare = 100*(np.array(test_fare) - np.array(pred_fare))/np.array(test_fare)
diff_time = 100*(np.array(test_time) - np.array(pred_time))/np.array(test_time)
print diff_fare.mean()

#make plots 
from matplotlib import pyplot as plt
plt.hist(diff_fare, bins=np.arange(-100, 100 + 20, 20))
plt.xlabel('percent difference in fare')
plt.savefig('diff_fare_neighbor.png')
plt.close()

plt.hist(diff_time, bins=np.arange(-100, 100 + 20, 20))
plt.xlabel('percent difference in time')
plt.savefig('diff_time_neighbor.png')
plt.close()

#save for later
import pickle
with open('diff_fare_neighbor.pkl','w') as f:
    pickle.dump(diff_fare,f)
with open('diff_time_neighbor.pkl','w') as f:
    pickle.dump(diff_time,f)
