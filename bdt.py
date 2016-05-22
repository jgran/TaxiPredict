import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import FloatType
from pyspark.sql.functions import lit
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.mllib.linalg import SparseVector
import pandas as pd
import numpy as np
import utils

#set spark configuration
conf = (SparkConf()
         .setMaster("local[16]")
         .setAppName("bdt")
         .set("spark.executor.memory", "1g")
         .set("spark.driver.memory", "6g")
        )
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

#read in dataframe created earlier
df = utils.get_df_by_name(sqlContext, 'mydf')

#don't need all the columns
df=df.select([c for c in df.columns if c in {"pc","grid_dist","short_dist","grid_short_ratio","grid_short_avg","timeofday","dayofweek","trip_time","total_notip","pick_grid","drop_grid","pick_traffic_index","drop_traffic_index", "pick_avg_speed", "drop_avg_speed", "pick_est_time", "drop_est_time"}])

#select features and standardize
rdd=df.rdd
features=rdd.map(lambda t: (t[0],t[1],t[2],t[5],t[6],t[9],t[10],t[11],t[12],t[15],t[16]))
standardizer = StandardScaler()
model = standardizer.fit(features)
features_transform = model.transform(features)                              

#select value we want to predict
#lab = rdd.map(lambda row: row[8])#time
lab = rdd.map(lambda row: row[7])#fare
transformedData = lab.zip(features_transform)
transformedData = transformedData.map(lambda row: LabeledPoint(row[0],[row[1]]))

#split into training and testing datasets
trainingData, testingData = transformedData.randomSplit([0.9,0.1],seed=1234)

#do the training and get predictions
model = GradientBoostedTrees.trainRegressor(trainingData, categoricalFeaturesInfo={}, numIterations=10)
predictions = model.predict(testingData.map(lambda x: x.features))
valuesAndPreds = testingData.map(lambda lp: lp.label).zip(predictions)
results = valuesAndPreds.toDF().toPandas()
results.columns = ['truth', 'pred']
results = results[results['truth'] > 0]
truth = np.array(results["truth"].tolist())
pred = np.array(results["pred"].tolist())
diff_fare = 100*(truth - pred)/truth

print 'mean = ' + str(diff_fare.mean())

#R-squared
metrics = RegressionMetrics(valuesAndPreds)
print("R-squared = %s" % metrics.r2)

#make some plots
plt.hist(diff_fare, bins=np.arange(-100, 100 + 10, 10))
plt.xlabel('percent difference in fare')
plt.savefig('diff_fare_rf.png')
plt.close()

plt.scatter(truth, pred, s=10, c='b')
plt.xlabel('True Fare [$]')
plt.ylabel('Predicted Fare [$]')
plt.xlim(0, 50)
plt.ylim(0, 50)
plt.savefig('fare_scatter_rf.png')
plt.close()

#save results for later
import pickle
with open('diff_fare_rf.pkl','w') as f:
    pickle.dump(diff_fare,f)
