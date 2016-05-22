from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark import SparkConf
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StringType
from pyspark.sql.functions import lit
from utils import *

conf = (SparkConf()
         .setMaster("local[16]")
         .setAppName("My app")
         .set("spark.executor.memory", "1g")
         .set("spark.driver.memory", "8g")
        )
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

#read in csv files to dataframe then combine
df1_temp = sqlContext.read.load('file:/hadoop/cms/store/user/jgran/taxi/trip_data_reduced.csv',format='com.databricks.spark.csv', header='true',inferSchema='true')
df2_temp = sqlContext.read.load('file:/hadoop/cms/store/user/jgran/taxi/trip_fare_reduced.csv',format='com.databricks.spark.csv', header='true',inferSchema='true')

oldColumns = df1_temp.schema.names
newColumns = [x.strip() for x in oldColumns]
df1 = reduce(lambda df1_temp, idx: df1_temp.withColumnRenamed(oldColumns[idx], newColumns[idx]), xrange(len(oldColumns)), df1_temp)

oldColumns = df2_temp.schema.names
newColumns = [x.strip() for x in oldColumns]
df2 = reduce(lambda df2_temp, idx: df2_temp.withColumnRenamed(oldColumns[idx], newColumns[idx]), xrange(len(oldColumns)), df2_temp)

df1=df1.select([c for c in df1.columns if c not in {"hack_license","vendor_id","rate_code","store_and_fwd_flag","dropoff_datetime"}])
df2=df2.select([c for c in df2.columns if c not in {"hack_license","vendor_id","payment_type"}])
df1 = df1.withColumnRenamed("pickup_datetime", "pickup")
df2 = df2.withColumnRenamed("pickup_datetime", "pickup2")
df12=df1.join(df2, (df1.medallion==df2.medallion)&(df1.pickup==df2.pickup2), "outer")
df3=df12.select([c for c in df12.columns if c not in {"medallion","pickup2"}])

#instantiate User Defined Functions used to add new features to dataframe
udf_shortest_distance=udf(shortest_distance, FloatType())
udf_distance_sf=udf(distance_sf, FloatType())
udf_grid_distance=udf(grid_distance, FloatType())
udf_grid_avg_speed=udf(grid_avg_speed, FloatType())
udf_grid_value=udf(grid_value, IntegerType())
udf_traffic_index=udf(traffic_index, FloatType())
udf_sec_to_min=udf(sec_to_min, IntegerType())
udf_avg=udf(avg, FloatType())
udf_divide=udf(divide, IntegerType())
udf_dayofweek=udf(dayofweek, IntegerType())
udf_timeofday=udf(timeofday, IntegerType())

#pick some shorted names
df3 = df3.withColumnRenamed("passenger_count", "pc")
df3 = df3.withColumnRenamed("pickup_latitude", "pick_lat")
df3 = df3.withColumnRenamed("pickup_longitude", "pick_lon")
df3 = df3.withColumnRenamed("dropoff_latitude", "drop_lat")
df3 = df3.withColumnRenamed("dropoff_longitude", "drop_lon")
df3 = df3.withColumnRenamed("fare_amount", "fare")
df3 = df3.withColumnRenamed("tolls_amount", "tolls")
df3 = df3.withColumnRenamed("tip_amount", "tip")
df3 = df3.withColumnRenamed("total_amount", "total")
df3 = df3.na.fill(0)

#do some cleaning to remove bogus data
df3 = df3.filter("`pick_lon` != 0")
df3 = df3.filter("`pick_lat` != 0")
df3 = df3.filter("`drop_lon` != 0")
df3 = df3.filter("`drop_lat` != 0")
df3 = df3.filter("`trip_time_in_secs` != 0")
df3 = df3.filter("`pick_lon` > -74.05")
df3 = df3.filter("`pick_lon` < -73.7")
df3 = df3.filter("`pick_lat` > 40.5")
df3 = df3.filter("`pick_lat` < 40.9")
df3 = df3.filter("`drop_lon` > -74.05")
df3 = df3.filter("`drop_lon` < -73.7")
df3 = df3.filter("`drop_lat` > 40.5")
df3 = df3.filter("`drop_lat` < 40.9")

#add some new features
df3 = df3.withColumn("dayofweek", udf_dayofweek("pickup"))
df3 = df3.withColumn("timeofday", udf_timeofday("pickup"))
df3 = df3.withColumn("grid_dist", udf_grid_distance(df3.pick_lat, df3.pick_lon, df3.drop_lat, df3.drop_lon))
df3 = df3.withColumn("short_dist", udf_shortest_distance(df3.pick_lat, df3.pick_lon, df3.drop_lat, df3.drop_lon))
df3 = df3.withColumn("grid_short_ratio", df3.grid_dist/df3.short_dist)
df3 = df3.withColumn("grid_short_avg", udf_avg(df3.grid_dist, df3.short_dist, lit(2.0)))
df3 = df3.withColumn("dist_sf", udf_distance_sf(df3.pick_lat, df3.pick_lon, df3.drop_lat, df3.drop_lon, df3.trip_distance))
df3 = df3.withColumn("avg_speed", df3.trip_distance/df3.trip_time_in_secs)
df3 = df3.withColumn("total_notip", df3.total - df3.tip)
df3 = df3.withColumn("trip_time", udf_sec_to_min(df3.trip_time_in_secs))
df3 = df3.withColumn("pick_grid", udf_grid_value(df3.pick_lat, df3.pick_lon))
df3 = df3.withColumn("drop_grid", udf_grid_value(df3.drop_lat, df3.drop_lon))
df3 = df3.withColumn("pick_traffic_index", udf_traffic_index(df3.pick_grid, lit(True)));
df3 = df3.withColumn("drop_traffic_index", udf_traffic_index(df3.pick_grid, lit(False)));
df3 = df3.withColumn("pick_avg_speed", udf_grid_avg_speed(df3.pick_grid))
df3 = df3.withColumn("drop_avg_speed", udf_grid_avg_speed(df3.drop_grid))
df3 = df3.filter("`pick_avg_speed` > 0")
df3 = df3.filter("`drop_avg_speed` > 0")
df3 = df3.na.drop()
df3 = df3.withColumn("pick_est_time", udf_divide(df3.grid_short_avg/df3.pick_avg_speed, lit(60.0)))
df3 = df3.withColumn("drop_est_time", udf_divide(df3.grid_short_avg/df3.drop_avg_speed, lit(60.0)))

#save dataframe for reading later
storage_dir = '/hadoop/cms/store/user/jgran/taxi/saved_dataframes/'
name = 'mydf'
df_writer = DataFrameWriter(df3)
df_writer.parquet(storage_dir+name)
