import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import datetime
import re
import pickle
from pyspark.sql import DataFrameWriter

def km_to_mi(d_km):
    return d_km*0.621371

def sec_to_min(sec):
    return int(sec/60)

def avg(x, y, n):
    return (x + y)/n

def divide(a, b):
    return int(a/b)

#haversine distance
def shortest_distance(lat1, lon1, lat2, lon2):
    radius = 6371
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    if d == 0:
        return float('NaN')
    else:
        return km_to_mi(d)

def grid_distance(lat1, lon1, lat2, lon2):
    return (shortest_distance(lat1, 0, lat2, 0) + shortest_distance(0, lon1, 0, lon2))

def distance_sf(lat1, lon1, lat2, lon2, dist):
    return (dist/grid_distance(lat1, lon1, lat2, lon2))

def diff_distance(lat1, lon1, lat2, lon2):
    return abs(lat1-lat2) + abs(lon1-lon2)

def to_datetime(time_string):
    pieces = re.findall(r"[\w']+", time_string)
    p = [int(s) for s in pieces]
    return datetime.datetime(p[0],p[1],p[2],p[3],p[4],p[5]) 

def dayofweek(time_string):
    return to_datetime(time_string).weekday()

def timeofday(time_string):
    t = to_datetime(time_string)
    return t.hour

def diff_timeofday(time1, time2):
    t1 = to_datetime(time1)
    t2 = to_datetime(time2)
    d1 = datetime.datetime(1,1,1,t1.hour,t1.minute,t1.second)
    d2 = datetime.datetime(1,1,1,t2.hour,t2.minute,t2.second)
    delta1 = (d1 - d2).seconds/60
    delta2 = (d2 - d1).seconds/60
    return min(delta1, delta2)

def fare_prediction(rate_fast, grid_dist, dist_sf, rate_slow, time_slow, offset):
    return rate_fast*5*grid_dist*dist_sf + rate_slow*time_slow + offset

def time_prediction(speed, grid_dist, dist_sf):
    return grid_dist*dist_sf/speed

def save_df(df, day_of_week, passenger_count):
    storage_dir = '/hadoop/cms/store/user/jgran/taxi/saved_dataframes/'
    name = 'df_'+day_of_week+'_'+passenger_count
    #df_writer = DataFrameWriter(df.coalesce(50))
    df_writer = DataFrameWriter(df)
    df_writer.parquet(storage_dir+name)
    
def get_df(myContext, day_of_week, passenger_count):
    storage_dir = '/hadoop/cms/store/user/jgran/taxi/saved_dataframes/'
    name = 'df_'+str(day_of_week)+'_'+str(passenger_count)
    return myContext.read.load('file:'+storage_dir+name,format='parquet', header='true',inferSchema='true')

def get_df_by_name(myContext, name):
    storage_dir = '/hadoop/cms/store/user/jgran/taxi/saved_dataframes/'
    return myContext.read.load('file:'+storage_dir+name,format='parquet', header='true',inferSchema='true')

def grid_value(lat, lon):
    lat_min = 40.5
    lat_max = 40.9
    lon_min = -74.05
    lon_max = -73.7
    lat_increment = (lat_max - lat_min)/10.0
    lon_increment = (lon_max - lon_min)/10.0
    lat_grid = -1
    lon_grid = -1
    for i in range(10):
        if (lat_min + (i+1)*lat_increment) > lat:
            lat_grid = i
            break
    for i in range(10):
        if (lon_min + (i+1)*lon_increment) > lon:
            lon_grid = i
            break
    return int(str(lat_grid)+str(lon_grid))

def plot_neighborhood(p_lon, d_lon, p_lat, d_lat, event):
    for i in range(len(p_lon)):
        if i == 0:
            plt.plot([p_lon[i], d_lon[i]], [p_lat[i], d_lat[i]], 'k-', lw=1, c='g', label='neighboring rides')
        else:
            plt.plot([p_lon[i], d_lon[i]], [p_lat[i], d_lat[i]], 'k-', lw=1, c='g')
    plt.plot([event[0], event[1]], [event[2], event[3]], 'k-', lw=5, c='b', label='ride to predict')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(loc='upper left', fancybox=True, fontsize=12)
    plt.savefig('neighborhood.png')
    plt.close()

#dictionary values extracted in interative spark session
def traffic_index(grid, do_pick):
    mean = -1
    std = -1
    traffic_dict = {}
    if do_pick:
        mean = 12523.1
        std = 49673.3
        traffic_dict = {11: 6, 12: 4, 15: 1, 20: 16, 21: 32, 22: 59, 23: 12, 24: 5, 25: 2, 27: 2, 28: 1, 29: 1, 30: 85, 31: 180, 32: 385, 33: 22, 34: 10, 35: 6, 36: 97, 37: 20813, 38: 8, 39: 2, 40: 54, 41: 9299, 42: 4430, 43: 795, 44: 69, 45: 72, 46: 196, 47: 351, 48: 7, 49: 3, 50: 6663, 51: 202387, 52: 18857, 53: 2287, 54: 484, 55: 901, 56: 342, 57: 46, 58: 9, 59: 16, 60: 92, 61: 277465, 62: 273676, 63: 10900, 64: 3916, 65: 23431, 66: 156, 67: 17, 68: 3, 69: 2, 70: 10, 71: 6748, 72: 86510, 73: 7309, 74: 52, 75: 10, 76: 9, 77: 3, 78: 1, 79: 2, 80: 4, 81: 1, 82: 1944, 83: 2363, 84: 149, 85: 70, 86: 20, 90: 1, 91: 2, 92: 9, 93: 246, 94: 120, 95: 16, 96: 7, 97: 1}
    else:
        mean = 11479.5
        std = 45618.6
        traffic_dict = {11: 41, 12: 113, 13: 5, 14: 1, 15: 20, 16: 3, 19: 1, 20: 270, 21: 402, 22: 739, 23: 217, 24: 31, 25: 6, 26: 19, 27: 7, 28: 55, 29: 24, 30: 889, 31: 1210, 32: 2394, 33: 449, 34: 201, 35: 27, 36: 132, 37: 8966, 38: 62, 39: 65, 40: 73, 41: 11856, 42: 11128, 43: 3235, 44: 509, 45: 520, 46: 544, 47: 662, 48: 193, 49: 67, 50: 6410, 51: 181568, 52: 23678, 53: 5439, 54: 1654, 55: 2369, 56: 1382, 57: 519, 58: 218, 59: 98, 60: 283, 61: 263396, 62: 272441, 63: 15037, 64: 6579, 65: 14219, 66: 712, 67: 316, 68: 143, 69: 110, 70: 29, 71: 5859, 72: 86912, 73: 12740, 74: 244, 75: 131, 76: 152, 77: 118, 78: 10, 79: 24, 80: 4, 81: 11, 82: 3397, 83: 7916, 84: 894, 85: 749, 86: 197, 87: 6, 88: 1, 90: 4, 91: 14, 92: 26, 93: 1535, 94: 1111, 95: 344, 96: 136, 97: 11}
    if grid in traffic_dict:
        return (traffic_dict[grid] - mean)/std
    else:
        return 0
    
def plot_traffic_index():
    data_2d = []
    for i in range(10):
        temp_array = []
        for j in range(10):
            grid = int(str(i)+str(j))
            traffic = traffic_index(grid, True)
            temp_array.append(traffic)
        data_2d.append(temp_array)
    data_array = np.array(data_2d)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_data, y_data = np.meshgrid( np.arange(data_array.shape[1]),
                                  np.arange(data_array.shape[0]) )
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = data_array.flatten()
    ax.bar3d( x_data,
              y_data,
              np.zeros(len(z_data)),
              1, 1, z_data, alpha = 0.4 )
    ax.set_xlabel('longitude grid')
    ax.set_ylabel('latitude grid')
    ax.set_zlabel('standard deviations from avg')
    plt.savefig('traffic_index.png')
    plt.close()


#dictionary values extracted in interative spark session
def grid_avg_speed(grid):
    the_dict = {11: 0.003163869436663554, 12: 0.0060655381521713975, 15: 0.0072971014492753629, 20: 0.0062059334027807812, 21: 0.0053370440419429838, 22: 0.026185165157880233, 23: 0.0058180365769201154, 24: 0.0067430059156054793, 25: 0.002988888888888889, 27: 0.0044675925925925924, 28: 0.0060171919770773642, 29: 0.0046666666666666662, 30: 0.0056388524422969792, 31: 0.0048683004768510047, 32: 0.004990550908224956, 33: 0.0046790508566821303, 34: 0.0065284755260764894, 35: 0.003958256172839506, 36: 0.0087388543139942255, 37: 0.011826311475010523, 38: 0.0069395516204363104, 39: 0.0079007936507936513, 40: 0.0033480994234721573, 41: 0.00588870645753466, 42: 0.0051524611887042238, 43: 0.0055111111157533253, 44: 0.0046916647979933033, 45: 0.0051272139774715545, 46: 0.0071417858332063046, 47: 0.0088764321330903721, 48: 0.0080277854756059235, 49: 0.0048167989417989415, 50: 0.0052632701398469478, 51: 0.0042912585334380585, 52: 0.0045480447225492229, 53: 0.0090386600409524762, 54: 0.0058732560539466825, 55: 0.0084641324419957777, 56: 0.0065826221122309734, 57: 0.0056839796524810289, 58: 0.0064800362531844012, 59: 0.0048893603085935142, 60: 0.0052624730566815277, 61: 0.0038596194186580357, 62: 0.0041774299391788656, 63: 0.0063509448163084729, 64: 0.0058644409239805026, 65: 0.0079913281705285711, 66: 0.0066876374394507039, 67: 0.0054210786449749478, 68: 0.0057183532016436903, 69: 0.0099862258953168047, 70: 0.0040782273425966787, 71: 0.0038448910144802556, 72: 0.0042355978242406903, 73: 0.004902939865316691, 74: 0.0046467686860267154, 75: 0.0037668102730602725, 76: 0.0068826989621464212, 77: 0.0049127762409167577, 78: 0.0027619047619047619, 79: 0.0, 80: 0.0045553009590090926, 81: 0.0, 82: 0.0048768123397471864, 83: 0.0066914072074714219, 84: 0.0044726104677811204, 85: 0.0059395513874421725, 86: 0.0057059962346956618, 90: 0.0, 91: 0.0015654761904761905, 92: 0.0041671201022332096, 93: 0.0054447538532569374, 94: 0.0047641746761677671, 95: 0.0056103287692312676, 96: 0.005603293792458686, 97: 0.032786885245901641}
    if grid in the_dict:
        return the_dict[grid]
    else:
        return -1.0
            
