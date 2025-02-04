import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Polygon, Point
import numpy as np
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from adjustText import adjust_text

# directory for inputs and outputs
dir_name = "C:/your_dir/"

# choose hexbin size from Uber h3
# Ref: https://towardsdatascience.com/uber-h3-for-data-analysis-with-python-1e54acdcc908
h3_level = 12


##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


### Retrieve the building polygon
# setting cs to 4326 (lat lon)
building = gpd.GeoDataFrame.from_file(dir_name + "POLYGON.shp").set_crs('EPSG:4326')

# View the building polygon
building.plot()


##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


### READ IN THE POLYGON AND POINTS DATA

# Read the points data extracted from GCP
df = pd.read_csv(dir_name + "tutela_data_points.csv")

# Apply band and MNO filters
df.Device_SIMServiceProviderBrandName.value_counts().head()
df.Connection_Band.value_counts().head()

df = df[df['Device_SIMServiceProviderBrandName'].isin([‘MNO3’])]
df = df[df['Connection_Band'] == '800 MHz (Band 20)']


# Create a new geometry column in the df which combines the lat and lon values
# into a shapely Point() object. 
# Note that the Point() constructor expects a tuple of float values, 
# so conversion must be included if the dataframe's column dtypes are not already set to float.
# combine lat and lon column to a shapely Point() object
df['geometry'] = df.apply(lambda x: Point((float(x.Location_Longitude), 
                                           float(x.Location_Latitude))), axis=1)

# Convert the pandas DataFrame into a GeoDataFrame. 
gdf = gpd.GeoDataFrame(df, geometry='geometry').set_crs('EPSG:4326')


##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


### CREATE GRIDS FOR AREA AND BUILDING USING H3


# create an object to be used for defining bounds of points being used
bounding_box = gdf['geometry']
bounding_box.plot()
xmin, ymin, xmax, ymax = bounding_box.total_bounds

# Create grid for total area of grid and interpolation
resolution = 0.00001
gridx = np.arange(xmin, xmax, resolution)
gridy = np.arange(ymin, ymax, resolution)

bounding_box_grid = np.transpose([np.tile(gridx, len(gridy)), np.repeat(gridy, len(gridx))])

bounding_box_grid = pd.DataFrame(bounding_box_grid).rename(columns={0:'Location_Longitude',
                                                                    1:'Location_Latitude'})

bounding_box_grid['geometry'] = bounding_box_grid.apply(
    lambda x: Point((float(x.Location_Longitude),
                     float(x.Location_Latitude))), axis=1)

bounding_box_grid = gpd.GeoDataFrame(bounding_box_grid, geometry='geometry').set_crs('EPSG:4326')

# Apply h3 grid to points data
# resolution 11 is 0.024910561km (so approx 25m)
def lat_lng_to_h3(row):
    return h3.geo_to_h3(
      row.geometry.y, row.geometry.x, h3_level)
 
bounding_box_grid['h3_ref'] = bounding_box_grid.apply(lat_lng_to_h3, axis=1)

master_grid = gpd.GeoDataFrame(bounding_box_grid.groupby('h3_ref').
                        agg(count=('h3_ref', 'count')).reset_index())

# turn h3 hexs into geo. boundary
master_grid['geometry'] = master_grid["h3_ref"].apply(lambda x: h3.h3_to_geo_boundary(h=x, geo_json=True))
# turn to Point
master_grid['geometry'] = master_grid['geometry'].apply(lambda x: [Point(x,y) for [x,y] in x])
# turn to Polygon
master_grid['geometry'] = master_grid['geometry'].apply(lambda x: Polygon([[poly.x, poly.y] for poly in x]))

master_grid = master_grid.set_crs('EPSG:4326')

# look at the grid
master_grid.plot()
master_grid = master_grid.drop(['count'], axis=1)


# grid for building
building_grid = gpd.sjoin(master_grid, building[['geometry']],
                          how="inner", predicate='intersects').drop('index_right', 1)  

building_grid = building_grid.set_crs('EPSG:4326')

# look at the grid
building_grid.plot()


##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


### JOIN THE POINTS AND POLYGONS DATA

# Spatial Joins to retrieve the grid id for the data points
# Using right join to building grid as I want to keep all grids to show entire building
gdf = gpd.sjoin(gdf, building_grid[['h3_ref', 'geometry']],
                how="right", predicate='intersects').drop('index_left', 1)  


##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


### AGGREGATE TO POLYGON AND CREATE MAP VIEW OF ALL ACTUAL DATA POINTS
# using sum so I can then divide 

grid_id_agg = gpd.GeoDataFrame(gdf.groupby(['h3_ref',
                                            #'geometry'
                                            #'Device_SIMServiceProviderBrandName',
                                            #'Connection_Band'
                                            ]).
                        agg(RSRP_mean=('QOS_RSRP', 'mean'),
                            RSRP_sum=('QOS_RSRP', 'sum'),
                            count=('QOS_RSRP', 'count')).reset_index())
       
# create new geodataframe of key metrics by polygons
# Keeping only polygons where we have data
grid_id_agg = gpd.GeoDataFrame(pd.merge(grid_id_agg[[
    'h3_ref',
    #'Device_SIMServiceProviderBrandName',
    #'Connection_Band',
    'RSRP_mean',
    'RSRP_sum',
    'count'
    ]],
    building_grid[['h3_ref','geometry']], on='h3_ref'))


# Take a look at RSRP distribution
grid_id_agg.describe()

# Remap rsrp to a set of discrete incremental values
# Remap rsrp to a set of discrete incremental values
bins = [-np.inf, -120, -110, -100, -90, -80, -60, np.inf]
labels = ['min to -130',
          '-130 to -110',
          '-110 to -100',
          '-100 to -90',
          '-90 to -80',
          '-80 to -60',
          'Missing']

# create map colour scheme
cmap = matplotlib.colors.ListedColormap(["red","orange","yellow","lightgreen","green","blue","black"], name='from_list', N=None)



## Clip data to building and map
clipped = gpd.GeoDataFrame(pd.merge(building_grid[['h3_ref','geometry']], 
                                    grid_id_agg[['h3_ref','RSRP_mean']], 
                                    on='h3_ref', how='left')).set_crs('EPSG:4326')

# Clip hexbins to the building and filll na with 0
clipped = clipped.clip(building)
clipped['RSRP_mean'] = clipped['RSRP_mean'].fillna(0)

# Apply bandings
clipped['QOS_RSRP_band'] = pd.cut(clipped['RSRP_mean'], bins=bins, labels=labels)

# create maps
fig, axJ = plt.subplots(1, figsize=(14, 8),dpi=80)
axJ.axis('off')
clipped.plot(column='QOS_RSRP_band', cmap=cmap, edgecolor='None', legend=True, ax=axJ)
plt.savefig(dir_name + "Known coverage.png")



##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
    

### FETCH CENTROIDS AND METRICS TO TRAIN MODEL

# Centroids of populated hexbins for each MNO
pop_grid_id_agg = grid_id_agg[grid_id_agg.RSRP_mean.notnull()]

pop_grid_id_agg["x"] = pop_grid_id_agg.centroid.x
pop_grid_id_agg["y"] = pop_grid_id_agg.centroid.y


##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


### NEAREST NEIGHBOUR INTERPOLATION

# Use Nearest Neighbour method to create interpolation model
# note that for building and testing this model I'm using the training data set
NearestND_model_RSRP_sum = NearestNDInterpolator(x = list(zip(pop_grid_id_agg["x"], 
                                           pop_grid_id_agg["y"])),
                              y = pop_grid_id_agg["RSRP_sum"],
                              tree_options={"balanced_tree": False},
                              rescale=False)

NearestND_model_count = NearestNDInterpolator(x = list(zip(pop_grid_id_agg["x"], 
                                           pop_grid_id_agg["y"])),
                              y = pop_grid_id_agg["count"],
                              tree_options={"balanced_tree": False},
                              rescale=False)

NearestND_RSRP_sum = NearestND_model_RSRP_sum(*np.meshgrid(gridx, gridy))
NearestND_count = NearestND_model_count(*np.meshgrid(gridx, gridy))

NearestND_RSRP_sum = pd.DataFrame(NearestND_RSRP_sum)
NearestND_count = pd.DataFrame(NearestND_count)


##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


### FORMAT THE MODEL OUTPUT IN PREPARATION FOR RUNNING SOME TESTS

# set some universal parameters for use in the subsequent function
df_gridy = pd.DataFrame(gridy)
df_gridy.rename(columns={0: 'Location_Latitude'}, inplace=True)


# Function to turn the model scores into amore user-friendly output
def RSRP_scores_to_df(model):
    scores = model.copy()
    # rename columns using Lon coordinates
    scores.columns = [gridx]
    # fetch the latitude values to turn into row names
    scores = pd.merge(scores, df_gridy, left_index=True, right_index=True)
    # 'melt' df into Lon/Lat stacked df
    scores = scores.melt(id_vars=["Location_Latitude"])
    # convert Lon object field to string
    scores['ESTR'] = scores["variable"].astype(str)
    # convert Lon string values to float
    scores["Location_Longitude"] = scores["ESTR"].str.split('(').str[1].str.split(',').str[0].str.split(')').str[0].astype(np.float64)
    # tidy up the output
    scores = scores.rename(columns={'value':'RSRP_sum'}).drop(['variable','ESTR'], axis=1)
    # add a constant for summarising convenience
    scores['const']=1
    # convert to GDF
    scores['geometry'] = scores.apply(lambda x: Point((float(x.Location_Longitude), 
                                               float(x.Location_Latitude))), axis=1)
    scores = gpd.GeoDataFrame(scores, geometry='geometry')
    # retrieve grid ids
    scores = gpd.sjoin(scores, building_grid, how="right", predicate='intersects')
    # drop spurious index_left column 
    scores = scores.drop(['index_left', 'const'], 1)
    return(scores)

# Note here that I'm overwriting the original 2D model output dataframes
NearestND_RSRP_sum = RSRP_scores_to_df(NearestND_RSRP_sum)



def count_scores_to_df(model):
    scores = model.copy()
    # rename columns using Lon coordinates
    scores.columns = [gridx]
    # fetch the latitude values to turn into row names
    scores = pd.merge(scores, df_gridy, left_index=True, right_index=True)
    # 'melt' df into Lon/Lat stacked df
    scores = scores.melt(id_vars=["Location_Latitude"])
    # convert Lon object field to string
    scores['ESTR'] = scores["variable"].astype(str)
    # convert Lon string values to float
    scores["Location_Longitude"] = scores["ESTR"].str.split('(').str[1].str.split(',').str[0].str.split(')').str[0].astype(np.float64)
    # tidy up the output
    scores = scores.rename(columns={'value':'count'}).drop(['variable','ESTR'], axis=1)
    # add a constant for summarising convenience
    scores['const']=1
    # convert to GDF
    scores['geometry'] = scores.apply(lambda x: Point((float(x.Location_Longitude), 
                                               float(x.Location_Latitude))), axis=1)
    scores = gpd.GeoDataFrame(scores, geometry='geometry')
    # retrieve grid ids
    scores = gpd.sjoin(scores, building_grid, how="right", predicate='intersects')
    # drop spurious index_left column 
    scores = scores.drop(['index_left', 'const'], 1)
    return(scores)

# Note here that I'm overwriting the original 2D model output dataframes
NearestND_count = count_scores_to_df(NearestND_count)



# Aggregate total projected RSRP and counts to polygon level
def final_score(RSRP, count):
    R = gpd.GeoDataFrame(RSRP.groupby(
        ['h3_ref']
        ).agg({
            'RSRP_sum':'sum'
            }).reset_index())

    C = gpd.GeoDataFrame(count.groupby(
        ['h3_ref']
        ).agg({
            'count':'sum'
            }).reset_index())

    # Join sum of projected counts and sum of projected RSRP
    scores = gpd.GeoDataFrame(pd.merge(R,C, on='h3_ref'))      

    scores['RSRP_mean'] = scores['RSRP_sum']/scores['count']
    return(scores)

NearestND_final = final_score(NearestND_RSRP_sum, NearestND_count)


##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


### Create map of ALL actuals plus modelled for chosen model

### Create map of ALL actuals plus modelled for chosen model
original_missings = clipped[clipped.QOS_RSRP_band == 'Missing']
original_complete = clipped[clipped.QOS_RSRP_band != 'Missing']

scored_missings = gpd.GeoDataFrame(pd.merge(NearestND_final[[
    'h3_ref',
    'RSRP_mean'
    ]],
    original_missings[['h3_ref','geometry']], on='h3_ref', how='inner', indicator=True))

# Concatenate known and modelled data
final_map = pd.concat([original_complete,scored_missings],ignore_index=True)
final_map = final_map.drop(['QOS_RSRP_band', '_merge'], 1)  
final_map.describe()


# Clip hexbins to the building and filll na with 0
final_map = final_map.clip(building)

# Apply bandings
final_map['QOS_RSRP_band'] = pd.cut(final_map['RSRP_mean'], bins=bins, labels=labels)

# create map
fig, axJ = plt.subplots(1, figsize=(14, 8),dpi=80)
axJ.axis('off')
final_map.plot(column='QOS_RSRP_band', cmap=cmap, edgecolor='None', legend=True, ax=axJ)
plt.savefig(dir_name + "Known plus modelled RSRP.png")



##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


### Add NQT test points to map

# Read in test locations
df_NQT = pd.read_csv(dir_name + "NQT_tests.csv")

# apply geometry to test locations
df_NQT['geometry'] = gpd.GeoSeries.from_wkt(df_NQT['position'])

# convert to GDF
gdf_NQT = gpd.GeoDataFrame(df_NQT, geometry='geometry').set_crs('EPSG:4326')

# convert to eastings and northings
gdf_NQT['lon'] = gdf_NQT.geometry.x
gdf_NQT['lat'] = gdf_NQT.geometry.y


# Known plus modelled map with NQT RSRP
fig, axJ = plt.subplots(1, figsize=(14, 8),dpi=80)
axJ.axis('off') # Set an axis to plot data onto
final_map.plot(column='QOS_RSRP_band', 
               cmap=cmap, 
               edgecolor='None', 
               legend=True, 
               ax=axJ) # chosing axis to plot to
gdf_NQT.plot(column='rsrp', 
             marker="D", 
             color='black', 
             markersize = 80, 
             ax=axJ) # chosing axis to plot to
# add labels for test values
texts = []
for i, txt in enumerate(gdf_NQT.rsrp):
    texts.append(axJ.text(gdf_NQT.lon[i], gdf_NQT.lat[i], txt,  fontsize=16))
adjust_text(texts)
plt.savefig(dir_name + "Known plus modelled with NQT RSRP.png")


# Known plus modelled map with NQT RSRP
fig, axJ = plt.subplots(1, figsize=(14, 8),dpi=80)
axJ.axis('off') # Set an axis to plot data onto
final_map.plot(column='QOS_RSRP_band', 
               cmap=cmap, 
               edgecolor='None', 
               legend=True, 
               ax=axJ) # chosing axis to plot to
gdf_NQT.plot(column='nqt_score', 
             marker="D", 
             color='black', 
             markersize = 80, 
             ax=axJ) # chosing axis to plot to
# add labels for test values
texts = []
for i, txt in enumerate(gdf_NQT.nqt_score):
    texts.append(axJ.text(gdf_NQT.lon[i], gdf_NQT.lat[i], txt,  fontsize=16))
adjust_text(texts)
plt.savefig(dir_name + "Known plus modelled with NQT overall.png")



##############################################################################
##############################################################################
############################# END - OF - PROGRAM #############################
##############################################################################
##############################################################################
