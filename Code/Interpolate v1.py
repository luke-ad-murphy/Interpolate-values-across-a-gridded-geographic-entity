# LIBRARIES USED
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.geometry import Polygon
import numpy as np
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from adjustText import adjust_text



##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


### Retrieve the building polygon
# setting cs to 4326 (lat lon)
building = gpd.GeoDataFrame.from_file("C:/your_location/POLYGON.shp").set_crs('EPSG:4326')

# View the building polygon
building.plot()

# Convert from Lat and Lon to Eastings and Northings
building = building.to_crs('EPSG:3347')
building.plot()
 

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################



### READ IN THE POLYGON AND POINTS DATA

# Read the points data extracted from GCP
df = pd.read_csv("C:/your_location /My_output.csv")

# Create a new geometry column in the df which combines the lat and lon values
# into a shapely Point() object. 
# Note that the Point() constructor expects a tuple of float values, 
# so conversion must be included if the dataframe's column dtypes are not already set to float.
# combine lat and lon column to a shapely Point() object
df['geometry'] = df.apply(lambda x: Point((float(x.Location_Longitude), 
                                           float(x.Location_Latitude))), axis=1)


# Now, convert the pandas DataFrame into a GeoDataFrame. 
# The geopandas constructor expects a geometry column which can consist of shapely geometry objects,
# so the column we created is fine:
gdf = gpd.GeoDataFrame(df, geometry='geometry').set_crs('EPSG:4326')

# Convert from Lat and Lon to Eastings and Northings
gdf = gdf.to_crs('EPSG:3347')

# create an object to be used for defining bounds of points being used
broader_bounds = gdf['geometry']
broader_bounds.plot()


# MNO and band count
gdf.Device_SIMServiceProviderBrandName.value_counts().head()
#Device_SIMServiceProviderBrandName=='MNO1'
#Device_SIMServiceProviderBrandName=='MON3'

gdf.Connection_Band.value_counts().head()
#Connection_Band=='1800+ MHz (Band 3)'

# filter to MNO1, MNO3 and LTE
gdf = gdf[gdf['Device_SIMServiceProviderBrandName'].isin(['MNO3'])]
gdf.Connection_Band.value_counts().head()

gdf = gdf[gdf['Connection_Band'] == '800 MHz (Band 20)']


##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


### Limit data points to building shape

# Spatial Joins to retrieve the grid id for the data points
# Using right join as I want to keep all grid ids for later
gdf = gpd.sjoin(gdf, building[['geometry']], how="inner", predicate='intersects')
gdf = gdf.drop('index_right', 1)  

# Add a field with 1 as a constant value
# Useful for understanding number of data points held within each polygon
gdf['const']=1


##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


### CREATE HEXBINS TO AGGREGATE DATA TO
# This will be applied to both known and modelled data
# Can play with these parameters

unit = 8

# Converting degrees to metres
# https://sciencing.com/convert-latitude-longtitude-feet-2724.html
approx_metres = (((2*3.14159265359) * 6371000) * unit) / 360
approx_metres

xmin, ymin, xmax, ymax = broader_bounds.total_bounds

# the "grid" used for the points of the hexagons to be created
a = np.sin(np.pi / 3)
cols = np.arange(xmin, xmax, 3 * unit)
rows = np.arange(ymin / a, ymax / a, unit)


hexagons = []
for x in cols:
  for i, y in enumerate(rows):
    if (i % 2 == 0):
      x0 = x
    else:
      x0 = x + 1.5 * unit
    hexagons.append(Polygon([
      (x0, y * a),
      (x0 + unit, y * a),
      (x0 + (1.5 * unit), (y + unit) * a),
      (x0 + unit, (y + (2 * unit)) * a),
      (x0, (y + (2 * unit)) * a),
      (x0 - (0.5 * unit), (y + unit) * a),
    ]))
    
    
grid = gpd.GeoDataFrame({'geometry': hexagons})
# Add the grid's area as a property - this may not be necessary
grid["grid_area"] = grid.area
#grid = grid.reset_index().rename(columns={"index": "grid_id"}).set_crs('EPSG:4326')
grid = grid.reset_index().rename(columns={"index": "grid_id"}).set_crs('EPSG:3347')

# Look at the grid
grid.plot()



# rework grid to include my building only
grid = gpd.sjoin(grid, building[['geometry']],
                  how="inner", predicate='intersects')

grid = grid.drop('index_right', 1)  

# Look at adjusted grid
grid.plot()


# write the grid to disk if needed - uncomment accordingly
#grid.to_file("C:/your_location /grid.shp")

# create the geojson file to be called later by folium process
#map_json = grid[['grid_id','geometry']]
#map_json.to_file("C:/your_location /BC_hexbins.json", driver="GeoJSON")



##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


### JOIN THE POINTS AND POLYGONS DATA

# Spatial Joins to retrieve the grid id for the data points
# Using right join as I want to keep all grid ids for later
gdf = gpd.sjoin(gdf, grid, how="right", predicate='intersects')
gdf['const']=1


##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


### AGGREGATE TO POLYGON AND CREATE MAP VIEW OF ALL ACTUAL DATA POINTS
# using sum so I can then divide 

grid_id_agg = gpd.GeoDataFrame(gdf.groupby(['grid_id','Device_SIMServiceProviderBrandName','Connection_Band']).
                        agg(RSRP_mean=('QOS_RSRP', 'mean'),
                            RSRP_sum=('QOS_RSRP', 'sum'),
                            count=('const', 'sum')).reset_index())
       
# create new geodataframe of key metrics by polygons
# Keeping only polygons where we have data
grid_id_agg = gpd.GeoDataFrame(pd.merge(grid_id_agg[[
    'grid_id',
    'Device_SIMServiceProviderBrandName',
    'Connection_Band',
    'RSRP_mean',
    'RSRP_sum',
    'count'
    ]],
    grid[['grid_id','geometry']], on='grid_id'))


grid_id_agg.describe()

# Remap rsrp to a set of discrete incremental values
bins = [-np.inf, -130, -110, -100, -90, -80, -60, np.inf]
labels = ['min to -130',
          '-130 to -110',
          '-110 to -100',
          '-100 to -90',
          '-90 to -80',
          '-80 to -60',
          'Missing']

# create map colour scheme
cmap = matplotlib.colors.ListedColormap(["red","orange","yellow","lightgreen","green","blue","black"], name='from_list', N=None)



# Clip hexbins to the building and filll na with 0
# Keep all grids anpopulate with MNO values
clipped = gpd.GeoDataFrame(pd.merge(grid, grid_id_agg[['grid_id','RSRP_mean']], on='grid_id', how='left'))

# Clip hexbins to the building and filll na with 0
clipped = clipped.clip(building)
clipped['RSRP_mean'] = clipped['RSRP_mean'].fillna(0)

# Apply bandings
clipped['QOS_RSRP_band'] = pd.cut(clipped['RSRP_mean'], bins=bins, labels=labels)

# create maps
fig, axJ = plt.subplots(1, figsize=(14, 8),dpi=80)
axJ.axis('off')
clipped.plot(column='QOS_RSRP_band', cmap=cmap, edgecolor='None', legend=False, ax=axJ)
plt.savefig("C:/your_location/Voda_Known_coverage.png")


##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
      

### FETCH CENTROIDS AND METRICS TO TRAIN MODEL
### SET UP DATA POINTS AND GRID TO INTERPLOATE OVER

# Centroids of populated hexbins for each MNO
grid_id_agg["x"] = grid_id_agg.centroid.x
grid_id_agg["y"] = grid_id_agg.centroid.y


# Get this warning when running on EPSG:4326:
# UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.
# Think this is OK across small polygons
# https://gis.stackexchange.com/questions/372564/userwarning-when-trying-to-get-centroid-from-a-polygon-geopandas

# Set up grid to interpolate over using the limits of the initial grid set up
# Resolution refers to the increments to impute between the min and max values 
resolution = unit / 6  # Grid set to 1/6th the size of hexbin degrees

gridx = np.arange(xmin, xmax, resolution)
gridy = np.arange(ymin, ymax, resolution)

# Calc the mean of target value to populate nan and missings
#MNO1_target_mean = gdf['QOS_RSRP'].mean()


##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


### NEAREST NEIGHBOUR INTERPOLATION - BY MNO

# Use Nearest Neighbour method to create interpolation model
# note that for building and testing this model I'm using the training data set
NearestND_model_RSRP_sum = NearestNDInterpolator(x = list(zip(grid_id_agg["x"], 
                                           grid_id_agg["y"])),
                              y = grid_id_agg["RSRP_sum"],
                              tree_options={"balanced_tree": False},
                              rescale=False)

NearestND_model_count = NearestNDInterpolator(x = list(zip(grid_id_agg["x"], 
                                           grid_id_agg["y"])),
                              y = grid_id_agg["count"],
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
    scores = gpd.sjoin(scores, grid, how="right", predicate='intersects')
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
    scores = gpd.sjoin(scores, grid, how="right", predicate='intersects')
    # drop spurious index_left column 
    scores = scores.drop(['index_left', 'const'], 1)
    return(scores)

# Note here that I'm overwriting the original 2D model output dataframes
NearestND_count = count_scores_to_df(NearestND_count)



# Aggregate total projected RSRP and counts to polygon level
def final_score(RSRP, count):
    R = gpd.GeoDataFrame(RSRP.groupby(
        ['grid_id']
        ).agg({
            'RSRP_sum':'sum'
            }).reset_index())

    C = gpd.GeoDataFrame(count.groupby(
        ['grid_id']
        ).agg({
            'count':'sum'
            }).reset_index())

    # Join sum of projected counts and sum of projected RSRP
    scores = gpd.GeoDataFrame(pd.merge(R,C, on='grid_id'))      

    scores['RSRP_mean'] = scores['RSRP_sum']/scores['count']
    return(scores)

NearestND_final = final_score(NearestND_RSRP_sum, NearestND_count)


##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


### Create map of ALL actuals plus modelled for chosen model
original_missings = clipped[clipped.QOS_RSRP_band == 'Missing']
original_complete = clipped[clipped.QOS_RSRP_band != 'Missing']

scored_missings = gpd.GeoDataFrame(pd.merge(NearestND_final[[
    'grid_id',
    'RSRP_mean'
    ]],
    original_missings[['grid_id','geometry']], on='grid_id', how='inner', indicator=True))

final_map = pd.concat([original_complete,scored_missings],ignore_index=True)
final_map = final_map.drop('QOS_RSRP_band', 1)  


# Clip hexbins to the building and filll na with 0
final_map = final_map.clip(building)

# Apply bandings
final_map['QOS_RSRP_band'] = pd.cut(final_map['RSRP_mean'], bins=bins, labels=labels)

# create map
fig, axJ = plt.subplots(1, figsize=(14, 8),dpi=80)
axJ.axis('off')
final_map.plot(column='QOS_RSRP_band', cmap=cmap, edgecolor='None', legend=False, ax=axJ)
plt.savefig("C:/your_location/MNO3 known plus modelled RSRP.png")

##############################################################################
##############################################################################
############################# END - OF - PROGRAM #############################
##############################################################################
##############################################################################
