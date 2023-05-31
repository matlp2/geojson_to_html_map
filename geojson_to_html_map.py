
#commande: python3 geojson_to_html_map.py && chromium generated_map.html



import folium
from folium import plugins

import json
import geojson

import math

# import my_html
import sys
sys.path.append('..')
from my_html import *


import configparser

config = configparser.ConfigParser()

config.read('config.ini')

# [METADATA]
config_METADATA_title = config.get('METADATA', 'title', fallback = 'carte sans titre')
if not config_METADATA_title:
	config_METADATA_title = "sans nom"


geojson_filename = config.get('METADATA', 'geojson', fallback = 'carte sans titre')

# [MAP]
config_MAP_markers =  config.getboolean('MAP', 'markers', fallback = True)
config_MAP_clusterize_markers =  config.getboolean('MAP', 'clusterize_markers', fallback = True)

# heatmap options
config_MAP_Heatmap =  config.getboolean('MAP', 'heatmap', fallback = True)
heatmap_weight_field = config.get('MAP', 'heatmap_weight_field', fallback = None)

config_MAP_polygons =  config.getboolean('MAP', 'polygons', fallback = True)
config_MAP_lines =  config.getboolean('MAP', 'lines', fallback = True)

# max_number_of_features_displayed
max_number_of_features_displayed = config.getint('MAP', 'max_number_of_features_displayed', fallback = -1)
if max_number_of_features_displayed < 0: max_number_of_features_displayed = float('inf')



# [FIELDS]
config_FIELDS = config['FIELDS'] if 'FIELDS' in config else dict()
config_FIELDS_all = config.getboolean('FIELDS', 'all', fallback = None)
popups_display_raw_data = config.getboolean('FIELDS', 'all_raw_data', fallback = False)


# récupérer les champs indiqué dans l'ordre:
champX_to_name = dict()

for champX, field_name in config_FIELDS.items():
	if champX[:5] == 'champ':
		champX_to_name[champX] = field_name

# human sorting: champ10 > champ2 (not alphabetical sorting)
import re
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(pair):
    return tuple( atoi(c) for c in re.split(r'(\d+)', pair[0]) )
fields = [field_name for champX, field_name in sorted(champX_to_name.items(), key=natural_keys)]

if config_FIELDS_all == None and not fields:
	config_FIELDS_all = True


# required_fields must be a json list
import ast # for parsin list
required_fields = set(ast.literal_eval(config.get('FIELDS', 'required_fields', fallback = '[]')))


# choropleth option
choropleth_enabled = config.getboolean('MAP', 'choropleth', fallback = False)
choropleth_name = config.get('MAP', 'choropleth_name', fallback = 'choropleth sans nom')
choropleth_legend_value_name = config.get('MAP', 'choropleth_legend_value_name', fallback = None)
choropleth_key = config.get('MAP', 'choropleth_key', fallback = None)
choropleth_value = config.get('MAP', 'choropleth_value', fallback = None)
# Variable in the GeoJSON file to bind the data to, exemple: ‘feature.id’ or ‘feature.properties.statename’.
#choropleth_geojson_value = config.get('MAP', 'choropleth_geojson_value', fallback = None)
choropleth_geojson_value = ('feature.properties.' + choropleth_value) if choropleth_value else None
choropleth_colors = ast.literal_eval(config.get('MAP', 'choropleth_colors', fallback = "['yellow','green']"))
choropleth_color_steps = config.getint('MAP', 'choropleth_color_steps', fallback = -1)


# make polygons invisible, useful if choropleth_enabled
polygons_invisible = config.getboolean('MAP', 'polygons_invisible', fallback = choropleth_enabled)

if choropleth_enabled and not config_MAP_polygons:
	# enable polygons if choropleth_enabled
	config_MAP_polygons = True
	#print("WARNING : choropleth enabled but polygons disabled: wont be able to clic on displayed zone to see popup info => enable polygons")


import pandas as pd
from io import StringIO

def make_choropleth(m):
	if not choropleth_enabled: return

	key_value_pairs = []
	valid_features = []

	min_val = float('inf')
	max_val = float('-inf')

	for feature in features:

		properties = feature.get('properties', None)
	
		if properties:
			key = properties.get(choropleth_key, None)
			value = properties.get(choropleth_value, None)
			#print(key)
			#print(value)
	
			
	
			if key != None and value != None:
				key_value_pairs.append((key, value))
				valid_features.append(feature)
				min_val = min(min_val, value)
				max_val = max(max_val, value)

	choropleth_data_parameter = {
		choropleth_key: [k for k,v in key_value_pairs],
		choropleth_value: [v for k,v in key_value_pairs],
	}

	#print(choropleth_data_parameter)
	#print(pd.read_csv('choropleth_us/US_Unemployment_Oct2012.csv'))
	#print(pd.DataFrame(choropleth_data_parameter))
	
	if not valid_features: return

	choropleth = folium.Choropleth(
		#geo_data = 'choropleth_us/us-states.geojson',
		#geo_data = data,
		geo_data = { "type": "FeatureCollection", "features": valid_features},
		#geo_data = geojson,
		name = choropleth_name,
		#data=pd.read_csv('choropleth_us/US_Unemployment_Oct2012.csv'),
		data = pd.DataFrame(choropleth_data_parameter),
		#columns=["State", "Unemployment"],
		columns = [choropleth_key, choropleth_value],
		#columns = [choropleth_value],
		#key_on="feature.id",
		key_on = choropleth_geojson_value,
		#fill_color = "YlGn",
		fill_opacity = 0.7,
		line_opacity = 0.3,
		legend_name = choropleth_legend_value_name,
		colormap = None
	)


	color_map = folium.LinearColormap(
					choropleth_colors,
					vmin = min_val,
					vmax = max_val
				)

	if choropleth_color_steps >= 0:
		color_map = color_map.to_step(choropleth_color_steps)
	

	def get_color(feature):
		
		x = feature['properties'][choropleth_value]
		
		color = color_map(x)

		return {
			'fillColor': color,
			"color": color, # border color
			"opacity": 0.3,
			"weight": 1.7
		}

	choropleth.geojson.style_function = get_color

	# from https://github.com/python-visualization/folium/issues/956
	def delete_legend_of_choropleth(choropleth):
		for key in choropleth._children:
			if key.startswith('color_map'):
				del(choropleth._children[key])

	# les couleurs de la légende sont mauvaise par default
	delete_legend_of_choropleth(choropleth)

	# ajoute la légende avec les bonnes couleurs
	color_map.add_to(m)

	choropleth.add_to(m)

	




lat_lng_centre_de_rennes = (48.109848,-1.679194)
#lat_lng_centre_de_rennes = (48.110871, -1.659487)



#with open('points-presentation-bacs-roulants.geojson') as f:
#with open('points-presentation-bacs-roulants-formaté.geojson') as f:
#with open('MultiPolygon/aires-de-jeux-des-espaces-verts-rennes.geojson') as f:
with open(geojson_filename) as f:
#with open('LineString/amenagements-velo-et-zones-de-circulation-apaisee.geojson') as f:
#with open('MultiLineString/boucles_velo.geojson') as f:
#with open('Polygon/cantons-dille-et-vilaine-version-rennes-metropole.geojson') as f:
    data = geojson.load(f)

#print(data)


def feature_is_accepted(feature):
	geometry = feature.get('geometry', None)
	if geometry:

		geometry_type = geometry.get('type', None)

		if not geometry_type: return False

		if required_fields:
			properties = feature.get('properties', dict()).keys()
			for required_field in required_fields:
				if required_field not in properties:
					return False
		
		#match geometry_type:
		if   geometry_type == 'Point': return config_MAP_markers
		elif geometry_type == 'MultiPoint': return config_MAP_markers
		elif geometry_type == 'Polygon': return config_MAP_polygons or choropleth_enabled
		elif geometry_type == 'MultiPolygon': return config_MAP_polygons or choropleth_enabled
		elif geometry_type == 'LineString': return config_MAP_lines
		elif geometry_type == 'MultiLineString': return config_MAP_lines
		else: return False

	else:
		return False


features = list(filter(feature_is_accepted, data.get('features', ())))

#print(features)

cilcle_delimiter = None

if len(features) > max_number_of_features_displayed:
	import copy
	from numpy import asarray, array, cross, linalg, dot
	import numpy as np
	import time

	import pyproj

	ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
	lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
	#lon, lat, alt = pyproj.transform(ecef, lla, x, y, z, radians=True)

	#def ECEF_to_LLH(xyz):
	#	lon, lat, alt = pyproj.transform(ecef, lla, xyz[0], xyz[1], xyz[2], radians=False)
	#	return np.array((lat, lon))
	#LLH_to_ECEF = lambda lat, lon: np.array(pyproj.transform(lla, ecef, lat, lon, 6378137.0, radians=False))
	#def LLH_to_ECEF(lat, lon):
	#	x, y, z = pyproj.transform(lla, ecef, lat, lon, 6378137.0, radians=False)
	#	return np.array((x,y,z))
	#'''
	#lat lng to 3D point with (0,0,0) at the center of the earth with earth of radius 1
	#https://stackoverflow.com/questions/10473852/convert-latitude-and-longitude-to-point-in-3d-space
	def LLH_to_ECEF(
		lat,
		lon,
		#alt
	):
		alt = 0 # pareil si sphere de rayon 1

		# see http://www.mathworks.de/help/toolbox/aeroblks/llatoecefposition.html

		#rad = np.float64(6378137.0)        # Radius of the Earth (in meters)
		rad = np.float64(1.0)        # Radius of the Earth (in meters)
		#f = np.float64(1.0/298.257223563)  # Flattening factor WGS84 Model
		f = np.float64(0)  # Flattening factor WGS84 Model

		# math.radians
		lat = math.radians(lat)
		lon = math.radians(lon)

		cosLat = np.cos(lat)
		sinLat = np.sin(lat)

		# ON PEUT SIMPLIFIER CAR ON A PAS BESOIN DE L'ALTITUDE
		#FF     = (1.0-f)**2
		#C      = 1/np.sqrt(cosLat**2 + FF * sinLat**2)
		#S      = C * FF

		#x = (rad * C + alt)*cosLat * np.cos(lon)
		#y = (rad * C + alt)*cosLat * np.sin(lon)
		#z = (rad * S + alt)*sinLat

		x = cosLat * np.cos(lon)
		y = cosLat * np.sin(lon)
		z = sinLat

		return np.array((x, y, z))

	#https://stackoverflow.com/questions/56945401/converting-xyz-coordinates-to-longitutde-latitude-in-python
	def ECEF_to_LLH(xyz):
		x, y, z = xyz
		R = 1 # pareil si sphere de rayon 1
		lat = np.degrees(np.arcsin(z/R))
		lon = np.degrees(np.arctan2(y,x))
		return (lat, lon)

		#a = 6378137.0/np.float64(6378137.0) #in meters
		#b = 6356752.314245/np.float64(6378137.0) #in meters

		#f = (a - b) / a
		#f_inv = 1.0 / f

		#e_sq = f * (2 - f)                       
		#eps = e_sq / (1.0 - e_sq)

		#p = math.sqrt(x * x + y * y)
		#q = math.atan2((z * a), (p * b))

		#sin_q = math.sin(q)
		#cos_q = math.cos(q)

		#sin_q_3 = sin_q * sin_q * sin_q
		#cos_q_3 = cos_q * cos_q * cos_q

		#phi = math.atan2((z + eps * b * sin_q_3), (p - e_sq * a * cos_q_3))
		#lam = math.atan2(y, x)

		#v = a / math.sqrt(1.0 - e_sq * math.sin(phi) * math.sin(phi))
		#h   = (p / math.cos(phi)) - v

		#lat = math.degrees(phi)
		#lon = math.degrees(lam)

		#return (lat, lon)

	#'''
	#xyz_centre_de_rennes = LLH_to_ECEF(lat_lng_centre_de_rennes[0], lat_lng_centre_de_rennes[1])
	xyz_centre_de_rennes = LLH_to_ECEF(lat_lng_centre_de_rennes[0], lat_lng_centre_de_rennes[1])

	#print(np.linalg.norm(xyz_centre_de_rennes))
	#print(xyz_centre_de_rennes)

	#xyz_centre_de_rennes = xyz_centre_de_rennes/np.linalg.norm(xyz_centre_de_rennes)
	
	#print(np.linalg.norm(xyz_centre_de_rennes))

	# comme on considere que la terre est de rayon 1, xyz_centre_de_rennes doit être un vecteur unitaire
	assert (abs(np.linalg.norm(xyz_centre_de_rennes) - 1) < .001) 

	# projection sur plan 2D
	axe_lateral = cross(xyz_centre_de_rennes, array((0,0,1)))
	axe_vertical = cross(xyz_centre_de_rennes, axe_lateral)
	# normalize
	axe_lateral = axe_lateral/linalg.norm(axe_lateral)
	axe_vertical = axe_vertical/linalg.norm(axe_vertical)


	# projeter sur le plan (sphere -> plan)
	def _3D_to_2D(point_3D):
		return array((
			dot(point_3D, axe_lateral),
			dot(point_3D, axe_vertical)
		))


	# dé-projeter (plan -> sphere)
	def _2D_to_3D(point_2D):

		# rayon de la terre considéré 1
		R = 1

		# direction de la normale au plan
		d = xyz_centre_de_rennes

		dd = np.dot(d,d)
		dd = 1

		# position du point_2D sur le plan de normale d et de centre (0,0,0)  
		o = point_2D[0]*axe_lateral + point_2D[1]*axe_vertical
		K = np.dot(d,o)/dd
		K = 0
		t = -K + np.sqrt(K*K + (R*R - np.dot(o,o))/dd)
		print('t =',t, 'K =', K, 'o + t*d =', o + t*d, 'dd =',dd)
		print('linalg.norm(o + t*d) =', linalg.norm(o + t*d))
		#time.sleep(.5)
		return o + t*d


	def lat_lng_to_2D(lat, lng):
		return _3D_to_2D(LLH_to_ECEF(lat, lng))

	def _2D_to_lat_lng(point_2D):
		return ECEF_to_LLH(_2D_to_3D(point_2D))

	#def _2D_to_lng_lat(point_2D):
	#	lng, lat = _2D_to_lat_lng(point_2D)
	#	return (lat, lng)

	_2D_centre_de_rennes = lat_lng_to_2D(
		lat_lng_centre_de_rennes[0],
		lat_lng_centre_de_rennes[1]
	)


	from shapely.geometry import shape
	
	centre_de_rennes = shape({
		"type": "Point",
		#"coordinates": (lat_lng_centre_de_rennes[1], lat_lng_centre_de_rennes[0])
		"coordinates": _2D_centre_de_rennes
	})

	def get_dist(feature):
		if geometry := feature.get('geometry', None):

			def conver_points_to_2D(l):
				
				try:
					assert len(l) == 2
					val1 = float(l[1])
					val2 = float(l[0])
					return lat_lng_to_2D(val1, val2)
				except:
					return [conver_points_to_2D(e) for e in l]
					#return []
			
			geometry_2D = copy.deepcopy(geometry)

			
			#print(geometry['coordinates'])
			#print(conv_list(geometry2['coordinates']))
			geometry_2D['coordinates'] = conver_points_to_2D(geometry_2D['coordinates'])

			#exit(1)
			#print(conv_list(geometry))
			#print(geometry2)
			#return float(shape(geometry).distance(centre_de_rennes))
			#return float(shape(conv_list(geometry)).distance(xyz_centre_de_rennes))
			return float(shape(geometry_2D).distance(centre_de_rennes))
		else:
			return float('inf')
	
	features.sort(key = get_dist)
	

	zoom_start = 13.45
	latitude = lat_lng_centre_de_rennes[1]

	def metersPerPixel(latitude, zoomLevel):
		earthCircumference = 40075017
		latitudeRadians = latitude * (math.pi/180)
		return earthCircumference * math.cos(latitudeRadians) / math.pow(2, zoomLevel + 8);


	def pixelValue(latitude, meters, zoomLevel):
		return meters / metersPerPixel(latitude, zoomLevel)


	circle_thickness_in_meters = 50000
	circle_thickness_in_pixel = pixelValue(latitude, circle_thickness_in_meters, zoom_start)
	
	polygon_circle = []
	polygon_circle2 = []

	#radius_lat_lng = get_dist(features[max_number_of_features_displayed])
	radius_2D = get_dist(features[max_number_of_features_displayed])
	#radius_meters = 6_371_000*radius_lat_lng

	for i in range(100):
		a = (i/100)*2*math.pi
		d = np.array((math.cos(a), math.sin(a)))
		polygon_circle.append(_2D_to_lat_lng(_2D_centre_de_rennes + radius_2D*d))
		polygon_circle2.append(_2D_to_lat_lng(_2D_centre_de_rennes + .4*d))
		#polygon_circle2.append(_2D_to_lat_lng(_2D_centre_de_rennes + np.array((.5, .5))))
		#polygon_circle.append([lat_lng_centre_de_rennes[0] + radius_lat_lng*c, lat_lng_centre_de_rennes[1] +  radius_lat_lng*s])
		#polygon_circle2.append([lat_lng_centre_de_rennes[0] + 100000*radius_lat_lng*c, lat_lng_centre_de_rennes[1] +  100000*radius_lat_lng*s])
		#polygon_circle2.append([lat_lng_centre_de_rennes[0] + 100000*radius_lat_lng*c, lat_lng_centre_de_rennes[1] +  100000*radius_lat_lng*s])
		#polygon_circle2.append([lat_lng_centre_de_rennes[0] + 10000*radius_lat_lng*math.cos(t*2*math.pi), lat_lng_centre_de_rennes[1] +  10000*radius_lat_lng*math.sin(t*2*math.pi)])



	'''
	cilcle_delimiter = folium.GeoJson(

		
		
		{
			"type": "FeatureCollection",
			"features": [
				{
					"type": "Feature",
					"geometry": {
						"type": "Polygon",
						"coordinates": [
							[[lng, lat] for lat, lng in polygon_circle],
							polygon_circle2
						]
					},
					"properties": {
						'id':1
					}
				}
			]
		}
		
		,
		style_function = lambda x: {
			'fillColor': 'red',
			'color': 'green',
			'opacity': 1,
			'weight': 100
		}
	)
	'''

	#print(cilcle_delimiter)
	#print(cilcle_delimiter.__dict__)
	#exit(1)

	
	
	cilcle_delimiter = folium.Polygon(
		[polygon_circle, polygon_circle2],
		#[polygon_circle2, polygon_circle],
		#[polygon_circle + polygon_circle2],
		#polygon_circle,
		#[polygon_circle + polygon_circle2],
		#fillColor = 'red',
		#popup = "popup text",
		#tooltip = 'hover text',
		color = 'black',
		#color = '#8a8a8a',
		#opacity = .8,
		fill = True,
		fillOpacity = 1,# ne marche pas
		weight = 0,
		#style_function = lambda x: {
		#	'fillColor': 'red',
		#	'color': 'green',
		#	'fill': True
		#}
	)

	features = features[:max_number_of_features_displayed]
	


#points de la forme [lat, lng, weight] ou [lat, lng]
points = []
points_popups = []


polygons_features = []
lines_features = []

#popups_display_raw_data = False
accept_property_name = lambda property_name: property_name in fields


def make_popup(feature):

	properties = feature.get('properties', None)

	if properties:
		if popups_display_raw_data:
			points_popups.append(str(json.dumps(properties, sort_keys=True, indent=4)))
		else:

			#html_table = Html()

			#with html_table:

			

			with Html() as html_table:

				with block('style','type="text/css"'):
					html_table += '''td {
padding: 0 15px;
}
tr:nth-of-type(odd) {
      background-color:#ccc;
}'''
				#with block('table', 'border=1 frame=hsides rules=rows class="alternate_color"'):
				with block('table','style=\'font-family:"Courier New", Courier, monospace; font-size:120%;\''):
						with block('tbody'):

							def add_line(field_name, field_value):
								with block('tr'):
									with block('td', one_liner = True):
										html_table.s += field_name
									with block('td', one_liner = True):
										with block('span', 'style="white-space: nowrap"', one_liner = True):
											html_table.s += ': ' + field_value

							if config_FIELDS_all:
								
								for prop in properties.items():
									add_line(str(prop[0]), str(prop[1]))

							else:
								for field in fields:
									add_line(field, str(properties.get(field, None)))


		return str(html_table)
	else:
		return '"feature" sans champ "properties" dans le jeojson'


points_heatmap = []

def accept_point(geojson_point, feature):

	global points
	global points_popups
	global popups_display_raw_data

	lat = float(geojson_point[1])
	lng = float(geojson_point[0])

	if math.isnan(lat) or math.isnan(lng):# NaN messes up MarkerCluster
		return

	point = [lat, lng]# inverser lat/lng

	if heatmap_weight_field:
		weight = 1
		if properties := feature.get('properties', None):
			if w := properties.get(heatmap_weight_field, None):
				weight = w
			else:
				print('WARNING: geojson feature property without the cluster weight', heatmap_weight_field, ', weight set to 0')
				weight = 0
		
		points_heatmap.append(point + [weight])

	points.append(point)

	points_popups.append(make_popup(feature))

	




			





for feature in features:

	#print(feature)

	geometry = feature.get('geometry', None)

	if geometry:
		#print(geometry)
		geometry_type = geometry.get('type', None)
		coordinates = geometry.get('coordinates', None)

		if geometry_type == 'Point':
			accept_point(coordinates, feature)
			
		elif geometry_type == 'MultiPoint':
			for p in coordinates:
				accept_point(p, feature)
				
		elif geometry_type == 'Polygon':
			polygons_features.append(feature)

		elif geometry_type == 'MultiPolygon':
			polygons_features.append(feature)
			#for list_of_polygons in coordinates:
			#	for polygon in list_of_polygons:
			#		polygon = [(p[1],p[0]) for p in polygon]
			#		polygons.append(polygon)
			#		polygons_popups.append('hello')

		elif geometry_type == 'LineString':
			#lines.append(coordinates)
			#lines_popups.append(make_popup(feature))
			lines_features.append(feature)

		elif geometry_type == 'MultiLineString':
			#lines.append(coordinates)
			#lines_popups.append(make_popup(feature))
			lines_features.append(feature)

				







def save_html_map():

	c = folium.Map(
		location = lat_lng_centre_de_rennes,
		zoom_start = 13.45
	)

	#folium.Marker(lat_lng_centre_de_rennes, popup="centre_de_rennes").add_to(c)
	
	
	#bins = list(state_data["Unemployment"].quantile([0, 0.25, 0.5, 0.75, 1]))
	#choropleth = folium.Choropleth(
	#	geo_data='points-presentation-bacs-roulants.geojson',
	#	line_color='purple',
	#	line_weight=300
	#)

	#for p in points[:60]: folium.Marker(p,popup="hello").add_to(c)

	

	#choropleth.add_to(c)
	
	make_choropleth(c)
	



	#folium.HeatMap(list(zip(lat, lng))).add_to(map_osm)
	#folium.plugins.HeatMap([[p[0].astype(float), p[1].astype(float)] for p in points if p[0] != and p[1] != ]).add_to(c)
	#folium.plugins.HeatMap([[float(p[0]), float(p[1])] for p in points[:35]]).add_to(c)

	
	#for line in lines:
		#folium.PolyLine(
		#	line,
		#	popup=lines_popups,
		#	weight=5
		#).add_to(c)

	for feature in lines_features:
		geoj = folium.GeoJson(
			feature,
			style_function = lambda x: {
					'fillColor': 'yellow',
					'color': 'blue', 
					'weight': 4
				}
		).add_to(c)

		folium.Popup(make_popup(feature)).add_to(geoj)

	
	for feature in polygons_features:

		geoj = folium.GeoJson(
			feature,
			style_function = lambda x: {
					'fillColor': 'transparent' if polygons_invisible else 'blue',
					'color': 'transparent' if polygons_invisible else 'blue',
				}
		).add_to(c)

		folium.Popup(make_popup(feature)).add_to(geoj)

	#if polygons:
	#	print(polygons)
	#	for polygon in polygons:
	#		break
	#		folium.Polygon(
	#			polygon,
	#			polygons_popups,
	#			color = "red",
	#			style_function = lambda x: {'fillColor': 'orange'}
	#		).add_to(c)

	if points:

		if config_MAP_Heatmap:
			folium.plugins.HeatMap(
				points_heatmap if heatmap_weight_field else points,
				name="heatmap"
			).add_to(c)

		if config_MAP_markers and config_MAP_clusterize_markers:

			folium.plugins.MarkerCluster(
				points,
				name = "Marker Cluster",
				popups = points_popups,
				
				icon_create_function = None#'''
		#    function(cluster) {
		#
		#    return L.divIcon({html: '<b>' + cluster.getChildCount() + '</b>',
		#
		#                      className: 'marker-cluster marker-cluster-small',
		#
		#                      iconSize: new L.Point(20, 20)});
		#
		#    }
		#'''
			).add_to(c)

		elif config_MAP_markers:

			for point, popup in zip(points, points_popups):
				folium.Marker(point, popup = popup).add_to(c)


	folium.LayerControl().add_to(c)

	#c.save(config_METADATA_title + '.html')

	if cilcle_delimiter:
		cilcle_delimiter.add_to(c)


	c.save(str(config_METADATA_title) + '.html')
	
	#return f'<iframe src="{nom}.html" width="100%" height="100%" seamless></iframe>'
	#return f'<iframe src="{nom}.html" width="100%" height="100%" seamless></iframe>'




save_html_map()


