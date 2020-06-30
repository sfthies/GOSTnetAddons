import networkx as nx
import numpy as np
import geopandas as gpd
import GOSTnets as gn
import pandas as pd
from shapely.geometry import LineString, Point, MultiPoint, MultiLineString
import mapclassify
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
import fiona

    
def make_fishnet(gpd_df, res, reduced = True):
    """
    Generates a polygon grid from a geopandas gdf given a resolution in metres specified as res.
    """
    
    from shapely.geometry import Polygon, Point
    
    if(gpd_df.crs['init']=='epsg:4326'):
        raise ValueError("gpd_df is not in a metric projection")
    
    xmin,ymin,xmax,ymax = gpd_df.total_bounds
    cols = list(range(int(np.floor(xmin)- 3*res), int(np.ceil(xmax)+3*res), res))
    rows = list(range(int(np.floor(ymin)-3*res), int(np.ceil(ymax)+3*res), res))
    
    rows.reverse()

    polygons = []
    for x in cols:
        for y in rows:
            polygons.append(Polygon([(x,y+res), (x+res, y+res), (x+res, y), (x, y)]))

    ## Transform fishnet into geoseries:
    fishnet = gpd.GeoSeries(polygons)
    fishnet.crs = gpd_df.crs
    
    if reduced == True:
        fishnet = fishnet[fishnet.centroid.within(gpd_df.buffer(0).unary_union.buffer(res/2))]
        
    print('Number of cols', len(cols), '. Number of rows', len(rows))
    return fishnet

def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    
    Returns a concave hull and a list with all edge points of the Delauny triangulation
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return MultiPoint(list(points)).convex_hull

    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    triangles = coords[tri.vertices]
    a = ((triangles[:,0,0] - triangles[:,1,0]) ** 2 + (triangles[:,0,1] - triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:,1,0] - triangles[:,2,0]) ** 2 + (triangles[:,1,1] - triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:,2,0] - triangles[:,0,0]) ** 2 + (triangles[:,2,1] - triangles[:,0,1]) ** 2) ** 0.5
    s = ( a + b + c ) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:,(0,1)]
    edge2 = filtered[:,(1,2)]
    edge3 = filtered[:,(2,0)]
    edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis = 0).tolist()
    m = MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points
   

def pretty_fisherjenks(y, k = 5, digits = 2):
    """
    Return pretty, rounded, Fisher Jenks classification schemes. For fast classifications use pretty_fisherjenks_sampled. 
    Relies on mapclassify. 
    -----------
    Parameters:
    y: input vector
    k: number of classes
    digits: degree of rounding
    -----------
    Returns: mapclassify object
    """
    original = mapclassify.FisherJenks(y, k = k)
    accuracies = (-(np.floor(np.log10(original.bins))-digits)).astype(int)
    pretty_bins = [round(limit, accuracies[i]) for i, limit in enumerate(original.bins)]
    return mapclassify.UserDefined(y, pretty_bins)

def pretty_fisherjenks_sampled(y, k = 5, digits = 2):
    """
    Return pretty, rounded, Fisher Jenks Sampled classification schemes. Relies on mapclassify. 
    -----------
    Parameters:
    y: input vector
    k: number of classes
    digits: degree of rounding
    -----------
    Returns: mapclassify object
    """
    original = mapclassify.FisherJenksSampled(y, k= k)
    accuracies = (-(np.floor(np.log10(original.bins))-digits)).astype(int)
    pretty_bins = [round(limit, accuracies[i]) for i, limit in enumerate(original.bins)]
    return mapclassify.UserDefined(y, pretty_bins)



def area_weighted_aggregation(gdf1, gdf2, col, crs_p = None):
    """
    gdf1: spatial unit of aggregation
    gdf2: spatial data to be aggregated
    col: columns of gdf2 selected for aggregation ### ADD functionality to autoselect numeric columns...
    crs: metric crs used for calculations
    """
    
    if crs_p == None:
        gdf1_p = gdf1.copy()
        gdf2_p = gdf2.copy()
    else:        
        gdf1_p = gdf1.to_crs(crs_p)
        gdf2_p = gdf2.to_crs(crs_p)
    
    gdf1_p['INDEX'] = gdf1_p.index
    
    gdf2_p['AREA'] = gdf2_p.area
    
    #Overlay
    overlay = gpd.overlay(gdf1_p, gdf2_p)
    
    #Calculate aggregation weights:
    overlay['WEIGHT'] = overlay.area/overlay['AREA']
    
    #Calculate weighted col values:
    for x in col:
        overlay[x] = overlay[x]*overlay['WEIGHT']
    
    agg_cols = col.copy()
    agg_cols.append('INDEX')
        
    result = overlay[agg_cols].groupby('INDEX').sum()
    return result   
    

def prepend(list, str): 
    """
    Prepend string to a list of strings
    """
    str += '{0}'
    list = [str.format(i) for i in list] 
    return(list) 
    
def nn_index(gdf1,gdf2, k = 1):
    """
    Function that takes a spatial point gdf1 and returns indices and distances
    of k nearest neighbors in spatial point gdf2 
    -----------
    Parameters:
    gdf1 : spatial point data frame
    gdf2 : spatial point data frame which contains potential neighbors
    k : number of neighbors
    """
    gd1 = gdf1.copy()
    gd2 = gdf2.copy()
           
    ## NN lookup
    #Import spatial from scipy for very quick NN lookup:
    from scipy import spatial
    
    G_tree = spatial.KDTree(np.array([gd2.geometry.x, gd2.geometry.y]).T)
    distances, indices = G_tree.query(np.array([gd1.geometry.x, gd1.geometry.y]).T, k = k)
    
    indici = gd2.index.values[indices]
    
    col_names = prepend(np.arange(1,k+1,1).astype(str), 'index_k_') + prepend(np.arange(1,k+1,1).astype(str), 'dist_k_')
    
    result = pd.DataFrame(np.column_stack([indici, distances]), columns= col_names, index = gd1.index)
    result[prepend(np.arange(1,k+1,1).astype(str), 'index_k_')]= result[prepend(np.arange(1,k+1,1).astype(str), 'index_k_')].astype(int)
    
    return result

def inverse_weighting(origin, origin_column, target, dist_thres, k, p = 1, map_to_nn = True):
    """
    Assigns spatial data contained in the origin data frame to the target data frame based on inverse distance weighting.
    ---------
    Arguments:
    
    origin: A spatial points geopandas.GeoDataFrame object with valid geometry column
    origin_column: the respective column in the origin data frame to be mapped
    target: a spatial points geopandas.GeoDataFrame object with valid geometry column
    dist_thres: A distance threshold in metres. Nearest neighbors beyond the threshold won't be considered for mapping
    k: number of nearest neighbors used for mapping
    p: Minkowski distance used for weight calculation (default = 1)
    map_to_nn: logical, if True and all neighbors further than dist_thres away, all data will be mapped to closest neighbor
                        if False an error message is produced    
    ---------
    """
    
    #Calculate indices and distances to k NN:
    temp_index_dist = nn_index(origin, target, k = k)
    
    #Set up vector to store results:
    results = pd.Series(0, index = target.index)
    
    to_nn_count = 0
    
    for ind, row in temp_index_dist.iterrows():
        #Select distances
        temp_dists = row.iloc[-k:]
        
        #Filter valid distances:
        if sum(temp_dists < dist_thres)>0:
            valid_temp_dists = temp_dists[temp_dists < dist_thres]
        else:
            if map_to_nn == True:
                valid_temp_dists = pd.Series(temp_dists[0])
                to_nn_count += 1
            else:
                return print('No neighbor within the distance threshold for origin: ', ind)
        
        #Calculate weights:
        temp_weights = valid_temp_dists**(-p)
        
        #Calculate values:
        temp_values = temp_weights*origin.loc[ind,origin_column]/temp_weights.sum()
        #and adjust their indices:
        temp_values.index = list(row.iloc[:len(valid_temp_dists)].astype(int))
        
        #Add values to the result column:
        results.loc[temp_values.index] += temp_values
        
    print(to_nn_count, ' values were mapped to their single nearest neighbor\n')
        
    return results

    
def unique(list1):
    x = np.array(list1)
    return list(np.unique(x))

def peartree_length_to_time(peartree_osmnx):
    """
    Renames "length" attribute in peartree generated networks to "time" (to avoid confusion with actual length in metres)
    """
    for u,v, dat in peartree_osmnx.edges(data=True):
        dat['time'] = dat['length']
        del dat['length']
    return peartree_osmnx

def peartree_relabel_nodes(G):
    """
    Relabels nodes (and edges) in peartree generated networks to refelect actual node names
    Only works if node names are seperated by "_"
    """
   
    G_temp = G.copy()
    ## Generate dict for relabeling:
    relab_dict = {}
    for noden, data in G_temp.nodes(data = True):
        if(len(noden.split('_'))==2):
           relab_dict[noden] = noden.split('_')[1]
        else:
           relab_dict[noden] = noden
    ## relabel nodes using nx
    G_final = nx.relabel_nodes(G_temp, relab_dict)
    return G_final


def NN_line_cut(linestring, point1, point2):
    """
    Function cuts shapely linestring into a segment between point1 and point2 based on NN mapping.
    Point 1 and point 2 are mapped to the closest coordinates in linestring (works best in metric projection)
    
    ADD SOME TOLERANCE FOR POINTS TO AVOID MAPPING TO POINTS FAR AWAY
    """
    from scipy import spatial

    line_coords = np.array(linestring.coords)
    G_tree = spatial.KDTree(np.array(line_coords))
    np_point_coords= np.array(list(zip(point1.coords, point2.coords)))
    distances, indices = G_tree.query(np_point_coords)
    
    if indices[0][0]<indices[0][1]:
        cut_line = LineString(line_coords[range(indices[0][0],indices[0][1]+1),])
    else:
        if indices[0][0]>indices[0][1]:
            cut_line = LineString(line_coords[range(indices[0][1],indices[0][0]+1),])
        else:
            cut_line = None
    
    return cut_line



def simple_line_cut(linestring, point1, dist_thres, tol, trace = True):
    from shapely.ops import nearest_points
    from shapely.ops import split
    from shapely.geometry import MultiPoint, LineString
    
    #Find nearest points on line:
    np1 = nearest_points(linestring, point1)
    
    # Return error message if points are too far away from line:
    dist1 = np1[0].distance(np1[1]) 
    if dist1> dist_thres:
        if trace:
            print('Point1 is', dist1, 'units apart from line which is more than the specified distance threshold of', dist_thres)
        return None
    
    #Do split:
    split_1 = split(linestring, np1[0].buffer(tol))
    if len(list(split_1))== 1:
        if trace:
            print('Splitting at mapped point 1 did not work')
        return None
    
    return list(split_1)

    
def double_line_cut(linestring, point1, point2, dist_thres, tol, trace = True):
    """
    Function maps points 1 and 2 on the linestring and cuts it into a segment.
    If point1 or point2 are further than tol away from the line, an error message is produced
    Eps is a small radius used for cutting the linestring appropriately
    """
    
    from shapely.ops import nearest_points
    from shapely.ops import split
    from shapely.geometry import MultiPoint, LineString
    
    ### Do first split:
    
    #Find nearest points on line:
    np1 = nearest_points(linestring, point1)
    
    # Return error message if points are too far away from line:
    dist1 = np1[0].distance(np1[1]) 
    if dist1> dist_thres:
        if trace:
            print('Point1 is', dist1, 'units apart from line which is more than the specified distance threshold of', dist_thres)
        return None
    
    #Do split:
    split_1 = split(linestring, np1[0].buffer(tol))
    if len(list(split_1))== 1:
        if trace:
            print('Splitting at mapped point 1 did not work')
        return None
    
    ### Prepare second split:
    
    # figure out which split works better:
    np2_1 = nearest_points(split_1[0], point2)
    np2_2 = nearest_points(split_1[-1], point2)
    
    #determine the better working split:
    if np2_1[0].distance(np2_1[1]) < np2_2[0].distance(np2_2[1]):
        np2 = np2_1
        temp_split = split_1[0]
    else: 
        np2 = np2_2
        temp_split = split_1[-1]
    
    #Return error message if this split does not work:
    dist2 = np2[0].distance(np2[1]) 
    if dist2 > dist_thres:
        if trace:
            print('Point2 is', dist2, 'units apart from line which is more than the specified distance threshold of', dist_thres)
        return None
          
    #Do split:
    split_2 = split(temp_split, np2[0].buffer(tol))
    if len(list(split_2))== 1:
        if trace:
            print('Splitting at mapped point 2 did not work')
        return None
    
    #Select correct split:
    if split_2[0].distance(np1[0]) < split_2[-1].distance(np1[0]):
        final_seg = split_2[0]
    else:
        final_seg = split_2[-1]  
    
    return final_seg

def GTFS_to_edge_gdf(GTFS, crs, trace = False, bad_trips = 'point_lines'):
    """
    Function takes partridge GTFS feed and transforms it into an edge gdf
    -------------
    Arguments:
    GTFS: Partridge GTFS geo feed object
    crs: The crs used for projections - should be metric to allow calculations in cartesian coordinates
    trace: If True, progress is reported
    bad_trips: How to handle invalid geometries on trips. One of the following:
        'point_lines': Determines if start and stop point are the same and returns eiter straight line or point geoemtry
        'points': returns "point trips", i.e. simply the start point to ensure a valid geometry        
        None: leaves geometry empty and can cause errors in later processing steps 
    -------------
    Returns:
    geopandas data frame with all edges, travel times, stop times, and geometries.
    travel times are calculated as the pure trip time from stop point u to stop point v
    stop times refer to the stop time at the destination v
    """
    
    #Generate empty pandas data frame:
    route_df= pd.DataFrame(columns=['route_id', 'trip_id', 'trip_section','u','v', 'time', 'stop_time', 'travel_time', 'start_time','shape_id', 'geometry'])
    
    #Prepare all transformations in respective crs:
    if crs == None:
        crs = GTFS.shapes.crs
    
    
    #Set indices of shapes and stops:
    GTFS_shapes =GTFS.shapes.to_crs(crs).set_index("shape_id")
    GTFS_stops = GTFS.stops.to_crs(crs).set_index("stop_id")
    
    

    #Iterate through all routes in the GTFS route data frame:
    for _,row in GTFS.routes.iterrows():
        
        #select temporary route:
        temp_route = row['route_id']

        #Obtain all trips on that route:
        temp_trips_df = GTFS.trips[GTFS.trips.route_id == temp_route]

        #Iterate through the trips:
        for _, trip_row in temp_trips_df.iterrows():

            #Obtain temporay trip and shape id:
            temp_trip, temp_shape_id = trip_row[['trip_id', 'shape_id']]

            #Get temporary shape (important for cutting out network edge elements):
            temp_shape = GTFS_shapes.loc[temp_shape_id, 'geometry']

            #Get all stops on the trip:
            temp_trip_stops =  GTFS.stop_times[GTFS.stop_times.trip_id == temp_trip].sort_values(by = 'stop_sequence', ascending = True).reset_index()
            
            #If there is less than 2 stops on the route, print trip_id and shape_id to check
            if len(temp_trip_stops)<=1:
                print(temp_trip, temp_shape_id)

            else:
                for i in range(len(temp_trip_stops)-1):

                    if trace == True:
                        print(temp_trip, i)
                    u = temp_trip_stops.loc[i, 'stop_id']
                    v = temp_trip_stops.loc[i+1, 'stop_id'] 

                    #if u !=v:
                    travel_time = temp_trip_stops.loc[i+1, 'arrival_time']- temp_trip_stops.loc[i, 'departure_time']
                    stop_time = temp_trip_stops.loc[i+1, 'departure_time']- temp_trip_stops.loc[i+1, 'arrival_time']
                    time = travel_time + stop_time

                    u_point = GTFS_stops.loc[u, 'geometry']
                    v_point = GTFS_stops.loc[v, 'geometry']

                    seg_geom = NN_line_cut(temp_shape, u_point, v_point)
                    #seg_geom = double_line_cut(temp_shape, u_point, v_point, dist_thres, tol, trace_linecut)

                    if (seg_geom == None):
                        print('Trip_id:', temp_trip, 'section:', i,'from stop', u, 'to stop', v, 'has no valid geometry')
                        if bad_trips == 'point_lines':
                            if u_point == v_point:
                                print('Replace with point')
                                seg_geom = u_point
                            else:
                                print('Replace with line')
                                seg_geom = LineString([u_point, v_point])
                        elif bad_trips == 'points':
                            print('Replace with point')
                            seg_geom = u_point
                        else:
                            print('No replacement')


                    dat_dict = {'u': u, 'v': v, 'time': time, 'travel_time': travel_time, 'stop_time': stop_time, 'geometry': seg_geom,
                               'start_time': temp_trip_stops.loc[i, 'departure_time'], 'route_id': temp_route, 'trip_id': temp_trip,
                                'shape_id': temp_shape_id, 'trip_section': i}

                    route_df = route_df.append(dat_dict, ignore_index = True)

    route_gdf = gpd.GeoDataFrame(route_df, geometry = 'geometry')
    
    route_gdf.crs = crs
    return route_gdf

def GTFS_generate_graph(route_data, stops_geodata, name):
    """
    Code to generate a networkx graph from route_df and all_stops_gdf. Name assigns a name to the generated data.
    -----------
    Arguments:
    route_data: a pandas or geopandas route data frame. Needs to contain columns 'u' and 'v' for origin destination pairs and potentially further attributes including a geometry
    stops_geodata: a geopandas stop points data frame that contains all stops in the network. Needs to contain the columns 'stop id' and 'geometry' and potentially further attributes such as waiting times, etc
    name: a name of the network. Required for some processing steps in osmnx
    """
    
    route_df = route_data.copy()
    all_stops_gdf = stops_geodata.copy()
    
    ## Ensure that geometries have same crs if route_df has geometry column
    if 'geometry' in route_df.columns:
        if route_df.crs != all_stops_gdf.crs:
            return print('Please ensure that route_data and stops_geodata are in the same crs projection')
    
    #Add x and y columns to all_stops_gdf if not available already
    if ('x' not in all_stops_gdf.columns or 'y' not in all_stops_gdf.columns):
        all_stops_gdf['x'] = all_stops_gdf.geometry.x
        all_stops_gdf['y'] = all_stops_gdf.geometry.y
    
    ### Generate Graph:
    G = nx.MultiDiGraph()
    G.name = name
    G.graph['crs'] = all_stops_gdf.crs
    
    #Set the index of stops:
    all_stops_gdf = all_stops_gdf.set_index('stop_id')
   
    #Subselect all stops that are part in the edge data frame:
    all_stops_gdf = all_stops_gdf.loc[unique(route_df.loc[:,['u','v']].values.reshape(-1)),:].sort_index()
       
    G.add_nodes_from(all_stops_gdf.index)
    attributes = all_stops_gdf.to_dict()

    ## Add all nodes to the network:
    for attribute_name in all_stops_gdf.columns:
        # only add this attribute to nodes which have a non-null value for it
        #attribute_values = {k:v for k, v in attributes[attribute_name].items() if pd.notnull(v)}
        attribute_values = {k:v for k, v in attributes[attribute_name].items()}
        nx.set_node_attributes(G, name=attribute_name, values=attribute_values)

    ## Add all routes to the network:
    for _, row in route_df.iterrows():
            attrs = {}
            for label, value in row.iteritems():
                if (label not in ['u', 'v', 'key']) and (isinstance(value, list) or pd.notnull(value)):
                    attrs[label] = value
            G.add_edge(row['u'], row['v'], **attrs)
    
    return G

def commuter_generate_graph(route_data, stops_geodata, name = "Matatu"):
    """
    Code to generate a networkx graph from route_df and all_stops_gdf. Name assigns a name to the generated data.
    Equivalent to GTFS_generate_graph but does not include key for edges.
    """
    
    route_df = route_data.copy()
    all_stops_gdf = stops_geodata.copy()
    
    ### Generate Graph:
    G = nx.MultiDiGraph()
    G.name = name
    G.graph['crs'] = all_stops_gdf.crs
    
    #Set the index
    all_stops_gdf.index = all_stops_gdf['stop_id']
    G.add_nodes_from(all_stops_gdf.index)
    attributes = all_stops_gdf.to_dict()

    ## Add all nodes to the network:
    for attribute_name in all_stops_gdf.columns:
        # only add this attribute to nodes which have a non-null value for it
        #attribute_values = {k:v for k, v in attributes[attribute_name].items() if pd.notnull(v)}
        attribute_values = {k:v for k, v in attributes[attribute_name].items()}
        nx.set_node_attributes(G, name=attribute_name, values=attribute_values)

    ## Add all routes to the network:
    for _, row in route_df.iterrows():
            attrs = {}
            for label, value in row.iteritems():
                if (label not in ['u', 'v', 'key']) and (isinstance(value, list) or pd.notnull(value)):
                    attrs[label] = value
            G.add_edge(row['u'], row['v'], **attrs)
    
    return G

def add_mode_to_network(G, mode):
    """
    Adds a new edge and node attribute "mode" to the network to distinguish e.g. bus from walk or train
    """
    G_temp = G.copy()
    count_e = 0
    count_n = 0
    
    #For edges:
    for u,v, data in G_temp.edges(data= True):
        if not 'mode' in data.keys():
            data['mode'] = mode
            count_e = count_e + 1
    
    #for nodes:
    for idn, data in G_temp.nodes(data = True):
        if not 'mode' in data.keys():
            data['mode'] = mode
            count_n = count_n + 1
    
    if count_e == 0:
        print('No mode attributes were added to edges')
    else:
        print(count_e, 'mode attributes were added to edges and set to', mode)
        
    if count_n == 0:
        print('No mode attributes were added to nodes')
    else:
        print(count_n, 'mode attributes were added to nodes and set to', mode)
        
    return(G_temp)    

def replace_mode_in_network(G, mode):
    """
    Replaces and adds a new edge and node attribute "mode" to the network to distinguish e.g. bus from walk or train
    """
    G_temp = G.copy()
    count_e = 0
    count_n = 0
    
    #For edges:
    for u,v, data in G_temp.edges(data= True):
        data['mode'] = mode
        count_e = count_e + 1
    
    #for nodes:
    for idn, data in G_temp.nodes(data = True):
        data['mode'] = mode
        count_n = count_n + 1
    
    
    print(count_e, 'mode attributes of edges were replaces and set to', mode)
    print(count_n, 'mode attributes of nodes were replaces and set to', mode)
        
        
    return(G_temp)    


def set_boarding_cost(G, cost):
    """
    Sets a general boarding cost for every node in a network - only needed if boarding_c is not provided
    (modify this function in the future to allow for variations)
    """
    
    G_temp = G.copy()
    
    #Loop through nodes and set boarding costs, called boarding_c:
    for idn, data in G_temp.nodes(data = True):
        data['boarding_c'] = cost
    
    return G_temp

def has_valid_time_edges(G):
    """
    Checks whether edge time attributes are all valid floats
    """
    for u,v, data in G.edges(data = True):
        if not 'time' in data.keys():
            raise ValueError('Time attribute is missing for some edges')
            return
        
        if isinstance(data['time'], float) ==False:
            raise ValueError('Time attribute is not of type float')
            return
        
def make_valid_time_edges(G):
    """
    Transforms all edge time attributes into valid floats
    """
    G_temp = G.copy()
    for u,v, data in G_temp.edges(data = True):
        if not 'time' in data.keys():
            raise ValueError('Time attribute is missing for some edges')
            return
        if isinstance(data['time'], float) ==False:
            data['time'] = float(data['time'])
    return G_temp
            
def get_nearest_nodes(base_net,point_gdf):
    """
    Function to obtain nearest nodes to a point gdf
    MAKE SURE THAT ALL INPUTS ARE IN SAME PROJECTION
    """
    #Extract gdf from the network:
    base_net_nodes = gn.node_gdf_from_graph(base_net)

    #Read gdf of new network nodes
    new_net_nodes = point_gdf.copy()
    new_net_nodes['x'] = new_net_nodes.geometry.x
    new_net_nodes['y'] = new_net_nodes.geometry.y

    ## NN lookup
    #Import spatial from scipy for very quick NN lookup:
    from scipy import spatial
    G_tree = spatial.KDTree(base_net_nodes[['x','y']].values)
    distances, indices = G_tree.query(new_net_nodes[['x','y']].values)

    #Add NN to new net gdf:
    new_net_nodes['NN'] = list(base_net_nodes['node_ID'].iloc[indices])

    #Add distance to the NN node:
    new_net_nodes['NN_dist'] = distances

    return new_net_nodes

def combine_networks(base, new, walk_speed = 4.5, walk_thres = 100, add_boarding_costs = False, connect_to_walk_only = True):
    """
    Combines two networks assuming that close nodes can be reached by walking
    MAKE SURE NETWORKS ARE PROJECTED IN UNIT WITH METERS BEFORE USING!
        
    Parameters
    ----------
    base: baseline network
    new: network to add
    add_boarding_costs: If True, nodes need column with "boarding_c" (boarding cost) information which will be included
    walk_speed: assumed walking speed in km/h 
    walk_thres: threshold in meters for maximum distance between connected nodes 
    connect_to_walk_only: if this is set to True, the new network will only be connected with nodes having the mode 'walk'
    """
    
    #Make a copy of the original networks
    base_net = base.copy()
    new_net = new.copy()
    
    if base_net.graph['crs'] != new_net.graph['crs']:
        raise ValueError('Networks should be in same projection')
        return
    
    if base_net.graph['crs']['init'] == 'epsg:4236':
        raise ValueError('Networks need to be in a metric projection')
        return
    
      
    #Extract gdf from the two objects:
    base_net_nodes = gn.node_gdf_from_graph(base_net)
    new_net_nodes = gn.node_gdf_from_graph(new_net)
        
    if connect_to_walk_only == True:
        #Only connect to nodes from the walkable network:
        base_net_nodes = base_net_nodes[base_net_nodes['mode']=='walk']
    
    ## NN lookup
    #Import spatial from scipy for very quick NN lookup:
    from scipy import spatial
    G_tree = spatial.KDTree(base_net_nodes[['x','y']].values)
    distances, indices = G_tree.query(new_net_nodes[['x','y']].values)
    
    #Add NN to new net gdf:
    new_net_nodes['NN'] = list(base_net_nodes['node_ID'].iloc[indices])
    
    #Add distance to the NN node:
    new_net_nodes['NN_dist'] = distances
    
    #Create lists with all nodes and edges to add to the original network:
    nodes_to_add = []
    edges_to_add = []
    
    #add all nodes from the new network
    for u, data in new_net.nodes(data = True):
        u = 'add_net_%s' % u
        nodes_to_add.append((u,data))

    #add all edges from the new network
    for u,v, data in new_net.edges(data = True):
        u = 'add_net_%s' % u
        v = 'add_net_%s' % v
        edges_to_add.append((u,v,data))
        
    ## add connections between old network and new network based on NN:
    # THIS is also were different waiting times are considered
    # Only add connection if closer than walk threshold

    for index, row in new_net_nodes.iterrows():
        #Node from new network:
        u = 'add_net_%s' % row['node_ID']
        #Node from old network:
        v = row['NN']
        
        # Data when walking from the station:
        data = {}
        data['length'] = row['NN_dist']
        data['infra_type'] = 'net_glue'
        data['mode'] = 'net_glue (walk)'
        data['time'] = row['NN_dist']/(walk_speed*1000/3600)
        #data['Wkt'] = LineString([row.geometry, gdf_base.geometry.loc[v]])
        if data['length']< walk_thres:
            edges_to_add.append((u, v, data))
        
        # Modify data when walking to the station (including waiting time / frequency)
        if add_boarding_costs == True:
            data2 = {}
            data2['length'] = row['NN_dist']
            data2['infra_type'] = 'net_glue'
            data2['mode'] = 'net_glue (walk)'
            data2['boarding_c'] = row['boarding_c']
            data2['boarding_w'] = row['NN_dist']/(walk_speed*1000/3600)
            data2['time'] = data2['boarding_c'] + data2['boarding_w']
            if data2['length']< walk_thres:
                edges_to_add.append((v, u, data2))
        else:
            if data['length']< walk_thres:
                edges_to_add.append((v, u, data))
    
    ## Add edges and nodes:
    base_net.add_nodes_from(nodes_to_add)
    base_net.add_edges_from(edges_to_add)
    
    #
    not_added_nodes = new_net_nodes[new_net_nodes['NN_dist']>walk_thres]
    if len(not_added_nodes)>0:
        print('The following nodes were not connected due to large distance from base_net:\n', not_added_nodes[['node_ID', 'NN_dist']])
    
    ## Give all nodes new names:
    base_net = nx.convert_node_labels_to_integers(base_net)
    
    return base_net


def select_trip_freq_bytime(start, end, GTFS_freq):
    indi = np.logical_and(GTFS_freq['end_time']>=start, GTFS_freq['start_time']<=end)
    GTFS_freq_selec = GTFS_freq[indi]

    #GTFS_freq_selec['e_start_time'] =
    GTFS_freq_selec['start_time_eff'] = np.maximum(GTFS_freq_selec['start_time'], start)
    GTFS_freq_selec['end_time_eff'] = np.minimum(GTFS_freq_selec['end_time'], end)
    GTFS_freq_selec['range_time_eff'] = GTFS_freq_selec['end_time_eff'] - GTFS_freq_selec['start_time_eff']
    
    return GTFS_freq_selec

def average_trip_freq(start, end, GTFS_freq):
    GTFS_freq_selec = select_trip_freq_bytime(start, end, GTFS_freq)
    wtavg = lambda x: np.average(x.loc[:,'headway_secs'], weights = x.loc[:,'range_time_eff'])
    avg_times = GTFS_freq_selec.groupby('trip_id').apply(wtavg)
    start_time = GTFS_freq_selec.groupby('trip_id')['start_time_eff'].min()
    end_time = GTFS_freq_selec.groupby('trip_id')['end_time_eff'].max()
    
    return pd.DataFrame({'avg_times': avg_times,
                  'op_start_time': start_time,
                  'op_end_time': end_time})
    

def prepare_network_data(GTFS, crs, start = None, end = None, bad_trips = 'point_lines', trace = False ):
    """
    Transforms a GTFS feed into an edge data frame and a stop point GeoDataFrame with all information needed to generate a network x transport network.
    If start and end specified, average headways are calculated and boarding costs added to the stop point data frame
    -----------
    Parameters:
    GTFS <partridge.GTFS.feed>: The respective GTFS feed
    crs <dict>: The crs used for all calculations. Should be metric
    start <float>: Start time in seconds after midnight
    end <float>: End time in seconds after midnight
    bad_trips <string>: How to handle invalid geometries? Valid options are: 'point_lines', 'points', None
    trace <bool>: Report progress?
    
    Returns:
    GTFS_edge <geopandas.GeoDataFrame>
    GTFS_stops <geopandas.GeoDataFrame>    
    """
    
    #Load edge data:
    GTFS_edge = GTFS_to_edge_gdf(GTFS, crs , trace,  bad_trips)
    #Add length and speed information:
    GTFS_edge['length'] = GTFS_edge.length
    GTFS_edge['speed'] = GTFS_edge['length']/GTFS_edge['travel_time']*3.6
    #Load stop data
    GTFS_stops = GTFS.stops.to_crs(crs)
    
    #Obtain all trips that are operative during a time and their average frequency
    if start != None:
        avg_trip_freq = average_trip_freq(start, end, GTFS.frequencies)
        operative_trips = avg_trip_freq.index.values
        
        #Select all operative trips:
        GTFS_edge = GTFS_edge.set_index('trip_id').loc[operative_trips,].reset_index()
        
        #Select operative nodes
        operative_nodes = pd.concat([GTFS_edge.merge(avg_trip_freq, how = 'left', on ='trip_id').loc[:,['u','avg_times', 'op_start_time', 'op_end_time']],
                             GTFS_edge.merge(avg_trip_freq, how = 'left', on ='trip_id').loc[:,['v','avg_times', 'op_start_time', 'op_end_time']].rename(columns = dict(v = 'u'))], axis = 0).groupby('u').agg({'avg_times': 'mean', 'op_start_time': 'min','op_end_time': 'max' }).reset_index()

        operative_nodes = operative_nodes.rename(columns = dict(avg_times = 'boarding_c',
                                     u = 'stop_id'))
        
        GTFS_stops =  gpd.GeoDataFrame(operative_nodes.merge(GTFS_stops, how = 'left', on = 'stop_id'), geometry = 'geometry', crs = crs)
        
    return GTFS_edge, GTFS_stops
    
    
def line_to_equi_points(line, inter_dist):
    """
    Function takes shapely line and returns a list of equi-distant points on that line
    -------
    Parameters:
    line: shapely line object
    inter_dist: interpolation distance. Make sure that the distance fits the specified crs/ projection
    """
    current_dist = inter_dist
    line_length = line.length
    ## append the starting coordinate to the list
    list_points = []
    list_points.append(Point(list(line.coords)[0]))

    ## while the current cumulative distance is less than the total length of the line
    while current_dist < line_length:
        ## use interpolate and increase the current distance
        list_points.append(line.interpolate(current_dist))
        current_dist += inter_dist
    
    return list_points


def linegdf_to_pointgdf(data, equi_dist, return_counts = True, add_line_id = False):
    """
    Transforms a geopandas GeoDataFrame with Spatial Line geometries to a GeoDataFrame consiting of all the points that define the lines.
    All line attributes are replicated.
    ------
    Parameters:
    data: a geopandas GeoDataFrame with spatial line geometries
    equi_dist: a distance at which the points shall be resampled
    return counts: if True, the returned data frame contains a column "COUNTS" specifying how many resampled point refer to the same line
    add_line_id: if True,  the returned data frame contains a column "ID" specifying to which unique line a point belongs
    """
    temp_crs = data.crs.copy()
    column_names = data.columns
    column_dtypes = data.dtypes
    all_vals = np.empty((0, len(column_names)))
    all_points = []
    all_counts = []
    all_ids = []

    for i, row in data.iterrows():
        temp_points = line_to_equi_points(row['geometry'], equi_dist)
        all_points.extend(temp_points)
        all_counts.extend([len(temp_points)]*len(temp_points))
        all_vals = np.append(all_vals, np.array([row.values]*len(temp_points)),axis =0)
        all_ids.extend([i]*len(temp_points))
        
    data_out = gpd.GeoDataFrame(all_vals, columns= column_names,
                                geometry = gpd.GeoSeries(all_points))
    
    data_out = data_out.astype(column_dtypes)
    
    if return_counts:
        data_out['COUNTS'] = all_counts
    
    if add_line_id:
        data_out['ID'] = all_ids
    
    data_out.crs = temp_crs 
    return data_out