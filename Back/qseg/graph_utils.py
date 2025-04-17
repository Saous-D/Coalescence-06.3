import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

import sys
def log(message):
    print(message)
    sys.stdout.flush()

""" reduced multiclass global graph """
def super_multiclass_global_Hamiltonian(gray_img,k,c,t_links , clustered_image,clusters, sigma=0.5, mu=1, **kwargs):
    """
  Calculate the linear and quadratic coefficients of the reduced global multiclass hamiltonian
  Parameters:
  gray_img (array): a grayscale image
  k number of classes
  c penalty coefficient vector
  t_links: representing the standard intensities of each class (used to calculate spectral penalties)
  mu (float) caracterizing balance between local and spectral term
  sigma (float): quantifying the sensitivity of the similarity function
  clustered_image (1-D array) cluster attributed by the clustering algo for each pixel
  clusters (list): mean intensities caracterizing clusters
  Returns:
  linear & quandratic terms
    """
    h, w = gray_img.shape
    nodes = np.zeros((h*w, 1))
    edges = []
    extra = []

    min_weight = 1
    max_weight = 0
    for i in range(len(clusters)):        
          for j in range(k):
                weight = 1 - gaussian_similarity(t_links[j],
                                                  clusters[i], sigma)
                extra.append(( i , j , weight))
                if min_weight > weight:
                    min_weight = weight
                if max_weight < weight:
                    max_weight = weight

    for i in range(h*w):
        x, y = i // w, i % w
        nodes[i] = gray_img[x, y]           
        if x > 0:
            j = (x - 1) * w + y
            weight = 1 - gaussian_similarity(gray_img[x, y], gray_img[x - 1, y], sigma)
            edges.append((i, j, weight))

        if y > 0:
            j = x * w + y - 1
            weight = 1 - gaussian_similarity(gray_img[x, y], gray_img[x, y - 1], sigma)
            edges.append((i, j, weight))


    # Initialize super_weights with default value of 0
    super_local_weights = defaultdict(float)
    n_local_weights = defaultdict(float)
    # Iterate over the quadratic keys
    for u, v, weight in edges:
        # Check if the elements belong to different clusters

        if clustered_image[u] != clustered_image[v]:
            i = clusters.index(clustered_image[u])
            j = clusters.index(clustered_image[v])
            n_local_weights[(min(i, j),max(i, j))] += 1
            # Update the weight in super_weights
            super_local_weights[(min(i, j),max(i, j))] += weight

    # Convert defaultdict back to a regular dictionary if needed
    super_local_weights = dict(super_local_weights)  
    n_local_weights = dict(n_local_weights)

    super_local_weights = {(i, j): super_local_weights[(i, j)] / n_local_weights[(i, j)] for i,j in super_local_weights.keys()}
    max_weight = max(np.max(list(super_local_weights.values())), max_weight )
    min_weight = min(np.min(list(super_local_weights.values())), min_weight )

    a=-1
    b=1                
    if max_weight-min_weight: 
        normalized_edges = [(node1,node2,-1*np.round(((b-a)*(( super_local_weights[(node1,node2)]-min_weight)/(max_weight-min_weight)))+a,4)) for node1,node2 in super_local_weights.keys()]
        normalized_extra = {(i,j): -1 * np.round(((b-a) * ((weight - min_weight) / (max_weight - min_weight))) + a, 4)
                            for i,j,weight in extra}         
    elif max_weight==0 and min_weight==0:
        normalized_edges = [(node1,node2,1) for node1,node2 in super_local_weights.keys()]
        normalized_extra = {(i,j):1 for i,j,weight in extra}
    else:
        normalized_edges = [(node1,node2,-1*np.round(super_local_weights[(node1,node2)],4)) for node1,node2 in super_local_weights.keys()]
        normalized_extra = {(i,j):-1 * np.round(weight, 4) for i,j,weight in extra}

    linear = {}
    for idx in range(len(clusters)):
          for j in range(k-1):
              linear[j*len(clusters) + idx] = (-1) * mu * normalized_extra[(idx,j+1)]
              for idy in range(k):
                  if idy != (j+1):
                      linear[j*len(clusters) + idx] += (1/(k-1)) * mu * normalized_extra[(idx,idy)]

    for iy, ix,edge_weight in normalized_edges:
            for j in range(k-1):
                        linear[j * len(clusters) + iy] += edge_weight
                        linear[j * len(clusters) + ix] += edge_weight


    quadratic = {(j * len(clusters) + iy , i * len(clusters) + ix):0 for iy, ix,edge_weight in normalized_edges for j in range(k-1) for i in range(k-1)}
    for iy, ix,edge_weight in normalized_edges:
          for j in range(k-1):
              quadratic[(j * len(clusters) + iy , j * len(clusters) + ix)] += -2*edge_weight
              for i in range(k-1):
                  if i!=j:
                      quadratic[(j * len(clusters) + iy , i * len(clusters) + ix)] += -edge_weight
    if k > 2:               
           for v in range(len(clusters)):
               for i in range(k-1):

                   for j in range(i+1,k-1):
                              quadratic[(i * len(clusters) + v , j * len(clusters) + v)] = c[v]

    return linear , quadratic   

""" superpixels Graph / reduced graph / calculate the reduced qubo while preserving locality info """
def super_Global_Hamiltonian(gray_img,clustered_image,clusters, global_highest_value, global_lowest_value, sigma, mu, callback=None, **kwargs):
    """
    Calculate the linear and quadratic coefficients of the reduced global hamiltonian (hypergraph / superpixel graph)
    Parameters:
    gray_img (array): a grayscale image
    clustered_image (1-D array) cluster attributed by the clustering algo for each pixel
    clusters (list): mean intensities caracterizing clusters
    global_highest_value, global_lowest_value: representing the standard intensities of each class (used to calculate spectral penalties)
    sigma (float): quantifying the sensitivity of the similarity function
    mu  (float) caracterizing balance between local and spectral term


    Returns:
    linear & quandratic terms
    """
    h, w = gray_img.shape
    nodes = np.zeros((h*w, 1))
    edges = []
    extra_max = []
    extra_min=[]
    min_weight = 1
    max_weight = 0

    for i in range(len(clusters)):
        weight = 1 - gaussian_similarity(global_highest_value,
                                       clusters[i], sigma)
        extra_max.append((i , weight))
        if min_weight > weight:
            min_weight = weight
        if max_weight < weight:
            max_weight = weight
        weight = 1 - gaussian_similarity(global_lowest_value,
                                          clusters[i], sigma)
        if min_weight > weight:
            min_weight = weight
        if max_weight < weight:
            max_weight = weight
        extra_min.append((i , weight))
    for i in range(h*w):
        x, y = i // w, i % w
        nodes[i] = gray_img[x, y]
        if x > 0:
            j = (x - 1) * w + y
            weight = 1 - gaussian_similarity(gray_img[x, y], gray_img[x - 1, y], sigma)
            edges.append((i, j, weight))

        if y > 0:
            j = x * w + y - 1
            weight = 1 - gaussian_similarity(gray_img[x, y], gray_img[x, y - 1], sigma)
            edges.append((i, j, weight))


    # Initialize super_weights with default value of 0
    super_local_weights = defaultdict(float)
    n_local_weights = defaultdict(float)
    # Iterate over the quadratic keys
    for u, v, weight in edges:
        # Check if the elements belong to different clusters

        if clustered_image[u] != clustered_image[v]:
            i = clusters.index(clustered_image[u])
            j = clusters.index(clustered_image[v])
            n_local_weights[(i, j)] += 1
            # Update the weight in super_weights
            super_local_weights[(i, j)] += weight

    # Convert defaultdict back to a regular dictionary if needed
    super_local_weights = dict(super_local_weights)  
    n_local_weights = dict(n_local_weights)

    super_local_weights = {(i, j): super_local_weights[(i, j)] / n_local_weights[(i, j)] for i,j in super_local_weights.keys()}


    max_weight = max(np.max(list(super_local_weights.values())), max_weight )
    min_weight = min(np.min(list(super_local_weights.values())), min_weight )

    # return max_weight , min_weight
    a=-1
    b=1                
    if max_weight - min_weight:
        normalized_edges = [(node1,node2, -1 * np.round(((b-a)*((super_local_weights[(node1,node2)]-min_weight)/(max_weight-min_weight)))+a,4)) for node1,node2 in super_local_weights.keys()]
 
        normalized_extra_max = [(i, -1 * np.round(((1 - (-1)) * ((weight - min_weight) / (max_weight - min_weight))) + (-1), 4))
                            for i,weight in extra_max]
        normalized_extra_min = [(i, -1 * np.round(((1 - (-1)) * ((weight - min_weight) / (max_weight - min_weight))) + (-1), 4))
                            for i,weight in extra_min]
    elif max_weight == 0 and min_weight == 0:
        normalized_edges = [(node1,node2,1) for node1,node2 in super_local_weights.keys()]
        normalized_extra_max = [(i,1) for i,weight in extra_max]
        normalized_extra_min = [(i,1) for i,weight in extra_min]
    else:
        normalized_edges = [(node1,node2,-1 * np.round(super_local_weights[(node1,node2)],4)) for node1,node2 in super_local_weights.key()]
        normalized_extra_max = [(i, -1 * np.round(weight, 4)) for i,weight in extra_max]
        normalized_extra_min = [(i, -1 *  np.round(weight, 4)) for i,weight in extra_min]


    quadratic = {(iy,ix):-2*edge_weight for iy, ix,edge_weight in normalized_edges if abs(edge_weight)!=0}
    linear = {}
    for idx in range(len(clusters)):
          linear[idx] = - mu *( normalized_extra_max[idx][1] - normalized_extra_min[idx][1])
    for iy, ix,edge_weight in normalized_edges:
          linear[iy] += round(edge_weight,2)
          linear[ix] += round(edge_weight,2)
    return linear , quadratic  

def Global_Hamiltonian(gray_img,global_highest_value, global_lowest_value , sigma, mu, **kwargs):
    h, w = gray_img.shape
    nodes = np.zeros((h*w, 1))
    edges = []
    extra_max = []
    extra_min=[]
    min_weight = 1
    max_weight = 0
    # highest_intensity_node = np.argmax(gray_img)
    for i in range(h*w):
        x, y = i // w, i % w
        nodes[i] = gray_img[x, y]
        # if i != highest_intensity_node:
        weight = 1 - gaussian_similarity(global_highest_value,
                                         gray_img[x, y], sigma)
 
        extra_max.append((i , weight))
        if min_weight > weight:
            min_weight = weight
        if max_weight < weight:
            max_weight = weight
        weight = 1 - gaussian_similarity(global_lowest_value,
                                         gray_img[x, y], sigma)
 
        extra_min.append((i , weight))
        if min_weight > weight:
            min_weight = weight
        if max_weight < weight:
            max_weight = weight
        if x > 0:
            j = (x - 1) * w + y
            weight = 1 - gaussian_similarity(gray_img[x, y], gray_img[x - 1, y], sigma)
            edges.append((i, j, weight))
            if min_weight > weight:
                min_weight = weight
            if max_weight < weight:
                max_weight = weight
        if y > 0:
            j = x * w + y - 1
            weight = 1 - gaussian_similarity(gray_img[x, y], gray_img[x, y - 1], sigma)
            edges.append((i, j, weight))
            if min_weight > weight:
                min_weight = weight
            if max_weight < weight:
                max_weight = weight
    a=-1
    b=1                
    if max_weight - min_weight:
        normalized_edges = [(node1,node2,-1*np.round(((b-a)*((edge_weight-min_weight)/(max_weight-min_weight)))+a,4)) for node1,node2,edge_weight in edges]
 
        normalized_extra_max = [(i, -1 * np.round(((1 - (-1)) * ((weight - min_weight) / (max_weight - min_weight))) + (-1), 4))
                            for i,weight in extra_max]
        normalized_extra_min = [(i, -1 * np.round(((1 - (-1)) * ((weight - min_weight) / (max_weight - min_weight))) + (-1), 4))
                            for i,weight in extra_min]
    elif max_weight == 0 and min_weight == 0:
        normalized_edges = [(node1,node2,1) for node1,node2,edge_weight in edges]
        normalized_extra_max = [(i,1) for i,weight in extra_max]
        normalized_extra_min = [(i,1) for i,weight in extra_min]
    else:
        normalized_edges = [(node1,node2,-1*np.round(edge_weight,4)) for node1,node2,edge_weight in edges]
        normalized_extra_max = [(i,-1 * np.round(weight, 4)) for i,weight in extra_max]
        normalized_extra_min = [(i,-1 * np.round(weight, 4)) for i,weight in extra_min]



    quadratic = {(iy,ix):-2*edge_weight for iy, ix,edge_weight in normalized_edges if abs(edge_weight)!=0}
    linear = {}
    for idx in range(len(nodes)):
          linear[idx] = - mu *( normalized_extra_max[idx][1] - normalized_extra_min[idx][1])
    for iy, ix,edge_weight in normalized_edges:
          linear[iy] += round(edge_weight,2)
          linear[ix] += round(edge_weight,2)
    return linear , quadratic


def gaussian_similarity(a, b, sigma):
  """
  Calculate the Gaussian similarity score between two values.

  The Gaussian similarity function is often used in image processing and graph-based algorithms
  to measure how close or similar two values (like pixel intensities) are to each other.

  Parameters:
  a (float): The first value.
  b (float): The second value.
  sigma (float): The standard deviation used in the Gaussian function. This parameter controls
                 how quickly the similarity score decreases with the difference between a and b.

  Returns:
  float: The Gaussian similarity score between a and b.
  """
  gaussian_similairity_score = np.exp(-((a - b)**2) / (2 * sigma**2))
  return gaussian_similairity_score


def gaussian_similarity_2D(a, b, sigma_1,sigma_2):
  """
  Calculate the Gaussian similarity score between two vectors.

  The Gaussian similarity function is often used in image processing and graph-based algorithms
  to measure how close or similar two values (like pixel intensities) are to each other.

  Parameters:
  a (float): The first 2D vector.
  b (float): The second 2D vector.
  sigma_i (float): The standard deviation used in the Gaussian function on the each vector. This parameter controls
                 how quickly the similarity score decreases with the difference between a and b.

  Returns:
  float: The Gaussian similarity score between a and b.
  """
  gaussian_similairity_score = np.exp(-(((a[0] - b[0])**2)/ (2 * sigma_1**2)) + (((a[1] - b[1])**2)/ (2 * sigma_2**2)))
  return gaussian_similairity_score


def image_to_grid_graph(gray_img, sigma=0.5):
  """
  Convert a grayscale image to a grid graph with Gaussian similarity as edge weights.

  Parameters:
  gray_img (numpy.ndarray): Grayscale image.
  sigma (float): Parameter for Gaussian similarity.

  Returns:
  list: List of edges with weights for the graph.
  """
  h, w = gray_img.shape
  nodes = np.zeros((h*w, 1))
  edges = []
  nx_elist = []
  min_weight = 1
  max_weight = 0
  for i in range(h*w):
    x, y = i//w, i%w
    nodes[i] = gray_img[x,y]
    if x > 0:
      j = (x-1)*w + y
      weight = 1-gaussian_similarity(gray_img[x,y], gray_img[x-1,y], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x-1,y),np.round(weight,2)))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight
    if y > 0:
      j = x*w + y-1
      weight = 1-gaussian_similarity(gray_img[x,y], gray_img[x,y-1], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x,y-1),weight))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight
  a=-1
  b=1
  if max_weight-min_weight: 
    normalized_nx_elist = [(node1,node2,-1*np.round(((b-a)*((edge_weight-min_weight)/(max_weight-min_weight)))+a,4)) for node1,node2,edge_weight in nx_elist]
  elif max_weight==0 and min_weight==0:
    normalized_nx_elist = [(node1,node2,1) for node1,node2,edge_weight in nx_elist]
  else:
    normalized_nx_elist = [(node1,node2,-1*np.round(edge_weight,4)) for node1,node2,edge_weight in nx_elist]
  return normalized_nx_elist

""" 2D grid in case you want to use 2D feature space """

def image_to_grid_graph_2D(gray_img, sigma_1 = 0.5 ,sigma_2=0.5):
  """
  Convert a grayscale image to a grid graph with Gaussian similarity as edge weights.

  Parameters:
  gray_img (numpy.ndarray): Grayscale image.
  sigma (float): Parameter for Gaussian similarity.

  Returns:
  list: List of edges with weights for the graph.
  """
  h, w, _ = gray_img.shape
  nodes = np.zeros((h*w, 1))
  edges = []
  nx_elist = []
  min_weight = 1
  max_weight = 0
  for i in range(h*w):
    x, y = i//w, i%w
    nodes[i] = gray_img[x,y][0]
    if x > 0:
      j = (x-1)*w + y
      weight = 1-gaussian_similarity_2D(gray_img[x,y], gray_img[x-1,y], sigma_1,sigma_2)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x-1,y),np.round(weight,2)))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight
    if y > 0:
      j = x*w + y-1
      weight = 1-gaussian_similarity_2D(gray_img[x,y], gray_img[x,y-1], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x,y-1),weight))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight
  a=-1
  b=1
    # normalizing weights
  if max_weight-min_weight: 
    normalized_nx_elist = [(node1,node2,-1*np.round(((b-a)*((edge_weight-min_weight)/(max_weight-min_weight)))+a,4)) for node1,node2,edge_weight in nx_elist]
  elif max_weight==0 and min_weight==0:
    normalized_nx_elist = [(node1,node2,1) for node1,node2,edge_weight in nx_elist]
  else:
    normalized_nx_elist = [(node1,node2,-1*np.round(edge_weight,4)) for node1,node2,edge_weight in nx_elist]
  return normalized_nx_elist


""" considering more neighbors / diagonal interactions """

def image_to_grid_graph_diag(gray_img, sigma=0.5):
  """
  Convert a grayscale image to a grid graph with Gaussian similarity as edge weights.

  Parameters:
  gray_img (numpy.ndarray): Grayscale image.
  sigma (float): Parameter for Gaussian similarity.

  Returns:
  list: List of edges with weights for the graph.
  """
  h, w = gray_img.shape
  nodes = np.zeros((h*w, 1))
  edges = []
  nx_elist = []
  min_weight = 1
  max_weight = 0
  for i in range(h*w):
    x, y = i//w, i%w
    nodes[i] = gray_img[x,y]
    if x > 0:
      j = (x-1)*w + y
      weight = 1-gaussian_similarity(gray_img[x,y], gray_img[x-1,y], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x-1,y),np.round(weight,2)))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight
    if y > 0:
      j = x*w + y-1
      weight = 1-gaussian_similarity(gray_img[x,y], gray_img[x,y-1], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x,y-1),weight))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight
    if y > 0 and x > 0 :
      j = x*w + y-1
      weight = 1-gaussian_similarity(gray_img[x,y], gray_img[x-1,y-1], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x-1,y-1),weight))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight
    if y > 0 and x < h-1 :
      j = x*w + y-1
      weight = 1-gaussian_similarity(gray_img[x,y], gray_img[x+1,y-1], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x+1,y-1),weight))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight        
    if y > 1 and x > 1 :
      j = x*w + y-1
      weight = 1-gaussian_similarity(gray_img[x,y], gray_img[x-2,y-2], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x-1,y-1),weight))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight
    if y > 1 and x < h-2 :
      j = x*w + y-1
      weight = 1-gaussian_similarity(gray_img[x,y], gray_img[x+2,y-2], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x+1,y-1),weight))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight             
  a=-1
  b=1
  if max_weight-min_weight: 
    normalized_nx_elist = [(node1,node2,-1*np.round(((b-a)*((edge_weight-min_weight)/(max_weight-min_weight)))+a,4)) for node1,node2,edge_weight in nx_elist]
  elif max_weight==0 and min_weight==0:
    normalized_nx_elist = [(node1,node2,1) for node1,node2,edge_weight in nx_elist]
  else:
    normalized_nx_elist = [(node1,node2,-1*np.round(edge_weight,4)) for node1,node2,edge_weight in nx_elist]
  return normalized_nx_elist


""" some far neighbors / trying to catch non-local information / 2nd neighbors """

def image_to_grid_graph_plus(gray_img, sigma=0.5):
  """
  Convert a grayscale image to a grid graph with Gaussian similarity as edge weights.

  Parameters:
  gray_img (numpy.ndarray): Grayscale image.
  sigma (float): Parameter for Gaussian similarity.

  Returns:
  list: List of edges with weights for the graph.
  """
  h, w = gray_img.shape
  nodes = np.zeros((h*w, 1))
  edges = []
  nx_elist = []
  min_weight = 1
  max_weight = 0
  for i in range(h*w):
    x, y = i//w, i%w
    nodes[i] = gray_img[x,y]
    if x > 0:
      j = (x-1)*w + y
      weight = 1-gaussian_similarity(gray_img[x,y], gray_img[x-1,y], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x-1,y),np.round(weight,2)))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight
    if y > 0:
      j = x*w + y-1
      weight = 1-gaussian_similarity(gray_img[x,y], gray_img[x,y-1], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x,y-1),weight))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight
    if  x > 1 :
      j = x*w + y-1
      weight = 1-gaussian_similarity(gray_img[x,y], gray_img[x-2,y], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x-2,y),weight))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight
    if y > 2 :
      j = x*w + y-1
      weight = 1-gaussian_similarity(gray_img[x,y], gray_img[x,y-2], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x,y-2),weight))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight          
  a=-1
  b=1
  if max_weight-min_weight: 
    normalized_nx_elist = [(node1,node2,-1*np.round(((b-a)*((edge_weight-min_weight)/(max_weight-min_weight)))+a,4)) for node1,node2,edge_weight in nx_elist]
  elif max_weight==0 and min_weight==0:
    normalized_nx_elist = [(node1,node2,1) for node1,node2,edge_weight in nx_elist]
  else:
    normalized_nx_elist = [(node1,node2,-1*np.round(edge_weight,4)) for node1,node2,edge_weight in nx_elist]
  return normalized_nx_elist

""" some far neighbors / trying to catch non-local information / 3rd neighbors """

def image_to_grid_graph_plus_plus(gray_img, sigma=0.5):
  """
  Convert a grayscale image to a grid graph with Gaussian similarity as edge weights.

  Parameters:
  gray_img (numpy.ndarray): Grayscale image.
  sigma (float): Parameter for Gaussian similarity.

  Returns:
  list: List of edges with weights for the graph.
  """
  h, w = gray_img.shape
  nodes = np.zeros((h*w, 1))
  edges = []
  nx_elist = []
  min_weight = 1
  max_weight = 0
  for i in range(h*w):
    x, y = i//w, i%w
    nodes[i] = gray_img[x,y]
    if x > 0:
      j = (x-1)*w + y
      weight = 1-gaussian_similarity(gray_img[x,y], gray_img[x-1,y], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x-1,y),np.round(weight,2)))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight
    if y > 0:
      j = x*w + y-1
      weight = 1-gaussian_similarity(gray_img[x,y], gray_img[x,y-1], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x,y-1),weight))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight
    if  x > 2 :
      j = x*w + y-1
      weight = 1-gaussian_similarity(gray_img[x,y], gray_img[x-3,y], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x-3,y),weight))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight
    if y > 2 :
      j = x*w + y-1
      weight = 1-gaussian_similarity(gray_img[x,y], gray_img[x,y-3], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x,y-3),weight))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight  

    if y > 2 and x > 2 :
      j = x*w + y-1
      weight = 1-gaussian_similarity(gray_img[x,y], gray_img[x-3,y-3], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x-3,y-3),weight))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight
    if y > 2 and x < h-3 :
      j = x*w + y-1
      weight = 1-gaussian_similarity(gray_img[x,y], gray_img[x+3,y-3], sigma)
      edges.append((i, j, weight))
      nx_elist.append(((x,y),(x+1,y-1),weight))
      if min_weight>weight:min_weight=weight
      if max_weight<weight:max_weight=weight       
  a=-1
  b=1
  if max_weight-min_weight: 
    normalized_nx_elist = [(node1,node2,-1*np.round(((b-a)*((edge_weight-min_weight)/(max_weight-min_weight)))+a,4)) for node1,node2,edge_weight in nx_elist]
  elif max_weight==0 and min_weight==0:
    normalized_nx_elist = [(node1,node2,1) for node1,node2,edge_weight in nx_elist]
  else:
    normalized_nx_elist = [(node1,node2,-1*np.round(edge_weight,4)) for node1,node2,edge_weight in nx_elist]
  return normalized_nx_elist


""" Highest value pixel fully connected with all others """

def image_to_grid_graph_oneFC(gray_img, sigma=0.5):
    """
    Convert a grayscale image to a grid graph with Gaussian similarity as edge weights,
    and make the node with the highest intensity fully connected with all others.

    Parameters:
    gray_img (numpy.ndarray): Grayscale image.
    sigma (float): Parameter for Gaussian similarity.

    Returns:
    list: List of edges with weights for the graph.
    """
    h, w = gray_img.shape
    nodes = np.zeros((h*w, 1))
    edges = []
    nx_elist = []
    min_weight = 1
    max_weight = 0
    highest_intensity_node = np.argmax(gray_img)
    
    for i in range(h*w):
        x, y = i // w, i % w
        nodes[i] = gray_img[x, y]
        
        if i != highest_intensity_node:
            weight = 1 - gaussian_similarity(gray_img[highest_intensity_node // w, highest_intensity_node % w],
                                             gray_img[x, y], sigma)
            edges.append((highest_intensity_node, i, weight))
            nx_elist.append(((highest_intensity_node // w, highest_intensity_node % w), (x, y), weight))
            if min_weight > weight:
                min_weight = weight
            if max_weight < weight:
                max_weight = weight

        if x > 0:
            j = (x - 1) * w + y
            weight = 1 - gaussian_similarity(gray_img[x, y], gray_img[x - 1, y], sigma)
            edges.append((i, j, weight))
            nx_elist.append(((x, y), (x - 1, y), weight))
            if min_weight > weight:
                min_weight = weight
            if max_weight < weight:
                max_weight = weight

        if y > 0:
            j = x * w + y - 1
            weight = 1 - gaussian_similarity(gray_img[x, y], gray_img[x, y - 1], sigma)
            edges.append((i, j, weight))
            nx_elist.append(((x, y), (x, y - 1), weight))
            if min_weight > weight:
                min_weight = weight
            if max_weight < weight:
                max_weight = weight

    if max_weight - min_weight:
        normalized_nx_elist = [(node1, node2, -1 * np.round(((1 - (-1)) * ((edge_weight - min_weight) / (max_weight - min_weight))) + (-1), 4))
                               for node1, node2, edge_weight in nx_elist]
    elif max_weight == 0 and min_weight == 0:
        normalized_nx_elist = [(node1, node2, 1) for node1, node2, edge_weight in nx_elist]
    else:
        normalized_nx_elist = [(node1, node2, -1 * np.round(edge_weight, 4)) for node1, node2, edge_weight in nx_elist]

    return normalized_nx_elist

""" fully connected graph """

def image_to_grid_graph_FC(gray_img, sigma=0.5):
    """
    Convert a grayscale image to a fully connected graph with Gaussian similarity as edge weights.

    Parameters:
    gray_img (numpy.ndarray): Grayscale image.
    sigma (float): Parameter for Gaussian similarity.

    Returns:
    list: List of edges with weights for the graph.
    """
    h, w = gray_img.shape
    nodes = np.zeros((h*w, 1))
    edges = []
    nx_elist = []
    min_weight = 1
    max_weight = 0
    for i in range(h * w):
        x1, y1 = i // w, i % w
        for j in range(i + 1, h * w):
            x2, y2 = j // w, j % w
            weight = 1 - gaussian_similarity(gray_img[x1, y1], gray_img[x2, y2], sigma)
            edges.append((i, j, weight))
            nx_elist.append(((x1, y1),(x2, y2),weight))

            if min_weight > weight:
                min_weight = weight
            if max_weight < weight:
                max_weight = weight     
    a=-1
    b=1
    if max_weight-min_weight: 
       normalized_nx_elist = [(node1,node2,-1*np.round(((b-a)*((edge_weight-min_weight)/(max_weight-min_weight)))+a,4)) for node1,node2,edge_weight in nx_elist]
    elif max_weight==0 and min_weight==0:
         normalized_nx_elist = [(node1,node2,1) for node1,node2,edge_weight in nx_elist]
    else:
             normalized_nx_elist = [(node1,node2,-1*np.round(edge_weight,4)) for node1,node2,edge_weight in nx_elist]
    return normalized_nx_elist


""" image to graph + calculating the weights of the fully connected fictive qubits """

def image_to_grid_graph_patches(gray_img, global_highest_value, global_lowest_value, sigma=0.5):
    """
    Convert a grayscale image to a grid graph with Gaussian similarity as edge weights,
    and make the node with the highest intensity fully connected with all others.

    Parameters:
    gray_img (numpy.ndarray): Grayscale image.
    sigma (float): Parameter for Gaussian similarity.
 
    Returns:
    list: List of edges with weights for the graph.
    """
    h, w = gray_img.shape
    nodes = np.zeros((h*w, 1))
    edges = []
    nx_elist = []
    extra = []
    extra_min=[]
    min_weight = 1
    max_weight = 0
    # highest_intensity_node = np.argmax(gray_img)
    for i in range(h*w):
        x, y = i // w, i % w
        nodes[i] = gray_img[x, y]
        # if i != highest_intensity_node:
        weight = 1 - gaussian_similarity(global_highest_value,
                                         gray_img[x, y], sigma)
        # edges.append((highest_intensity_node, i, weight))
        # nx_elist.append(((highest_intensity_node // w, highest_intensity_node % w), (x, y), weight))
        extra.append((i , weight))
        if min_weight > weight:
            min_weight = weight
        if max_weight < weight:
            max_weight = weight
       
        # if i != highest_intensity_node:
        weight = 1 - gaussian_similarity(global_lowest_value,
                                         gray_img[x, y], sigma)
        # edges.append((highest_intensity_node, i, weight))
        # nx_elist.append(((highest_intensity_node // w, highest_intensity_node % w), (x, y), weight))
        extra_min.append((i , weight))
        if min_weight > weight:
            min_weight = weight
        if max_weight < weight:
            max_weight = weight
        if x > 0:
            j = (x - 1) * w + y
            weight = 1 - gaussian_similarity(gray_img[x, y], gray_img[x - 1, y], sigma)
            edges.append((i, j, weight))
            nx_elist.append(((x, y), (x - 1, y), weight))
            if min_weight > weight:
                min_weight = weight
            if max_weight < weight:
                max_weight = weight
        if y > 0:
            j = x * w + y - 1
            weight = 1 - gaussian_similarity(gray_img[x, y], gray_img[x, y - 1], sigma)
            edges.append((i, j, weight))
            nx_elist.append(((x, y), (x, y - 1), weight))
            if min_weight > weight:
                min_weight = weight
            if max_weight < weight:
                max_weight = weight
    if max_weight - min_weight:
        normalized_nx_elist = [(node1, node2, -1 * np.round(((1 - (-1)) * ((edge_weight - min_weight) / (max_weight - min_weight))) + (-1), 4))
                               for node1, node2, edge_weight in nx_elist]
        normalized_extra = [(i, -1 * np.round(((1 - (-1)) * ((weight - min_weight) / (max_weight - min_weight))) + (-1), 4))
                            for i,weight in extra]
        normalized_extra_min = [(i, -1 * np.round(((1 - (-1)) * ((weight - min_weight) / (max_weight - min_weight))) + (-1), 4))
                            for i,weight in extra_min]
    elif max_weight == 0 and min_weight == 0:
        normalized_nx_elist = [(node1, node2, 1) for node1, node2, edge_weight in nx_elist]
        normalized_extra = [(i,1) for i,weight in extra]
        normalized_extra_min = [(i,1) for i,weight in extra_min]
    else:
        normalized_nx_elist = [(node1, node2, -1 * np.round(edge_weight, 4)) for node1, node2, edge_weight in nx_elist]
        normalized_extra = [(i,-1 * np.round(weight, 4)) for i,weight in extra]
        normalized_extra_min = [(i,-1 * np.round(weight, 4)) for i,weight in extra_min]
 
    return normalized_nx_elist, normalized_extra, normalized_extra_min

""" same but """
def image_to_grid_graph_patches_2D(gray_img, global_highest_value, global_lowest_value, sigma_1 = 0.5,sigma_2 = 0.5):
    """
    Convert a grayscale image to a grid graph with Gaussian similarity as edge weights,
    and make the node with the highest intensity fully connected with all others.
 
    Parameters:
    gray_img (numpy.ndarray): Grayscale image.
    sigma (float): Parameter for Gaussian similarity.
 
    Returns:
    list: List of edges with weights for the graph.
    """
    h, w , _= gray_img.shape
    nodes = np.zeros((h*w, 1))
    edges = []
    nx_elist = []
    extra = []
    extra_min=[]
    min_weight = 1
    max_weight = 0
    # highest_intensity_node = np.argmax(gray_img)
    for i in range(h*w):
        x, y = i // w, i % w
        nodes[i] = gray_img[x, y][0]
        # if i != highest_intensity_node:
        weight = 1 - gaussian_similarity_2D([global_highest_value,global_highest_value],
                                         gray_img[x, y], sigma_1,sigma_2)
        # edges.append((highest_intensity_node, i, weight))
        # nx_elist.append(((highest_intensity_node // w, highest_intensity_node % w), (x, y), weight))
        extra.append((i , weight))
        if min_weight > weight:
            min_weight = weight
        if max_weight < weight:
            max_weight = weight
       
        # if i != highest_intensity_node:
        weight = 1 - gaussian_similarity_2D([global_lowest_value,global_lowest_value],
                                         gray_img[x, y], sigma_1,sigma_2)
        # edges.append((highest_intensity_node, i, weight))
        # nx_elist.append(((highest_intensity_node // w, highest_intensity_node % w), (x, y), weight))
        extra_min.append((i , weight))
        if min_weight > weight:
            min_weight = weight
        if max_weight < weight:
            max_weight = weight
        if x > 0:
            j = (x - 1) * w + y
            weight = 1 - gaussian_similarity_2D(gray_img[x, y], gray_img[x - 1, y], sigma_1,sigma_2)
            edges.append((i, j, weight))
            nx_elist.append(((x, y), (x - 1, y), weight))
            if min_weight > weight:
                min_weight = weight
            if max_weight < weight:
                max_weight = weight
        if y > 0:
            j = x * w + y - 1
            weight = 1 - gaussian_similarity_2D(gray_img[x, y], gray_img[x, y - 1], sigma_1,sigma_2)
            edges.append((i, j, weight))
            nx_elist.append(((x, y), (x, y - 1), weight))
            if min_weight > weight:
                min_weight = weight
            if max_weight < weight:
                max_weight = weight
                
    if max_weight - min_weight:
        normalized_nx_elist = [(node1, node2, -1 * np.round(((1 - (-1)) * ((edge_weight - min_weight) / (max_weight - min_weight))) + (-1), 4))
                               for node1, node2, edge_weight in nx_elist]
        normalized_extra = [(i, -1 * np.round(((1 - (-1)) * ((weight - min_weight) / (max_weight - min_weight))) + (-1), 4))
                            for i,weight in extra]
        normalized_extra_min = [(i, -1 * np.round(((1 - (-1)) * ((weight - min_weight) / (max_weight - min_weight))) + (-1), 4))
                            for i,weight in extra_min]
    elif max_weight == 0 and min_weight == 0:
        normalized_nx_elist = [(node1, node2, 1) for node1, node2, edge_weight in nx_elist]
        normalized_extra = [(i,1) for i,weight in extra]
        normalized_extra_min = [(i,1) for i,weight in extra_min]
    else:
        normalized_nx_elist = [(node1, node2, -1 * np.round(edge_weight, 4)) for node1, node2, edge_weight in nx_elist]
        normalized_extra = [(i,-1 * np.round(weight, 4)) for i,weight in extra]
        normalized_extra_min = [(i,-1 * np.round(weight, 4)) for i,weight in extra_min]
 
    return normalized_nx_elist, normalized_extra, normalized_extra_min


""" draw """

def draw(G, image):
  """
  Draw the graph G with the given image as node colors.

  Parameters:
  G (networkx.Graph): Graph to be drawn.
  image (numpy.ndarray): Grayscale image for node colors.
  """
  pixel_values = image
  plt.figure(figsize=(min(12,2*image.shape[0]),min(12,2*image.shape[0])))
  default_axes = plt.axes(frameon=True)
  pos = {(x,y):(y,-x) for x,y in G.nodes()}
  nx.draw_networkx(G,
                  pos=pos,
                  node_color=1-pixel_values,
                  with_labels=True,
                  node_size=1200,
                  cmap=plt.cm.Greys,
                  alpha=0.5,
                  ax=default_axes)
  nodes = nx.draw_networkx_nodes(G, pos, node_color=1-pixel_values,
                  node_size=1200,
                  cmap=plt.cm.Greys)
  nodes.set_edgecolor('k')
  edge_labels = nx.get_edge_attributes(G, "weight")
  nx.draw_networkx_edge_labels(G,
                              pos=pos,
                             edge_labels=edge_labels)



def draw_graph_cut_edges(G, image, cut_edges):
  """
  Draw the graph G with the given image as node colors, with the cut edges depicted as red dashed lines.

  Parameters:
  G (networkx.Graph): Graph to be drawn.
  image (numpy.ndarray): Grayscale image for node colors.
  cut_edges (list): Each tuple in the list contains two nodes corresponds to an edge that is cut.
  """
  pixel_values = image
  plt.figure(figsize=(min(12,2*image.shape[0]),min(12,2*image.shape[0])))
  default_axes = plt.axes(frameon=True)
  pos = {(x,y):(y,-x) for x,y in G.nodes()}
  nx.draw_networkx(G,
                  pos=pos,
                  node_color=1-pixel_values,
                  with_labels=True,
                  node_size=1200,
                  cmap=plt.cm.Greys,
                  alpha=0.8,
                  ax=default_axes)
  nodes = nx.draw_networkx_nodes(G, pos, node_color=1-pixel_values,
                  node_size=1200,
                  cmap=plt.cm.Greys)
  nodes.set_edgecolor('k')
  nx.draw_networkx_edges(G,
                         pos=pos,
                         edgelist=cut_edges,
                         width=6,
                         alpha=0.5,
                         edge_color="r",
                         style="dashed")
  edge_labels = nx.get_edge_attributes(G, "weight")
  nx.draw_networkx_edge_labels(G,
                               pos=pos,
                               edge_labels=edge_labels)