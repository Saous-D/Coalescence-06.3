from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
# Importing functions from the modules in the qseg package
from qseg.graph_utils import super_Global_Hamiltonian, image_to_grid_graph, draw, draw_graph_cut_edges, image_to_grid_graph_diag,image_to_grid_graph_plus, image_to_grid_graph_FC,image_to_grid_graph_oneFC, image_to_grid_graph_patches,image_to_grid_graph_patches_2D, Global_Hamiltonian
from qseg.dwave_utils import dwave_solver, annealer_solver, hybrid_solver,hybrid_solver_Global_Hamiltonian, annealer_solver_patches, hybrid_annealer_solver_patches
from qseg.utils import decode_binary_string, ndwi, ndvi

# Additional necessary imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import networkx as nx
from qiskit_optimization.applications import Maxcut
import dimod
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import imageio
from sklearn.datasets import load_digits
from glob import glob
from dwave.system.samplers import LeapHybridSampler

from rasterio.plot import reshape_as_raster, reshape_as_image
from dotenv import load_dotenv
import os
import io

# Charger le fichier .env
load_dotenv()
private_token = os.getenv('PRIVATE_TOKEN')
if private_token:
    print(f"Private token reçu : {'*' * (len(private_token) - 3)}{private_token[-3:]}")
else:
    print("Aucun jeton privé n'a été reçu.")

dwave_sampler = DWaveSampler(token = private_token, solver={'topology__type': 'pegasus'})

def log(message):
    print(message)
    sys.stdout.flush()

no_water_thresh = -0.25
water_thresh = 0.5
vartype = dimod.BINARY
def threshold(array, inequality):
    result = inequality(array)
    return result.all()
water_inequality = lambda x: x > water_thresh 
no_water_inequality = lambda x: x < no_water_thresh
def extract_rgb_bands(image):
        
        img_data = imageio.imread(image)
        rgb_bands = [3, 2, 1]
        rgb_data = img_data[rgb_bands, :, :]
        rgb_data = rgb_data / np.max(rgb_data)
        rgb_data_reshaped = reshape_as_image(rgb_data)
        return rgb_data_reshaped

patch_size = 32
def binary_segmentation(image_path, image_height, image_width,
                        water_penalty, no_water_penalty,
                        sigma, mu, no_water_thresh, water_thresh, callback=None):
        #load image
        image = imageio.imread(image_path)
        #NDWI
        green = image[2:3,:,:]
        swir = image[11:12,:,:]
        NDWI = (green - swir)/(green+swir) 
        gray_image = NDWI.squeeze(0)
        # machine instanciation
        dwave_sampler = DWaveSampler(token = private_token, solver={'topology__type': 'pegasus'})
        sampler = EmbeddingComposite(dwave_sampler)
        segmentation_mask = np.zeros((image_height, image_width))
        for y in range(0, image_height, patch_size):
            for x in range(0, image_width, patch_size):
                # Extract 32x32 patch
                patch = gray_image[y:y+patch_size, x:x+patch_size]
                if threshold(patch, water_inequality):
                    segmentation_mask[y:y+patch_size, x:x+patch_size] = np.ones((patch_size, patch_size))                       
                elif     threshold(patch, no_water_inequality):
                    segmentation_mask[y:y+patch_size, x:x+patch_size] = np.zeros((patch_size, patch_size))
                else:    
                    linear, quadratic = Global_Hamiltonian(patch,global_highest_value = water_penalty,global_lowest_value = no_water_penalty , sigma = sigma, mu = mu)
                    sample_set, connection_time,  response_time = dwave_solver(dwave_sampler, sampler, linear, quadratic, runs=2000)
                    samples_dataframe = sample_set.to_pandas_dataframe() # samples into a dataframe
                    #segmentation
                    solution_binary_string_i = samples_dataframe.iloc[0][:-3]
                    segmentation_mask_i = decode_binary_string(solution_binary_string_i,32,32)        
                    segmentation_mask[y:y+patch_size, x:x+patch_size]= segmentation_mask_i
        return segmentation_mask

def super_segmentation(image_path, image_height, image_width,
                        water_penalty, no_water_penalty,
                        sigma, mu, k, callback=None):
        if callback:
            callback("loading...")
        #load image
        image = imageio.imread(image_path) 
        dwave_sampler = DWaveSampler(token = private_token, solver={'topology__type': 'pegasus'})     
        #NDWI
        green = image[2:3,:,:]
        swir = image[11:12,:,:]
        NDWI = (green - swir)/(green+swir) 
        img = NDWI.squeeze(0)
        if callback:
            callback("QUBO Formulation")
        # # QUBO FORMULATION        
        if callback:
            callback("creating superpixels")
        result_image, label, center = perform_kmeans(img,K = k)
        clustered_image = result_image.flatten()
        clusters = list(center.squeeze(1))
        water_penalty = np.max(clusters) # max(0.4, img.max()) ##### c problematique si y a une seule classe
        no_water_penalty= np.min(clusters) #min(-0.5, img.min())
        linear,quadratic = super_Global_Hamiltonian(img, clustered_image, clusters , global_highest_value = water_penalty, global_lowest_value = no_water_penalty , sigma = sigma, mu = mu)
        if callback:
            callback("send to dwave solver")
        sampler = EmbeddingComposite(dwave_sampler)
        sample_set, response_time = dwave_solver(dwave_sampler, sampler,linear, quadratic, runs=2000)
        samples_dataframe = sample_set.to_pandas_dataframe() # samples into a dataframe
        if callback:
            callback("segmentation") 
        #segmentation
        solution_binary_string = samples_dataframe.iloc[0][:-3]
        full_size_label_image = create_full_size_label_image(label.flatten(), solution_binary_string, img.shape)
        kernel_size = 3  # Adjust this based on the size of noise you want to remove
        if callback:
            callback("Apply median filtering")
        # Apply median filtering
        filtered_mask = cv2.medianBlur(full_size_label_image, kernel_size)
        return filtered_mask

def create_superpixel_image(center, img_shape, superpixel_size=(32, 32)):
    superpixel_h, superpixel_w = superpixel_size

    # Create a blank image for superpixels
    superpixel_image = np.zeros((superpixel_h, superpixel_w), dtype=np.float32)

    # Assign each superpixel to the corresponding cluster center
    for i in range(superpixel_h):
        for j in range(superpixel_w):
            cluster_index = i * superpixel_w + j
            superpixel_image[i, j] = center[cluster_index]

    return superpixel_image

def create_full_size_label_image(label, superpixel_labels, img_shape):
    h, w = img_shape

    # Create a blank image for the full-size labels
    full_size_label_image = np.zeros((h, w), dtype=np.uint8)

    # Map each cluster to the corresponding superpixel label
    cluster_to_superpixel_label = {i: superpixel_labels[i] for i in range(len(superpixel_labels))}

    # Assign the superpixel label to each pixel in the full-size image based on cluster
    for y in range(h):
        for x in range(w):
            cluster_index = label[y * w + x]
            full_size_label_image[y, x] = cluster_to_superpixel_label[cluster_index]
 
    return full_size_label_image

def perform_kmeans(img, K=2, attempts=10):
    # Reshape the image to a 2D array of pixels
    vectorized_img = img.reshape((-1, 1))
    # Convert to float32
    vectorized_img = np.float32(vectorized_img)
    # Normalize the pixel values to [0, 1]
    vectorized_img = (vectorized_img + 1) / 2.0
    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(vectorized_img, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    # Rescale the centers back to the range [-1, 1]
    center = center * 2.0 - 1.0
    # Map the labels to the center values
    res = center[label.flatten()]
    # Reshape the result back to the original image shape
    result_image = res.reshape((img.shape))
    return result_image,label, center



def return_heatmap(image_path, color="winter"):
    """
    Generates a heatmap (NDWI) from a TIFF image and returns it as a PNG image in memory.
    
    Parameters:
        image_path (str): Path to the input .tif file
        color (str): Matplotlib colormap to use (e.g. 'viridis', 'inferno', etc.)

    Returns:
        io.BytesIO: PNG image buffer containing the heatmap
    """
    # Load the image (assumed to be multi-band)
    image = imageio.imread(image_path)

    # Extract green and SWIR bands (adjust indices as needed)
    green = image[2, :, :]
    swir = image[11, :, :]

    # Compute NDWI (Normalized Difference Water Index)
    NDWI = (green - swir) / (green + swir + 1e-6) # Avoid division by zero

    # Plot NDWI heatmap
    fig, ax = plt.subplots()
    # interpolation='nearest' ensures sharp edges in pixelated data (no smoothing)
    cax = ax.imshow(NDWI, cmap=color, interpolation='nearest')  
    ax.axis('off')
    # Create colorbar
    cbar = fig.colorbar(cax)

    # Replace numeric ticks with custom labels (Dry at bottom, Water at top)
    cbar.set_ticks([NDWI.min(), NDWI.max()])
    cbar.set_ticklabels(["Dry", "Water"])


    plt.tight_layout(pad=0)

    # Save figure to in-memory PNG
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)

    return buf
 

def heatmap(image_path, type, color="winter_r"):
    """
    Generates a heatmap from a TIFF image and returns it as a PNG image in memory.

    Parameters:
        image_path (str): Path to the input .tif file
        (Optional) color (str): Matplotlib colormap to use (e.g. 'viridis', 'inferno', etc.)

    Returns:
        io.BytesIO: PNG image buffer containing the heatmap
    """

    if type == "water":
        index = ndwi(image_path)
    elif type == "vegetation":
        index = ndvi(image_path)
    else:
        raise ValueError("Invalid type. Use 'water' or 'vegetation'.")
    # Plot NDWI heatmap
    fig, ax = plt.subplots()
    # interpolation='nearest' ensures sharp edges in pixelated data (no smoothing)
    cax = ax.imshow(index, cmap=color, interpolation='nearest')  
    ax.axis('off')
    # Create colorbar
    cbar = fig.colorbar(cax)

    # Replace numeric ticks with custom labels (Dry at bottom, Water at top)
    cbar.set_ticks([index.min(), index.max()])
    cbar.set_ticklabels(["No " + type, type])
    plt.title(f"{type} heatmap")

    # Save figure to in-memory PNG
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return buf


def quantum_scan(image_path, type):
    """
    Perform quantum segmentation on the input image using D-Wave's quantum annealer.

    Parameters:
        image_path (str): Path to the input .tif file
        type (str): Type of segmentation ('water' or 'vegetation')
    Returns:
        np.ndarray: Segmented image
    """

    if type == "water":
        index = ndwi(image_path)
    elif type == "vegetation":
        index = ndvi(image_path)
    else:
        raise ValueError("Invalid type. Use 'water' or 'vegetation'.")
    
    """ performing clustering ; number of clusters = number of nodes in the hypergraph """
    result_image, label, centers = perform_kmeans(index,K = 16)

    """ Initialize sampler and embedding """
    dwave_sampler = DWaveSampler(token = private_token, solver={'topology__type': 'pegasus'})
    sampler = EmbeddingComposite(dwave_sampler)

    clustered_image = result_image.flatten()
    clusters = list(centers.squeeze(1))

    linear,quadratic = super_Global_Hamiltonian(index, clustered_image, clusters , global_highest_value = np.max(clusters), global_lowest_value = np.min(clusters) , sigma=0.2, mu=2)
    sample_set,  response_time = dwave_solver(dwave_sampler, sampler, linear, quadratic, runs=2000)
    samples_dataframe = sample_set.to_pandas_dataframe() # samples into a dataframe
    #segmentation
    solution_binary_string = samples_dataframe.iloc[0][:-3]

    full_size_label_image = create_full_size_label_image(label.flatten(), solution_binary_string , index.shape)

    return full_size_label_image

from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches



def overlay_masks(water_mask, vegetation_mask):
    """
    Overlay two binary masks (water and vegetation) on a single image.
    
    Parameters:
        water_mask (np.ndarray): Binary mask for water
        vegetation_mask (np.ndarray): Binary mask for vegetation

    Returns:
        png image: PNG image buffer containing the overlay
    """
    # Combine the masks
    combined_mask = water_mask + 2 * vegetation_mask

    # Create a color map
    colors = ["#653700", "blue", "green", "yellow"]
    cmap = ListedColormap(colors)

    # Display the combined mask
    plt.imshow(combined_mask, cmap=cmap, vmin=0, vmax=3)
    plt.axis('off')
    plt.title("Overlap of Water and Vegetation Segmentation")

    # Create legend patches
    legend_patches = [
        mpatches.Patch(color=colors[0], label='No Water &\nNo Vegetation'),
        mpatches.Patch(color=colors[1], label='Mask\nWater'),
        mpatches.Patch(color=colors[2], label='Mask\nVegetation'),
        mpatches.Patch(color=colors[3], label='Water &\nVegetation\nOverlap')
    ]

    # Display the legend
    plt.legend(handles=legend_patches, 
               loc='center left', 
               bbox_to_anchor=(1, 0.5),  # -> décalage horizontal à droite
               frameon=True)
    
    # Save figure to in-memory PNG
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    buf.seek(0)
    return buf