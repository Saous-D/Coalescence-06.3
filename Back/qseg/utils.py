import numpy as np
import cv2

def decode_binary_string(x, height, width):
    """
    Decode a binary string into a binary segmentation mask.
    Parameters:
    x (list): Binary string representing the segmentation.
    height (int): Height of the image.
    width (int): Width of the image.
    Returns:
    numpy.ndarray: Segmentation mask.
    """
    mask = np.zeros([height, width])
    for index, segment in enumerate(x):
        mask[index // width, index % width] = segment
    return mask
def decode_binary_string_multi(x,k, height, width):
    """
    Decode a binary string into a binary segmentation mask.
    Parameters:
    x (list): Binary string representing the segmentation.
    height (int): Height of the image.
    width (int): Width of the image.
    Returns:
    numpy.ndarray: Segmentation mask.
    """
    mask = np.zeros([height, width])
    for i in range(k-1):
            for index in range(height* width):
                if  x[index + i*height* width]!=0:
                        mask[index // width, index % width] = x[index + i*height* width] * (i+1)/(k-1)
    return mask

def generate_intervals(interval, k):
    start, end = interval
    if k == 1:
        return [start]
    step = (end - start) / (2 * k)
    return [start + step + 2* i * step for i in range(k)]
def optimal_feasible_solution(x_hat,N_nodes,k,quadratic):
    I=[]
    for v in range(N_nodes):
        clicks = 0
        for j in range(k-1):
            clicks += x_hat[j * N_nodes + v]
        if clicks > 1:
            I.append(v)
    x_bar = x_hat
    l = 1
    C = []
    E = []
    P = []
    while len(I)>0:
        a = I[0]
        for j in range(k-1):
            if x_hat[j * N_nodes + a] == 1:
                P.append(j)
            for u in I:
                if  x_hat[j * N_nodes + u] ==  x_hat[j * N_nodes + a]:
                    C.append(u)
        for edge in quadratic.keys():   
            if edge[0] in C and edge[1] in C :
                E.append(edge)
        min_j = -1  
        min_energy = float('inf')
        for j in P:
            energy = 0
            for edge in E:
                energy += quadratic[edge] * x_bar[j * N_nodes + edge[0]] * x_bar[j * N_nodes + edge[1]]
            if energy <    min_energy:
                min_j = j
        for    u in C:
            for j in P:
                if j != min_j:
                    x_bar[j * N_nodes + u] = 0
        I = [element for element in I if element not in C]           
        l+=1 
    return x_bar

def perform_kmeans(img, K=1024, attempts=10):
    # Reshape the image to a 2D array of pixels
    vectorized_img = img.reshape((-1, 1))
    # Convert to float32
    vectorized_img = np.float32(vectorized_img)
    # Normalize the pixel values to [0, 1]
    # vectorized_img = (vectorized_img + 1) / 2.0
    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(vectorized_img, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    # Rescale the centers back to the range [-1, 1]
    # center = center * 2.0 - 1.0
    # Map the labels to the center values
    res = center[label.flatten()]
    # Reshape the result back to the original image shape
    result_image = res.reshape((img.shape))
    return result_image,label, center
 
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
    full_size_label_image = np.zeros((h, w), dtype=np.float64)
 
    # Map each cluster to the corresponding superpixel label
    cluster_to_superpixel_label = {i: superpixel_labels[i] for i in range(len(superpixel_labels))}
 
    # Assign the superpixel label to each pixel in the full-size image based on cluster
    for y in range(h):
        for x in range(w):
            cluster_index = label[y * w + x]
            full_size_label_image[y, x] = cluster_to_superpixel_label[cluster_index]
 
    return full_size_label_image