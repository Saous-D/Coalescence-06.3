a remetre dans Coalescence-06.3
# *********************************


import gc
import tensorflow as tf
import numpy as np
import os
from keras.models import load_model
from Image_forgery_detection.ela import convert_to_ela_image
from scipy import stats
from dwave.system import LeapHybridSampler
from dwave.system import DWaveSampler, EmbeddingComposite
from dotenv import load_dotenv
from pathlib import Path
import time


private_token = os.getenv('PRIVATE_TOKEN')
if private_token:
    print(f"Private token reçu : {'*' * (len(private_token) - 3)}{private_token[-3:]}")
else:
    print("Aucun jeton privé n'a été reçu.")


#preparation of the ELA image and image pre-processing
def prepare_image(fname, n_pix_h, n_pix_v):
    image_size = (n_pix_h, n_pix_v)
    scale, ela_img=convert_to_ela_image(fname, 95) #90
    ela_image=np.array(ela_img.resize(image_size)).flatten()/scale #255.0
    return(scale, ela_image)
    # return ela_image as a numpy array


#supervised image classification based on ELA image and a training set contained in the file "trained_model.h5"
def predict_result(fname, n_pix_h, n_pix_v):
    model = load_model("Image_forgery_detection/trained_model.h5")  # load the trained model
    class_names = ["Falsified", "Authentic"]  # classification outputs

    n_pix_h, n_pix_v = 128, 128

    scale, test_image = prepare_image(fname,n_pix_h,n_pix_v)
    print("Tableau de l'image convertie en ELA:",test_image)
    folder_name = os.path.join(os.path.dirname(fname), "Images_infos")
    os.makedirs(folder_name, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(fname))[0] + '_ela_pixels.txt'
    print("Fichier contenant l'intensité ELA des pixels :",file_name)
    with open(folder_name+file_name, 'w') as f:
     
        n_px=len(test_image)
        f.write("Nombre de pixels :"+str(n_px))
        f.write("Intensité ELA :")
        for i in range(n_px):
            f.write(str(i)+"\t"+str(test_image[i])+"\n")
    test_image_d=test_image    
    test_image = test_image.reshape(-1, n_pix_h, n_pix_v, 3)
    
    try:
        os.remove(folder_name+file_name)
        print("Fichier supprimé avec succès.")
    except FileNotFoundError:
        print("Le fichier n'existe pas.")
    except PermissionError:
        print("Permission refusée.")
    except Exception as e:
        print(f"Erreur lors de la suppression : {e}")

    y_pred = model.predict(test_image)
    print("Keras prediction result :",y_pred)
    y_pred_class = round(y_pred[0][0])

    #prediction = class_names[y_pred_class]
    if class_names[y_pred_class]=="Authentic":
        prediction="Authentic"
    else:
        prediction="Forged"
    if y_pred <= 0.5:
        confidence = f"{(1-(y_pred[0][0])) * 100:0.2f}"
    else:
        confidence = f"{(y_pred[0][0]) * 100:0.2f}"
    #**********************************************************
    #*******Forcer Python à nettoyer les objets inutiles*******
    #**********************************************************
    del model
    gc.collect()
    tf.keras.backend.clear_session()
    #**********************************************************
    return (test_image, test_image_d, scale, prediction, confidence)

    
#computation of the QUBO matrix, useful for the quantum-annealing-based image segmentation    
def calc_QUBO_matrix(pixs_mat,x_opt, i_b, j_b, np_bh, np_bv):
    
    nq=np_bh*np_bv
    Q_b_mat=np.zeros((nq,nq), dtype=np.float64)
    
    for i in range(np_bh):
        for j in range(np_bv):
            Q_b_mat[i*np_bv+j][i*np_bv+j]=np.sign(pixs_mat[i_b*np_bh+i][j_b*np_bv+j])*(pixs_mat[i_b*np_bh+i][j_b*np_bv+j])**2.0
    
    for i in range(np_bh-1):
        for j in range(np_bv):
            Q_b_mat[i*np_bv+j][(i+1)*np_bv+j]=-1.0*pixs_mat[i_b*np_bh+i][j_b*np_bv+j]*pixs_mat[i_b*np_bh+i+1][j_b*np_bv+j]
            
    for i in range(np_bh):
        for j in range(np_bv-1):        
            Q_b_mat[i*np_bv+j][i*np_bv+j+1]=-1.0*pixs_mat[i_b*np_bh+i][j_b*np_bv+j]*pixs_mat[i_b*np_bh+i][j_b*np_bv+j+1]
    #coupling between image blocks
    if j_b>0:
        for i in range(np_bh):
            Q_b_mat[i*np_bv][i*np_bv]=Q_b_mat[i*np_bv][i*np_bv]-1.0*x_opt[(i+1)*np_bv-1]*pixs_mat[i_b*np_bh+i][j_b*np_bv]*pixs_mat[i_b*np_bh+i][(j_b-1)*np_bv]
    
    if i_b>0:
        for j in range(np_bv):
            Q_b_mat[j][j]=Q_b_mat[j][j]-1.0*x_opt[(np_bv-1)*np_bv-1+j]*pixs_mat[i_b*np_bh][j_b*np_bv+j]*pixs_mat[i_b*np_bh-1][j_b*np_bv+j]
    
    Q_b_dict={}
    Q_b_inds=[]
    for l in range(nq):
        for k in range(1,l+1):
            Q_b_dict[(k,l)]=Q_b_mat[k][l]
            Q_b_inds.append((k,l))
    return(Q_b_inds,Q_b_dict)
    
#save the segmented image for each of the three colors (Red, Green, Blue) of the RGB scale
def save_segm_image_col(fname, segm_image_col, color, n_pix_h, n_pix_v):
    folder_name=str(fname).split("'")[1].split("Images_essai")[0]+"Images_infos/"
    file_name=str(fname).split("/Images_essai/")[1].split(".")[0]+'_SegmentedELAImage_'+str(color)+'_pixels.txt'
    print("Fichier contenant l'image segmentée à partir des pixels de couleur "+str(color)+" de l'image ELA :",file_name)
    with open(folder_name+file_name, 'w') as f:
        for j in range(n_pix_v):
            for i in range(n_pix_h):
                f.write("("+str(i)+","+str(j)+")\t"+str(segm_image_col[i][j])+"\n")

#save the segmented image in a txt file
def save_segm_image(fname, segm_image, n_pix_h, n_pix_v):
    folder_name=str(fname).split("'")[1].split("Images_essai")[0]+"Images_infos/"
    file_name=str(fname).split("/Images_essai/")[1].split(".")[0]+'_SegmentedELAImage_pixels.txt'
    print("Fichier contenant l'image segmentée à partir de l'image ELA :",file_name)
    with open(folder_name+file_name, 'w') as f:
        for i in range(n_pix_h):
            for j in range(n_pix_v):
                f.write("("+str(i)+","+str(j)+")\t"+str(segm_image[i*n_pix_v+j])+"\n")

    
#find the forged region in the ELA-processed image   
def find_forged_region(fname, test_image,n_pix_h,n_pix_v):
    time_i=time.time()
    n_data=len(test_image)
    n_pixs=int(n_data/3)
    Red_pixs=np.zeros(n_pixs, dtype=np.float64)
    Green_pixs=np.zeros(n_pixs, dtype=np.float64)
    Blue_pixs=np.zeros(n_pixs, dtype=np.float64)
    for i in range(n_data):
        if (i%3==0):
            Red_pixs[int(i/3)]=test_image[i]
        elif (i%3==1):
            Green_pixs[int((i-1)/3)]=test_image[i]
        else:
            Blue_pixs[int((i-2)/3)]=test_image[i]
    Delta_e=np.unique(test_image)[1]
    e_mod_R=stats.mode(Red_pixs)[0]
    e_mod_G=stats.mode(Green_pixs)[0]
    e_mod_B=stats.mode(Blue_pixs)[0]
    for j in range(n_pixs):
        Red_pixs[j]=(-Red_pixs[j]+e_mod_R)/Delta_e
        Green_pixs[j]=(-Green_pixs[j]+e_mod_R)/Delta_e
        Blue_pixs[j]=(-Blue_pixs[j]+e_mod_R)/Delta_e
    
    Red_pixs_mat=Red_pixs.reshape(n_pix_h, n_pix_v, order='C')
    Green_pixs_mat=Green_pixs.reshape(n_pix_h, n_pix_v, order='C')
    Blue_pixs_mat=Blue_pixs.reshape(n_pix_h, n_pix_v, order='C')
    np_bh=32
    np_bv=32
    nbh=int(n_pix_h/np_bh)
    nbv=int(n_pix_v/np_bv)
    Colors=["Rouge","Vert","Bleu"]
    #creare una funzione ad hoc, evitando comandi ripetitivi
    segm_image_R=np.zeros((n_pix_h,n_pix_v), dtype=np.int64)
    segm_image_R=fill_segm_image(Colors[0], Red_pixs_mat, segm_image_R, np_bh, np_bv, nbh, nbv)
    # save_segm_image_col(fname,segm_image_R,Colors[0],n_pix_h,n_pix_v)       
    
    segm_image_G=np.zeros((n_pix_h,n_pix_v), dtype=np.int64)
    segm_image_G=fill_segm_image(Colors[1], Green_pixs_mat, segm_image_G, np_bh, np_bv, nbh, nbv)
    # save_segm_image_col(fname,segm_image_G,Colors[1],n_pix_h,n_pix_v)

    segm_image_B=np.zeros((n_pix_h,n_pix_v), dtype=np.int64)
    segm_image_B=fill_segm_image(Colors[2], Blue_pixs_mat, segm_image_B, np_bh, np_bv, nbh, nbv)
    # save_segm_image_col(fname,segm_image_B,Colors[2],n_pix_h,n_pix_v)
    
    segm_image=np.zeros((n_pixs), dtype=np.int64)         
    for i in range(n_pix_h):
        for j in range(n_pix_v):
            if (segm_image_R[i][j]==1) and (segm_image_G[i][j]==1) and (segm_image_B[i][j]==1):
                segm_image[i*n_pix_v+j]=1
    # save_segm_image(fname,segm_image,n_pix_h,n_pix_v)   
    
    time_f=time.time()
    duration=time_f-time_i
    print("Temps d'exécution de la routine d'identification des régions falsifiées :",duration," s")
    gc.collect() # Pour libérer les blocs mémoire temporaires après avoir manipuler les grosses matrices
    return(segm_image)
    

#get the best quantum solution among the Sampleset
def get_min_xt(best_sample,n_pixs_b):
    min_xt = np.zeros(n_pixs_b,dtype=np.float64)
    i=0
    for index, value in best_sample.items():
        min_xt[i] = value
        i=i+1
    return(min_xt)
  
#get the solution of the qubo problem based on quantum annealing. The chosen solver is LeapHybrid solver.
#The input image is a 32x32-pixel block of the ELA-processed image
def dwave_solve(Q_b_dict,Q_b_inds,n_pixs_b): 
    anneal_time=50
    n_reads=10000
    # load_dotenv(dotenv_path=Path('dwave_systems_key.env'))
    dwave_token = private_token
    sampler = LeapHybridSampler(solver={'category': 'hybrid'},token=dwave_token) #EmbeddingComposite(DWaveSampler(token=dwave_token, solver="Advantage_system6.4"))
    #LeapHybridSampler(solver={'category': 'hybrid'},token=dwave_token)  #does not support the num_reads
    sampleset=sampler.sample_qubo(Q_b_dict, label="ai_anomalie") #num_reads=n_reads, annealing_time=anneal_time
    #Errore : Sampleset has no attribute 'items'. What is it?
    print("Type de donnée de sampleset :",type(sampleset))
    print("Type de donnée de sampleset :",type(sampleset.first.sample))
    for index, value in sampleset.first.sample.items():
        print("Index :",index,"Value :",value,"of the pseudo-dictionary sampleset.first.sample")
    #Q_b_posInds=calc_pos_of_index(Q_b_inds) #it is a python dictionary 
    min_xt = get_min_xt(sampleset.first.sample,n_pixs_b) #sampleset.first.sample
    return(min_xt)

#starting from the solution of the qubo problem for each image block, this function allows to assemble each block into a 128x128-pixel image, for each RGB color.
def fill_segm_image(Color, col_pixs_mat, segm_image_col, np_bh, np_bv, nbh, nbv):
    x_opt=np.zeros((np_bh*np_bv),dtype=np.int64)
    for i_b in range(nbh):
        for j_b in range(nbv):
            print("Couleur : "+str(Color)+" - résolution avec D-Wave du problème QUBO relatif au bloc (",i_b,",",j_b,")")
            Q_b_inds,Q_b_dict=calc_QUBO_matrix(col_pixs_mat, x_opt, i_b, j_b, np_bh, np_bv)
            x_opt=dwave_solve(Q_b_dict,Q_b_inds,np_bh*np_bv)
            for i in range(i_b*np_bh,(i_b+1)*np_bh):
                for j in range(j_b*np_bv,(j_b+1)*np_bv):
                    segm_image_col[i][j]=x_opt[(i-i_b*np_bh)*np_bv+(j-j_b*np_bv)]
    return(segm_image_col)
