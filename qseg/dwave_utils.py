import time
import dimod
#qpu
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from qiskit_optimization.applications import Maxcut
import networkx as nx
import numpy as np
# Hybrid
from dwave.system.samplers import LeapHybridSampler

"""
  quantum processor unit QPU
"""

def dwave_solver(dwave_sampler, sampler, linear, quadratic, runs=10000, **kwargs):
 
  """
  Solve a binary quadratic model using D-Wave sampler.
 
  Parameters:
  dwave_sampler: qpu instance
  sampler: problem embedding
  linear (dict): Linear coefficients of the model.
  quadratic (dict): Quadratic coefficients of the model.
  private_token (str): API token for D-Wave.
  runs (int): Number of reads for the sampler.
 
  Returns:
  dimod.SampleSet: Sample set returned by D-Wave sampler.
  float: Response time.
  """
  vartype = dimod.BINARY
  bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype)
  start_time = time.time()
  sample_set = sampler.sample(bqm, num_reads=runs)
  response_time = time.time() - start_time
  return sample_set,  response_time



def annealer_solver(G, private_token, n_samples=2000, **kwargs):
  """
  Solve the Maxcut problem on graph G using a D-Wave annealer.

  Parameters:
  G (networkx.Graph): Graph for which Maxcut is to be solved.
  private_token (str): API token for D-Wave.
  n_samples (int): Number of samples to collect.

  Returns:
  pandas.DataFrame: Dataframe containing samples.
  dict: Dictionary containing information about execution times.
  """
  # linear and quadratic terms of the Hamiltonian  
  start_time = time.time()
  w = -1 * nx.adjacency_matrix(G).todense()
  max_cut = Maxcut(w)
  qp = max_cut.to_quadratic_program()
  linear = qp.objective.linear.coefficients.toarray(order=None, out=None)
  quadratic = qp.objective.quadratic.coefficients.toarray(order=None, out=None)
  linear = {int(idx):-round(value,2) for idx,value in enumerate(linear[0])}
  quadratic = {(int(iy),int(ix)):-quadratic[iy, ix] for iy, ix in np.ndindex(quadratic.shape) if iy<ix and abs(quadratic[iy, ix])!=0}
  problem_formulation_time = time.time() - start_time
  # quantum solver  
  sample_set, connection_time, embedding_time, response_time = dwave_solver(linear, quadratic, private_token, runs=n_samples)
  info_dict = sample_set.info['timing'].copy()

  start_time = time.time()
  samples_df = sample_set.to_pandas_dataframe() # samples into a dataframe
  sample_fetch_time = time.time() - start_time

  info_dict['problem_formulation_time'] = problem_formulation_time
  info_dict['connection_time'] = connection_time
  info_dict['embedding_time'] = embedding_time
  info_dict['response_time'] = response_time
  info_dict['sample_fetch_time'] = sample_fetch_time
  return samples_df, info_dict
    
"""
Solving Min-cut Hamiltonian on Hybrid solver 
Theoretically, there are no embedding issues with hybrid solvers, allowing us to work with larger patches. However, it's important not to overlook memory constraints, especially considering that weighted graphs can be quite heavy structures.
"""

def hybrid_solver(G, private_token, n_run=1, **kwargs):
  """"
  Parameters:
  G (networkx.Graph): Graph for which Maxcut is to be solved.
  private_token (str): API token for D-Wave.
  n_samples (int): Number of samples to collect.

  Returns:
  pandas.DataFrame: Dataframe containing samples.
  dict: Dictionary containing information about execution times.
  """
  start_time = time.time()
  w = -1 * nx.adjacency_matrix(G).todense()
  max_cut = Maxcut(w)
  qp = max_cut.to_quadratic_program()
  linear = qp.objective.linear.coefficients.toarray(order=None, out=None)
  quadratic = qp.objective.quadratic.coefficients.toarray(order=None, out=None)
  linear = {int(idx):-round(value,2) for idx,value in enumerate(linear[0])}
  quadratic = {(int(iy),int(ix)):-quadratic[iy, ix] for iy, ix in np.ndindex(quadratic.shape) if iy<ix and abs(quadratic[iy, ix])!=0}
  problem_formulation_time = time.time() - start_time
  
  vartype = dimod.BINARY
  bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype)

  start_time = time.time()
  sampler=LeapHybridSampler(token = private_token)
  connection_time = time.time() - start_time

  start_time = time.time()
  sample_set = sampler.sample(bqm)
  response_time = time.time() - start_time
  
  start_time = time.time()
  samples_df = sample_set.to_pandas_dataframe()
  sample_fetch_time = time.time() - start_time

  info_dict = {}
  info_dict['problem_formulation_time'] = problem_formulation_time
  info_dict['connection_time'] = connection_time
  info_dict['embedding_time'] = None
  info_dict['response_time'] = response_time
  info_dict['sample_fetch_time'] = sample_fetch_time

  return samples_df, info_dict

"""
Initial Attempt to Incorporate Spectral Information
In our first attempt, we aim to enhance the problem formulation by introducing a penalty term. This penalty term is designed to favor pixels with intensities above a specified threshold to be classified as water pixels. Conversely, it penalizes pixels with lower intensities that are labeled as water.
s is a binary mask indicating whether the intensity of each pixel is above the specified threshold or not.
"""

def hybrid_solver_Global_Hamiltonian(G, s,lambda_, private_token, n_run=1, **kwargs):
  """"
  Parameters:
  G (networkx.Graph): Graph for which Maxcut is to be solved.
  private_token (str): API token for D-Wave.
  n_samples (int): Number of samples to collect.

  Returns:
  pandas.DataFrame: Dataframe containing samples.
  dict: Dictionary containing information about execution times.
  """
  start_time = time.time()
  # linear and quadratic terms of the Hamiltonian  
  w = -1 * nx.adjacency_matrix(G).todense()
  max_cut = Maxcut(w)
  qp = max_cut.to_quadratic_program()
  linear = qp.objective.linear.coefficients.toarray(order=None, out=None)
  quadratic = qp.objective.quadratic.coefficients.toarray(order=None, out=None)
  linear = {int(idx):-round(value,2) for idx,value in enumerate(linear[0])}
  quadratic = {(int(iy),int(ix)):-quadratic[iy, ix] for iy, ix in np.ndindex(quadratic.shape) if iy<ix and abs(quadratic[iy, ix])!=0}
    
  new_quadratic = {}
  spectral_linear = {}

  for idx, value in linear.items():
      spectral_linear[idx] = value - lambda_ * (-2*s[idx]+1)  
    

                
  problem_formulation_time = time.time() - start_time
  
  vartype = dimod.BINARY
   # QUBO 
  bqm = dimod.BinaryQuadraticModel(spectral_linear, quadratic, 0.0, vartype) 

  start_time = time.time()
  sampler=LeapHybridSampler(token = private_token) #hybrid solver
  connection_time = time.time() - start_time

  start_time = time.time()
  sample_set = sampler.sample(bqm)
  response_time = time.time() - start_time
  
  start_time = time.time()
  samples_df = sample_set.to_pandas_dataframe()
  sample_fetch_time = time.time() - start_time

  info_dict = {}
  info_dict['problem_formulation_time'] = problem_formulation_time
  info_dict['connection_time'] = connection_time
  info_dict['embedding_time'] = None
  info_dict['response_time'] = response_time
  info_dict['sample_fetch_time'] = sample_fetch_time

  return samples_df, info_dict

### THALES'S WORK: Markivian random fields



### OURS

"""
this is our most mature solution presented in the innovation lab with ALLIANZ. A more refined way to define the penalty term that will enhance the min-cut term.
Similarily to Thales's Hamiltonian, we'll define 2 auxiliary qubits to represent the different classes. However, the auxiliary qubits will be fictive. In other words, we'll define an equivalent mathematical penalty without physical variables. The 2nd main difference, as we want to conserve an unsupervised framework, the weights our the different terms of the hamiltonian will be computed via a similarity metric and no learning phase is needed.
"""


def annealer_solver_patches(G, penalty_weights_max, penalty_weights_min, mu ,dwave_sampler, n_samples=2000, **kwargs):
# def annealer_solver_patches(G, penalty_weights_max, penalty_weights_min, mu ,private_token, n_samples=2000, **kwargs):

  """
  Solve the Maxcut problem on graph G using a D-Wave annealer.
 
  Parameters:
  G (networkx.Graph): Graph for which Maxcut is to be solved.
  private_token (str): API token for D-Wave.
  n_samples (int): Number of samples to collect.
 
  Returns:
  pandas.DataFrame: Dataframe containing samples.
  dict: Dictionary containing information about execution times.
  """
      # linear and quadratic terms of the Hamiltonian  

  start_time = time.time()
  w = -1 * nx.adjacency_matrix(G).todense()
  max_cut = Maxcut(w)
  qp = max_cut.to_quadratic_program()
  linear = qp.objective.linear.coefficients.toarray(order=None, out=None)
  quadratic = qp.objective.quadratic.coefficients.toarray(order=None, out=None)
  linear = {int(idx):-round(value,2) for idx,value in enumerate(linear[0])}
  quadratic = {(int(iy),int(ix)):-quadratic[iy, ix] for iy, ix in np.ndindex(quadratic.shape) if iy<ix and abs(quadratic[iy, ix])!=0}

    # adding penalty terms
  penalty_linear = {}

  for idx, value in linear.items():
       penalty_linear[idx] = value - mu *( penalty_weights_max[idx][1]-penalty_weights_min[idx][1])
 
 
   
  problem_formulation_time = time.time() - start_time
  sample_set, connection_time, embedding_time, response_time = dwave_solver(dwave_sampler, penalty_linear, quadratic,  runs=n_samples) #solver 
  # sample_set, connection_time, embedding_time, response_time = dwave_solver(penalty_linear, quadratic,private_token,  runs=n_samples) #solver 

  info_dict = sample_set.info['timing'].copy()
 
  start_time = time.time()
  samples_df = sample_set.to_pandas_dataframe()
  sample_fetch_time = time.time() - start_time
 
  info_dict['problem_formulation_time'] = problem_formulation_time
  info_dict['connection_time'] = connection_time
  info_dict['embedding_time'] = embedding_time
  info_dict['response_time'] = response_time
  info_dict['sample_fetch_time'] = sample_fetch_time
  return samples_df, info_dict
    
"""
same thing but on hybrid solver.
"""
def hybrid_annealer_solver_patches(G, penalty_weights_max, penalty_weights_min, private_token,mu , n_samples=2000, **kwargs):
  """
  Solve the Maxcut problem on graph G using a D-Wave annealer.
 
  Parameters:
  G (networkx.Graph): Graph for which Maxcut is to be solved.
  private_token (str): API token for D-Wave.
  n_samples (int): Number of samples to collect.
 
  Returns:
  pandas.DataFrame: Dataframe containing samples.
  dict: Dictionary containing information about execution times.
  """
  start_time = time.time()
  w = -1 * nx.adjacency_matrix(G).todense()
  max_cut = Maxcut(w)
  qp = max_cut.to_quadratic_program()
  linear = qp.objective.linear.coefficients.toarray(order=None, out=None)
  quadratic = qp.objective.quadratic.coefficients.toarray(order=None, out=None)
  linear = {int(idx):-round(value,2) for idx,value in enumerate(linear[0])}
  quadratic = {(int(iy),int(ix)):-quadratic[iy, ix] for iy, ix in np.ndindex(quadratic.shape) if iy<ix and abs(quadratic[iy, ix])!=0}
 
  penalty_linear = {}
 
  for idx, value in linear.items():
       penalty_linear[idx] = value - mu *( penalty_weights_max[idx][1]-penalty_weights_min[idx][1])
 
 
   
  problem_formulation_time = time.time() - start_time



  vartype = dimod.BINARY
  bqm = dimod.BinaryQuadraticModel(penalty_linear, quadratic, 0.0, vartype)

  start_time = time.time()
  sampler=LeapHybridSampler(token = private_token)
  connection_time = time.time() - start_time

  start_time = time.time()
  sample_set = sampler.sample(bqm)
  response_time = time.time() - start_time
  
  start_time = time.time()
  samples_df = sample_set.to_pandas_dataframe()
  sample_fetch_time = time.time() - start_time  

  info_dict = {}
  info_dict['problem_formulation_time'] = problem_formulation_time
  info_dict['connection_time'] = connection_time
  info_dict['embedding_time'] = None
  info_dict['response_time'] = response_time
  info_dict['sample_fetch_time'] = sample_fetch_time
  return samples_df, info_dict
