import matplotlib.pyplot as plt
import numpy as np
import random, sys
"""
modified from https://colab.research.google.com/drive/1Gz2WbAyXN5EJUZlmo6H3IN7NIhUByl4f
"""
def stabilization(x, gamma):
  return gamma / x

def repression(x, beta):
  return beta / (x + beta)

def activation(x, alpha):
  return x / (x + alpha)

"""
chatgpt, when asked to generate random variates of the time between negative binomial events with a given
overdispersion:

Generate the number of trials until r failures occur using a negative binomial distribution with parameters 
r and p.

Generate r overdispersion factors from a gamma distribution with shape parameter θ and scale parameter 
1/λ.

Generate r baseline waiting times between failures from an exponential distribution with rate parameter 
λ.

Multiply each baseline waiting time by the corresponding overdispersion factor.

Sum the adjusted waiting times to obtain the time until the r-th failure in the negative binomial process.

This procedure models the time between events in a negative binomial distribution while incorporating 
overdispersion with the parameter θ. Adjust the distribution parameters and sample size based on your 
specific requirements.
"""

import numpy as np

def generate_time_between_events_with_overdispersion(r, p, theta, rate, size):
  # Step 1: Generate number of trials until r failures
  trials_until_r_failures = np.random.negative_binomial(r, p, size)

  # Step 2: Generate overdispersion factors
  overdispersion_factors = np.random.gamma(shape=theta, scale=1/rate, size=(size, r))

  # Step 3: Generate baseline waiting times between failures
  baseline_waiting_times = np.random.exponential(scale=1/rate, size=(size, r))

  # Step 4: Multiply baseline waiting times by overdispersion factors
  adjusted_waiting_times = baseline_waiting_times * overdispersion_factors

  # Step 5: Sum adjusted waiting times to get time until r-th failure
  time_until_r_failures = np.sum(adjusted_waiting_times, axis=1)

  return trials_until_r_failures, time_until_r_failures

def exaple_generate_random_time_between_nbinom_with_overdispersion_theta():
  # Example usage
  r = 3
  p = 0.3
  theta = 2.0  # Overdispersion parameter
  rate = 0.2
  size = 1000

  trials, times_between_events = generate_time_between_events_with_overdispersion(r, p, theta, rate, size)

# You can now analyze the 'times_between_events' array, which contains the time between events with overdispersion



def Gillespie(N, V, y0, tlen, seed = None, spikes = []):
  '''N is the stoichiometry matrix
  rate function V. function that gives us the propensities as a function of the current state.
    V is the k reactivities or the equiibrium associations. Note V(y) gives the matrix we are looking for when starting the beginning of the algorithm.
  y0 is the initial condition/state vector.
  tlen is max length of time. We will build up to this.
  '''
  t = 0.0 #starting time
  ts = [0.0]  #reporting array, to save information
  y = np.copy(y0) #using the given initial condition
  res = [list(y)]  #lists because these will be dynamically resizing bcs we will be randomly choosing our time intervals. We could pre-allocate these in a np.array to make more efficient.
  random.seed(a=seed)
  i = 0
  status_msg = ""
  while True: #just continuously looping until there is a break from within

    """
    if i % 5e3 == 0:
      backspace = "\b" * len(status_msg)
      status_msg = f"Gillespie {i=}"
      print(backspace + status_msg, file=sys.stdout, flush = True, end="")
    """
    """
    for spiki in range(len(spikes)-1,-1,-1):
      if (abs(ts[-1] - spikes[spiki][0]) < .1):
        y[0] *= spikes[spiki][1]
        del spikes[spiki]
"""
    i += 1

    # prevent a runaway situation
    if sum(y) > 1e5:
      break

    prop = V(y) #propensities
    a0 = sum(prop) #to see total propensity
    if a0 == 0:
      break
    dt = random.expovariate(a0) #same thing as dt = np.random.exponential(1.0/a0)
    if not np.isscalar(dt):
      dt = dt[0]

    if t + dt > tlen: #if waiting time will exceed time limit
      break

    #picking with reaction to do
    idx = random.choices(population=range(len(prop)),  #population is the indexes of these propensities
                         weights = prop,   #propensities
                         k=1) #we only want one value to be picked because every time we execute a reaction, the propensities change

    #Pulling out the columns from the stoich matrix for all the specifes with respect to that reaction
    if N.ndim == 1:
      change_to_apply = N[idx]
    else:
      change_to_apply = N[:,idx] #idx applied to the state vector
    
    #need to re shape because it comes out as a 2D array
    change_to_apply.shape = len(change_to_apply) #converting to 1D array

    #Adding the time
    t += dt
    #How the state is going to change
    y+= change_to_apply #this is a np.array

    #saving the time and results so that we can use it later
    ts.append(t)
    res.append(list(y))
  print(file=sys.stdout)
  return(ts, np.array(res))

def buffering_two_bodies():
  stoich = np.array([[1,-1, 0, 0],
                     [0, 0, 1,-1]])

  k_x = 1.
  g_x = 1.
  k_y = 1.
  g_y = 1.
  f_kx = lambda x,y: x * k_x * activation(y, 1/4) #* repression(x, 25)
  f_gx = lambda x: x*g_x
  f_ky = lambda x,y: x * k_y * activation(y, 1/4) #* repression(x, 25)
  f_gy = lambda x: x*g_y
  prop_functions = lambda V: [f_kx(V[0], V[1]), f_gx(V[0]), f_ky(V[1],V[0]), f_gy(V[1])]
  X_init = 40
  Y_init = 40
  (ts, res) = Gillespie(stoich, prop_functions, np.array([X_init,Y_init]), 500, seed = None)
  xcounts = res[:,0]
  ycounts = res[:,1]
  
  plt.figure()
  plt.plot(ts, xcounts, label='X')
  plt.plot(ts, ycounts, label='Y')

  plt.xlabel('Time')
  plt.ylabel('Count')
  titleStr = f"#X0={X_init}, #{k_x=}, #{g_x=}"
  plt.title(titleStr)
  plt.legend()
  plt.show()

def two(X_init = 50, Y_init = 50, k_x = 1, g_x = 1, k_y = 1, g_y = 1, alpha = 50, beta = 50, tlen=50, seed=None):
  stoich = np.array([[1,-1, 0, 0],
                     [0, 0, 1, -1]])
  
  if alpha is not None:
    f_alpha = lambda x: activation(x, alpha)
  else: 
    f_alpha = lambda x: 1*x

  if beta is not None:
    f_beta = lambda x: repression(x, beta)
  else: 
    f_beta = lambda x: 1*x

  f_x = lambda x, y: x*k_x * f_alpha(y) * f_beta(x)
  f_y = lambda x, y: y*k_y * f_alpha(x) * f_beta(y)


  prop_functions = lambda y: [  f_x(y[0],y[1]), 
                                g_x * y[0],
                                f_y(y[0],y[1]),
                                g_y * y[1]]

  (ts, res) = Gillespie(stoich, prop_functions, [X_init, Y_init], tlen, seed = seed)
  return (ts, res, seed)

def main():
  X_init = 200
  Y_init = 200
  k_x = 1.1
  g_x = 1
  k_y = 1.1
  g_y = 1
  alpha = .01
  beta = 5000
  seed = 1
  tlen = 500
  plot_xs = np.arange(0, tlen, tlen/1000)

  gillespie_out = two(X_init = X_init, Y_init = Y_init, 
      k_x= k_x, g_x = g_x, 
      k_y= k_y, g_y = g_y, 
      alpha = alpha, beta=beta,
      tlen=tlen, seed=seed)  
  x, y = interpolate_two(plot_xs, gillespie_out)
  x.shape = plot_xs.shape
  y.shape = plot_xs.shape

  filename=f"Gillespie_sims/feedback/X{X_init}_kx{k_x}_Y{Y_init}_ky{k_y}_alpha{alpha}_beta{beta}_seed{seed}.png"

  plt.figure(figsize=((20,5)))
  plt.plot(plot_xs, x, label='X')
  plt.plot(plot_xs, y, label='Y')
  plt.ylim(bottom = 0)
  plt.xlabel('Time')
  plt.ylabel('Count')
  titleStr = f"{X_init=}, {k_x=}, {Y_init=}, {k_y=},{alpha=}, {beta=}, {seed=}"
  plt.title(titleStr)
  plt.legend()
  plt.savefig(filename)
  plt.show()
  
  plt.close()

  


def simple(X_init = 50, k_x = 1, g_x = 1, beta = 50, tlen=50, seed=None):
  stoich = np.array([1,-1])
  # X_init = 50
  # k_x = 1.5 
  # g_x = 1 

  if beta is not None:
    f_k = lambda x: x*k_x * repression(x, beta) 
  else:
    f_k = lambda x: x*k_x 


  f_g = lambda x: x*g_x
  prop_functions = lambda x: [f_k(x), f_g(x)]
  (ts, res) = Gillespie(stoich, prop_functions, [X_init], tlen, seed = seed)#, spikes=[(12.4,2), (25,.5)])
  xcounts = res[:,0]
  if False:
    plt.figure()
    plt.plot(ts, xcounts, label='X')
    plt.ylim(bottom = 0)
    plt.xlabel('Time')
    plt.ylabel('Count')
    titleStr = f"#X0={X_init}, #{k_x=}, #{g_x=}"
    plt.title(titleStr)
    plt.legend()
    plt.show()
  return (ts, res, seed)


def osc_V_yc(y,c):
    return np.array([c*y[0]*y[1], c*y[1]*y[2], c*y[2]*y[0]])

def oscillation():

  osc_N = np.array([[-1,0,1],
                    [1,-1,0],
                    [0,1,-1]])

  y0 = np.array([200, 100, 100])
  c = 0.05
  osc_V = lambda y: osc_V_yc(y,c)
  tlen = 2
  seed = 2
  (ts, res) = Gillespie(osc_N, osc_V, y0, tlen, seed)
  xcounts = res[:,0]
  ycounts = res[:,1]
  zcounts = res[:,2]
  
  plt.figure()
  plt.plot(ts, xcounts, label='X')
  plt.plot(ts, ycounts, label='Y')
  plt.plot(ts, zcounts, label='Z')
  plt.xlabel('Time')
  plt.ylabel('Count')
  titleStr = '#X0='+str(y0[0])+', #Y0='+str(y0[1])+', #Z0='+str(y0[2])+', c='+str(c)
  plt.title(titleStr)
  plt.legend()
  plt.show()

def interpolate(plot_xs, gillespie_out):
  ts, res, _ = gillespie_out
  xcounts = res[:,0]
  interp = np.interp(plot_xs, ts, xcounts)
  interp.shape = (1, len(interp))
  return interp

def interpolate_two(plot_xs, gillespie_out):
  ts, res, _ = gillespie_out
  xcounts = res[:,0]
  ycounts = res[:,1]
  interp_x = np.interp(plot_xs, ts, xcounts)
  interp_x.shape = (1, len(interp_x))
  interp_y = np.interp(plot_xs, ts, ycounts)
  interp_y.shape = (1, len(interp_y))
  return interp_x, interp_y

import os  # for getpid()
def launchfunc(argtuple):
  plot_xs, X_init, k_x, g_x, beta, tlen, seed, i = argtuple
  print(f"Running simple() on {os.getpid()}. {i=}, {seed=}", file=sys.stderr)
  return interpolate( plot_xs, simple(X_init = X_init, k_x= k_x, g_x = g_x, beta = beta, tlen=tlen, seed=seed))

def main_simple():
  fig, ax = plt.subplots(2)

  ax[0].set_xlabel('Time')
  ax[0].set_ylabel('Count')

  N = 1000 # number of simulations to aggregate
  random.seed(1) # change this to vary the simulations
  seeds = random.sample(range(10000), k = N)
  tlen = 50 # total time (x limit of plot)

  # model parameters
  X_init = 1000
  k_x = 1.5
  g_x = 1
  beta = 1000
  
  # smoothe
  plot_xs = np.arange(0, tlen, tlen/1000)
  concat = None
  
  import time
  start = time.time()
  
  MP = True
  if MP:
    import multiprocessing as mp
    with mp.Pool(processes = 6) as p:
      arglist = [(plot_xs, X_init, k_x, g_x, beta, tlen, seed, i) for (i,seed) in enumerate(seeds)]
      interps = p.map(launchfunc,arglist)
  else:
    interps = [launchfunc(plot_xs, X_init, k_x, g_x, beta, tlen, seed, i) for (i, seed) in enumerate(seeds)]

  end = time.time()
  print(f"elapsed: {end - start}")

  

  concat = np.concatenate(interps, axis = 0)

  mean = np.mean(concat, axis=0)
  sd = np.std(concat, axis=0)
  for q in np.arange(0,1.125,.125):
    qvec = np.quantile(concat, q, axis=0)
    ax[0].plot(plot_xs, qvec, label="%.2f" % q)

  q0 = np.quantile(concat, 0, axis=0)
  q1 = np.quantile(concat, .25, axis=0)
  q2 = np.quantile(concat, .5, axis=0)
  q3 = np.quantile(concat, .75, axis=0)
  q4 = np.quantile(concat, 1, axis=0)

  print(f"{X_init=}, {k_x=}, {g_x=}, {beta=}, iqr={q3[-1]-q1[-1]}")
  iqr_end = round(q3[-1]-q1[-1],2)
  
  titleStr = f"{X_init=}, {k_x=}, {g_x=}, {beta=}, {iqr_end=}"
  #titleStr = f"{X_init=}, {k_x=}, {g_x=}"
  ax[0].set(title=titleStr)
  ax[0].legend()
  bottom, top = ax[0].get_ylim()
  ax[0].set_ylim(0, top, auto=None)


  
  iqr_total = q3 - q1
  #plt.figure()
  ax[1].set(title="Increase in IQR over time")
  ax[1].set_xlabel('Time')
  ax[1].set_ylabel('IQR')
  ax[1].plot(plot_xs, iqr_total)
  plt.show()
  #buffering_two_bodies()

if __name__ == '__main__': main()