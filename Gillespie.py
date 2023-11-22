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

def feedback():
  #Gillespie2(X, Y, tlen, alpha, beta, params, seed = None)
  tlen = 1000
  params = {
    'k_x': 1.125,
    'g_x': 1,
    'k_y': 1,
    'g_y': 1,
    'alpha': 0.25,
    'beta': 1e3
  }
  X0 = [10]
  Y0 =[10]
  (ts, X, Y) = Gillespie2(X0,Y0,500,params)
  plt.figure()
  plt.plot(ts, X, label='X')
  plt.plot(ts, Y, label='Y')
  #plt.plot(ts, zcounts, label='Z')
  plt.xlabel('Time')
  plt.ylabel('Count')
  titleStr = '#X0='+str(X[0])+', #Y0='+str(Y[0]) #+', #Z0='+str(y0[2])+', c='+str(c)
  plt.title(titleStr)
  plt.legend()
  plt.show()

def interpolate(plot_xs, gillespie_out):
  ts, res, _ = gillespie_out
  xcounts = res[:,0]
  interp = np.interp(plot_xs, ts, xcounts)
  interp.shape = (1, len(interp))
  return interp

def main():
  plt.figure()

  plt.xlabel('Time')
  plt.ylabel('Count')

  N = 10 # number of simulations to plot
  random.seed(1) # change this to vary the simulations
  seeds = random.sample(range(10000), k = N)
  tlen = 15 # total time (x limit of plot)

  # model parameters
  X_init = 1000
  k_x = 1
  g_x = 1
  beta = None #1000
  
  # smoothe
  plot_xs = np.arange(0, tlen, tlen/1000)
  concat = None
  
  interps = []

  for i,seed in enumerate(seeds):

    print(f"Running simple() {i=}, {seed=}", file=sys.stderr)
    interp = interpolate( plot_xs, simple(X_init = X_init, k_x= k_x, g_x = g_x, beta = beta, tlen=tlen, seed=seed))

    # plt.plot(plot_xs, interp, label=str(seed))
    interps.append(interp)
  

  concat = np.concatenate(interps, axis = 0)

  mean = np.mean(concat, axis=0)
  sd = np.std(concat, axis=0)
  for q in np.arange(0,1,.1):
    qvec = np.quantile(concat, q, axis=0)
    plt.plot(plot_xs, qvec, label="%.1f" % q)

  q0 = np.quantile(concat, 0, axis=0)
  q1 = np.quantile(concat, .25, axis=0)
  q2 = np.quantile(concat, .5, axis=0)
  q3 = np.quantile(concat, .75, axis=0)
  q4 = np.quantile(concat, 1, axis=0)
  # print(q0[-1],q1[-1],q2[-1],q3[-1],q4[-1])
  
  # plt.plot(plot_xs, q0, label="q0")
  # plt.plot(plot_xs, q1, label="q1")

  # plt.plot(plot_xs, q2, label="median")

  # plt.plot(plot_xs, q3, label="q3")
  # plt.plot(plot_xs, q4, label="q4")
  # plt.plot(plot_xs, mean, label="mean")
  # plt.plot(plot_xs, mean + 3 * sd, label="mean + 3sd")
  # plt.plot(plot_xs, mean - 3 * sd, label="mean - 3sd")
  

  print(f"{X_init=}, {k_x=}, {g_x=}, {beta=}, iqr={q3[-1]-q1[-1]}")
  iqr = round(q3[-1]-q1[-1],2)
  titleStr = f"{X_init=}, {k_x=}, {g_x=}, {beta=}, {iqr=}"
  #titleStr = f"{X_init=}, {k_x=}, {g_x=}"
  plt.title(titleStr)
  plt.legend()
  plt.ylim(bottom = 0)
  plt.show()
  
  #buffering_two_bodies()

main()