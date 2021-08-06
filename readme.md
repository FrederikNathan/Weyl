# Weyl simulation


# Core:

##scripts 
###caltech_run_weyl.py
script for launching a weyl sweep on caltech cluster. 
Dependencies: wq


##modules 
###weyl_queue.py (wq)
module for adding to queue, reading from queue, and launching from queue. 
Dependencies: mem

### master_equation_multimode.py (mem)
module for launching weyl simulation. Combines time-domain and frequency domain solvers. 
Dependencies: wl,rgf,tds

### weyl_liouvillian.py (wl)
module for constructing the Weyl model and representing it in operator space.
Dependencies: vec

### vectorization.py (vec)
Module for vectorizing matrices

### recursive_greens_function.py (rgf)
Module for inverting Weyl liouvillian in frequency space, using recursive greens function method. 

### time_domain_solver.py (tds)
Module for solving weyl problem in time-domain
Dependencies: wl,so3

###so3
Moduele for representing weyl problem in SO(3) representation, and useful functions associated. Used for solving weyl in time-domain.



# Data processing
Plotting scripts are contained within data_analysis/ 

## Modules
###data_refiner.py (DP)
Module for saving raw data  into single npz file. Automatically updates when loaded

### data_interpolation.py
Module for interpolating raw data to grids.
Dependencis (DP)
### data_plotting.py (pl)
Module with useful plotting functions


## Scripts (in data_analysis_)
###polar_interpolation.py
Plot conversion per electron in a 2-d plane

###power_vs_tau.py
Plot conversion power vs. tau

###power_vs_mu.py
Plot conversion power vs. mu

###power_vs_kz.py
Plot conversion power vs. kz



# Queue setup
Queue generating scripts are found in queue_scripts/
##scripts 

### run_from_queuelist.py $queue_list $n_runs $series_length
script for launching a sweep over several parameters. The parameter sets are stored in $queue_list.
Submits $n_runs jobs to queue dispatcher, each run samples $series_length k-points 

###wakanda_run_weyl.py
script for launching a weyl sweep on nbi cluster. 
Dependencies: wq

###caltech_run_weyl.py
script for launching a weyl sweep on caltech cluster. 
Dependencies: wq

###queue_scripts/big_sweep.py
Script for generating "big" sweep of the whole fermi surface, with dense points around weyl point. 

### check_queue.py
Check status of queue

### clean_queue.py
Reset status of "running" entries in queue to "waiting" (if the runs were terminated prematurely)

##modules 
###weyl_queue.py (wq)
module for adding to queue, reading from queue, and launching from queue. 
Dependencies: mem

###queue_scripts/queue_generator.py
Module for generating useful grids of k-points ot be probed



#Support modules
Basic.py, Units.py



