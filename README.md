# hyperparam_ga
### Version 0.2
Hyperparam optimization by genetic algorihtm
Simple implementation of an genetic algo for evolve a set of hyperparams evaluated by a score function. 
# Usage for simple node
```
import hyperparam_ga as hga
ga.main.sample_nmist(10, 5, 5, 'sample.txt')
```

# Usage for parallel
## Create a file example.py
```
import hyperparam_ga as hga
ga.main.sample_nmist_parallel(10, 5, 5, 'sample.txt')
```
## Run with MPI
```
mpirun -np 10 python3 example.py
```
