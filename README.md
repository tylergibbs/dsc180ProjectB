# dsc180ProjectB



# Comparisons of Causal Discovery Optimization Methods

__This project extrapolates 2 differetn methods for causal descovery into timeseries data and compares them to the effectivness of existing methods:

- GOLEM
- DAGMA

This project adapts existing implentations of GOLEM and DAGMA into timeseries data using the methods outlined in DYNOTEARS. We have a simple way to generate statistics on all 3 methods, repicates the metrics in the original papers, and compares their performance across consistent metrics. Finaly we make minor imporvments implementing an early stopping threashold in all 3. We expect DAGMA to perform the best, followed by GOLEM, followed by NOTEARS as this is the inverse order they were developed.

## Data:

We have two sources of data.

(1) Synthetic data prodiced by synthetic_dataset in data_loader/synthetic_dataset

(2) Data from the cdt library, 
	(a) Sacks Dataset
	(b) Tuebingen


## Running the project

* To install the dependencies, run the following command from the root directory of the project: `pip install -r requirements.txt`

### Viewing the project through jupyter notebook
* All data and graphs have been generated at run.ipynb. This is the simplest and fasted way to run the code.
* This replicates all analisys in the paper
  
### Building the project stages using `run.py`

* 'python run.py all' will run through and generate all the sections in run.ipynb in one script

* 'python run.py main' will run through examples of main.py 

* 'python run.py compare' will run a set of comparisons between the methods on a set of data

* 'python run.py replicate' will generate replications of the results of the 3 papers 

* 'python run.py real_data' will run the 3 methods on two sets of real world data, sacks and tuebingen


### Run your own using main.py

* You can run each method on generated data using 'python main.py' comand from the root directory

* Many examples are provided in run.ipynb

* 'python main.py' is followed by the name of the method in CAPS(NOTEARS, GOLEM, DAGMA) followed by its required arguments

```rb
python3 main.py --method GOLEM \
                     --seed 1 \
                     --d 10 \
                     --graph_type ER \
                     --degree 4 \
                     --noise_type gaussian_ev \
                     --equal_variances \
                     --lambda_1 2e-2 \
                     --lambda_2 5.0 \
                     --checkpoint_iter 5000

python3 main.py  --method DAGMA \
                     --seed 1 \
                     --d 10 \
                     --lambda_1 2e-2 \
                     --graph_type ER \
                     --degree 4 \
                     --noise_type gaussian_ev \
                     --equal_variances \
                     --checkpoint_iter 1000 \
                     --loss l2

!python3 main.py  --method NOTEARS \
                     --seed 1 \
                     --d 10 \
                     --graph_type ER \
                     --degree 4 \
                     --noise_type gaussian_ev \
                     --equal_variances \
                     --lambda_1 2e-2 \
                     --loss l2
``` 
 
  
## Reference
Ng, Ignavier, AmirEmad Ghassami, and Kun Zhang. "On the role of sparsity and dag constraints for learning linear dags." *Advances in Neural Information Processing Systems* 33 (2020): 17943-17954.

Zheng, Xun, et al. "Dags with no tears: Continuous optimization for structure learning." *Advances in neural information processing systems* 31 (2018).

Bello, Kevin, Bryon Aragam, and Pradeep Ravikumar. "Dagma: Learning dags via m-matrices and a log-determinant acyclicity characterization." *Advances in Neural Information Processing Systems* 35 (2022): 8226-8239.

Pamfil, Roxana, et al. "Dynotears: Structure learning from time-series data." International Conference on Artificial Intelligence and Statistics. PMLR, 2020.


