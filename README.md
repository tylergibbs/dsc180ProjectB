
# Advancement of Structure Learning Methods for Time Series Data

This project extrapolates 2 different methods for causal discovery into time series data and compares them to the effectiveness of existing methods:

- GOLEM
- DAGMA

This project adapts existing implementations of GOLEM and DAGMA into time series data using the methods outlined in DYNOTEARS. We replicate the metrics in the DYNOTEARS papers, and compare their performance to a baseline of DYNOTEARS. Finally we make minor improvements implementing an early stopping threshold. We expect DAGMA to perform the best, followed by GOLEM, followed by NOTEARS as this is the inverse order they were developed.

## Data:

We have two sources of data.

(1) Synthetic data produced by synthetic_dataset in src/data/synthetic_dataset

(2) Data from the yfinance library, 


## Running the project

* To install the dependencies, run the following command from the root directory of the project: `pip install -r requirements.txt`

### Building the project stages using `run.py`
This section allows the user to replicate the results in our report

* 'python run.py all' will generate all data and graphs used in our report

* 'python run.py snp100' will generate a graph illustrating the structural relationship in the snp100  

* 'python run.py data' will regenerate all the data in results/ this will take time so it is recommended to use the provided data

* 'python run.py graphs' uses the data in results/ to generate all the graphs in our report

* 'python run.py clean' removes all files generated by all

### Run your own using main.py

You can run each method on generated data or a provided csv file using 'python main.py' command from the root directory

'python main.py' is followed by the name of the method (DYNOTEARS, GOLEMTS, DAGMATS) followed by arguments specifying the data and output directory.

The -p autoregressive order must always be specified

If you are using existing data you give the filename of the csv file under --data as shown below

```rb
python3 main.py GOLEMTS \
                      -p 1 \
                      --data filename.csv \ 
``` 

If no --out parameter is specified it will write to output.json. If the --out parameter is specified it will write to that file
```rb
python3 main.py GOLEMTS \
                      --data filename.csv \
                      -p 1 \
                      --out new_output.json
``` 

It can also generate synthetic data using the following two parameters
* -n number of observations
* -d number of variables
* --degree number of causal relationships
The following parameters are optional to further specify the type of synthetic data generated
* --graph_type ER, SF respectively Erdös-Rényi or scale-free graphs
* --noise_type EV, EN, EXP GUMBEL, which are respectively gaussian equal variance, gaussian nonequal variance, exponential, gumbel 

```rb
python3 main.py GOLEMTS \
                     -n 10 \
                     -d 3 \
                     -p 1 \
                     --degree 4 

python3 main.py  DAGMATS \
                     -n 10 \
                     -d 3 \
                     -p 1 \
                     --graph_type ER \
                     --degree 4 \
                     --noise_type EV 

python3 main.py  DYNOTEARS \
                     -n 10 \
                     -d 3 \
                     -p 1 \
                     --graph_type SF \
                     --degree 4 \
                     --noise_type EXP 
``` 

### Use the python methods for maximum flexibility

The 3 methods each run their respective method with the given hyperparameters

* src.train.run_DYNOTEARS(n, d, p, Y, w_thresh=0.01, epochs=100):

* src.train.run_GOLEMTS(n, d, p, Y, lambda_1=0.1, lambda_2=1, ev=True, lr=3e-3, lambda_3=9, epochs=1000):

* src.train.run_DAGMATS(n, d, p, Y, lambda1=0.01, lambda2=0.03, lr=0.02, w_threshold=0, epochs=1000)


  
## Reference
Ng, Ignavier, AmirEmad Ghassami, and Kun Zhang. "On the role of sparsity and dag constraints for learning linear dags." *Advances in Neural Information Processing Systems* 33 (2020): 17943-17954.

Zheng, Xun, et al. "Dags with no tears: Continuous optimization for structure learning." *Advances in neural information processing systems* 31 (2018).

Bello, Kevin, Bryon Aragam, and Pradeep Ravikumar. "Dagma: Learning dags via m-matrices and a log-determinant acyclicity characterization." *Advances in Neural Information Processing Systems* 35 (2022): 8226-8239.

Pamfil, Roxana, et al. "Dynotears: Structure learning from time-series data." International Conference on Artificial Intelligence and Statistics. PMLR, 2020.


