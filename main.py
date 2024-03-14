import json
import logging
import argparse
import numpy as np
import pandas as pd

from src.data.generate_data import SyntheticDataset
from src.train import *


def get_args():
    parser = argparse.ArgumentParser(
                    prog='main',
                    description='Allows the user to run DYNOTEARS, GOLEMTS, and DAGMATS on generated or given data')

    parser.add_argument("method", choices = ['GOLEMTS', 'DAGMATS', 'DYNOTEARS'])
    parser.add_argument("--data", required=False)
    parser.add_argument("--out", required=False)
    parser.add_argument("-n", type=int, required=False)
    parser.add_argument("-d", type=int, required=False)
    parser.add_argument("-p", type=int, required=False)
    parser.add_argument("--graph_type", choices=["ER", "SF"], required=False)
    parser.add_argument("--degree", type=int, required=False)
    parser.add_argument("--noise_type", choices=['EV', 'NV', 'EXP', 'GUMBEL'], required=False)
    return parser.parse_args()


def main():
    # Get arguments parsed
    args = get_args()

    output_file = 'output.json'
    if args.out:
       output_file = args.out
    if args.p is None:
       print("must specify autoregesive order p")

    if ((args.data == None) + (args.n == None)) > 1:
       print("only one of data, dag, n can be set")
       return 
    # Load dataset
    elif args.data is not None:
       dag = None
       Y = pd.read_csv(args.data).to_numpy()
    elif args.n is not None:
       if args.d is None:
          print("if n is specified d must be specified")
          return
       if args.degree is None:
          print("if n is specified degree must be specified")
          return
       if args.graph_type is None:
          args.graph_type = "ER"
       if args.noise_type is None:
          args.noise_type = "EV"
       dataset = SyntheticDataset(args.n, args.d, args.p, args.graph_type, args.degree,
                               args.noise_type)
       dag = dataset.A
       Y = dataset.Y
    else:
       print("None of data or n were set, so no data was specified")
       return

    p = args.p
    n = Y.shape[0]
    d = Y.shape[1]//(p+1)

    #Method
    if args.method == 'GOLEMTS':
       print("GOLEMTS")
       dag_est = run_GOLEMTS(n, d, p, Y) 
    elif args.method == "DAGMATS":
       print("DAGMATS")
       dag_est = run_DAGMATS(n, d, p, Y)
    elif args.method == "DYNOTEARS":
       print("DYNOTEARS")
       dag_est = run_DYNOTEARS(n, d, p, Y)  


    output = {"dag":dag.tolist(),
              "dag_est":dag_est.tolist()
             }
    out = open(output_file, "w")
    json.dump(output, out) 
    out.close()
    

if __name__ == '__main__':
    main()

