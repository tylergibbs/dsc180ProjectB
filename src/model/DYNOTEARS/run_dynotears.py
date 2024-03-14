from src.data.generate_data import SyntheticDataset
import src.model.DYNOTEARS.dynotears as dynotears

import numpy as np

def gen_dag(n, d, p, noise_type='EV', degree=2):
    dag_obj = SyntheticDataset(n=n, d=d, p=p, B_scale=1.0, graph_type='ER', degree=degree, A_scale=1.0, noise_type=noise_type)
    return dag_obj

def learn_dag(dag_obj, w_thresh=0.01):
    X_lag = dag_obj.Y[:, dag_obj.d:]
    result, w_est, a_est = dynotears.from_numpy_dynamic(dag_obj.X, X_lag, w_threshold=w_thresh)
    A = np.vstack([w_est, a_est])
    return A


def run(n, d, p, noise_type='EV', degree=2, w_thresh=0.01):
    dag_obj = gen_dag(n=n, d=d, p=p, noise_type=noise_type, degree=degree)
    A = learn_dag(dag_obj=dag_obj, w_thresh=w_thresh)
    return dag_obj.A, A
