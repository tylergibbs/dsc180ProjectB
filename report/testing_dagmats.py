from src.data.generate_data import SyntheticDataset
import src.utill.testing_utils as testing_utils
import numpy as np
import jsonlines
import src.utill.grid_search as grid_search
import time
import copy
import torch

import src.model.DYNOTEARS.run_dynotears as run_dynotears
from src.model.GOLEMTS.model import GolemTS
import src.model.GOLEMTS.trainer as trainer
from src.model.DAGMATS import nonlinear_dagma
from src.model.DAGMATS import dagmats

global_threshold = 0.2

# functions for all models

# dynotears
def dynotears(dag_obj):
    return run_dynotears.learn_dag(dag_obj)


# golemts EV
def golemts_EV(dag_obj):
    model = GolemTS(n=dag_obj.n, d=dag_obj.d, p=dag_obj.p, Y=dag_obj.Y, lambda_1=0.01, lambda_2=1.0, A_init=None, ev=True, lr=3e-3, lambda_3=5.0)
    likes, evs = trainer.train(model, dag_obj.Y, epochs=50_000, warmup_epochs=0, log=False)
    model_B = model.B.detach().numpy()
    model_B[np.abs(model_B) < 0.2] = 0
    model_A = model_B[:, :dag_obj.d]
    return model_A

    

# golemts NV
def golemts_NV(dag_obj):
    model = GolemTS(n=dag_obj.n, d=dag_obj.d, p=dag_obj.p, Y=dag_obj.Y, lambda_1=0.01, lambda_2=1.0, A_init=None, ev=False, lr=3e-3, lambda_3=5.0)
    likes, evs = trainer.train(model, dag_obj.Y, epochs=50_000, warmup_epochs=20_000, log=False)
    model_B = model.B.detach().numpy()
    model_B[np.abs(model_B) < 0.2] = 0
    model_A = model_B[:, :dag_obj.d]
    return model_A


# dagmaTS
def dagma_ts(dag_obj):
    p = dag_obj.p
    d = dag_obj.d
    n = dag_obj.n
    lambda1 = 0.05
    lambda2 = 0.01
    lr = 0.001
    thresh = 0.2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    X = dag_obj.X
    Y = dag_obj.Y
    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)
    
    eq_model = dagmats.DagmaTS(n=n, p=p, d=d, device=device)
    eq_model = eq_model.to(device)

    model = dagmats.DagmaLinear(eq_model, verbose=False)
    W_est = model.fit(X, Y, lambda1=lambda1, lambda2=lambda2, lr=lr, w_threshold=thresh)
    return W_est

# dagnaTS_nl
def dagmats_nl(dag_obj):
    p = dag_obj.p
    d = dag_obj.d
    n = dag_obj.n
    h1_dim = 20
    lambda1 = 0.02
    lambda2 = 0.005
    lr = 0.02
    eq_model = nonlinear_dagma.DagmaMLP(dims=[(p+1) * d, h1_dim, 1], out_dims=d, bias=True)
    model = nonlinear_dagma.DagmaNonlinear(eq_model)
    W_est = model.fit(dag_obj.X, dag_obj.Y, lambda1=lambda1, lambda2=lambda2, lr=lr, w_threshold=0)
    return W_est

function_dict = {
    'DAGMATS': dagma_ts
}


def gen_dags(
        ns = [50, 500],
        degrees=[2, 4],
        nodes = [5, 10, 20, 50, 100],
        ps = [1],
        noise_types = ['EV', 'NV', 'EXP', 'GUMBEL'],
        mlps = [False, True],
        reps=10
    ):
    dag_list = []
    dag_stats = []
    for mlp in mlps:
        for noise_type in noise_types:
            for p in ps:
                for d in nodes:
                    for degree in degrees:
                        for n in ns:
                            for i in range(reps):
                                dag_obj = SyntheticDataset(
                                    n=n,
                                    p=p,
                                    d=d,
                                    graph_type='ER',
                                    degree=degree,
                                    noise_type=noise_type,
                                    mlp=mlp,
                                    A_scale=1.5,
                                    B_scale=1.0
                                )
                                dag_list.append(dag_obj)
                                dag_stats.append({
                                    'n': n,
                                    'p': p,
                                    'd': d,
                                    'degree': degree,
                                    'noise_type': noise_type,
                                    'mlp': mlp
                                })
    return dag_list[1448:], dag_stats[1448:]


def test_all_methods(output_dir):
    dag_list, dag_stats = gen_dags()
    for name, func in function_dict.items():
        for dag_obj, dag_stat in zip(dag_list, dag_stats):
            # get estimated dag
            start = time.time()
            A_est = func(dag_obj)
            end = time.time()
            elapsed = end - start
            # add runtime and model name to future dictionary containing stats
            param_dict = copy.deepcopy(dag_stat)
            param_dict['runtime'] = elapsed
            param_dict['model_name'] = name
            # post process
            A_est_dag, A_true_dag = grid_search.postprocess_A(A_est, dag_obj.A, graph_thres=global_threshold)
            metrics = grid_search.save_metrics(A_est_dag, A_true_dag, param_dict, output_dir)
            print(metrics)

