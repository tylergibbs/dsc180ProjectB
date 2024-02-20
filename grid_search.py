import testing_utils
import numpy as np
import jsonlines

# convert A to a square wam, and threshold weights
def postprocess_A(A_est,  A_true, graph_thres=0.1):
   # convert est to dag:
    d = A_est.shape[1]
    p = int(A_est.shape[0] / d) - 1
    A_dag = np.hstack([A_est, np.zeros(((p+ 1) * d, p * d))]) 
    A_true_dag = np.hstack([A_true, np.zeros(((p+ 1) * d, p * d))]) 
    # threshold 
    A_dag = testing_utils.postprocess(A_dag, graph_thres)
    return A_dag, A_true_dag

# given square matricies A_est and A_true, compute metrics and save in 
# output dir along with hyperparameters
def save_metrics(A_est, A_true, param_dict, output_dir):
    metric_dict = testing_utils.count_accuracy(A_est != 0, A_true != 0)
    metric_dict |= param_dict
    with jsonlines.open(output_dir, mode='a') as f:
        f.write(metric_dict)
    return metric_dict

# given grid (list of dictionaries with hyperparameters)
# model_func: model function that with args corresponding to grid dicts
# output_dir, where to save the results
# log performance metrics for each entry in grid
def perform_grid_search(grid, model_func, output_dir):
    # we have the grid already, now we just iterate thru it
    for g in grid:
        # get estimantes and true dag
        A_est, A_true = model_func(**g)
        # post process
        A_est, A_true = postprocess_A(A_est, A_true)
        # save metrics
        metrics = save_metrics(A_est, A_true, param_dict=g, output_dir=output_dir)
        print(metrics)

# example usage using dagma
def test_dagma(output_dir):
    # required imports for dagma
    from dagma_nl import nonlinear_dagma
    from dagma_nl.generate_data import SyntheticDataset



    # constuct grid
    def dagma_grid(h1_dim_list, lambda1_list, lambda2_list, lr_list):
        out_list = []
        for h1_dim in h1_dim_list:
            for lambda1 in lambda1_list:
                for lambda2 in lambda2_list:
                    for lr in lr_list:
                        out_list.append({
                            'h1_dim': h1_dim,
                            'lambda1': lambda1,
                            'lambda2': lambda2,
                            'lr': lr
                        })
        return out_list
    
    # create dagma function
    def dagma(h1_dim, lambda1, lambda2, lr):
        # these vars are fixed
        n, d, p = 1000, 5, 3
        dag_obj = SyntheticDataset(n, d, p, B_scale=1.0, graph_type='ER', degree=2, A_scale=1.0, noise_type='EV', mlp=True)

        A_true = dag_obj.A
        X = dag_obj.X
        Y = dag_obj.Y

        # we manipulate these
        # h1_dim = 30
        # lambda1=0.01
        # lambda2=0.03
        # lr=0.02

        eq_model = nonlinear_dagma.DagmaMLP(dims=[(p+1) * d, h1_dim, 1], out_dims=d, bias=True)
        model = nonlinear_dagma.DagmaNonlinear(eq_model)
        adj = eq_model.fc1_to_adj()
        W_est = model.fit(X, Y, lambda1=lambda1, lambda2=lambda2, lr=lr, w_threshold=0)
        return W_est, A_true
    
    grid = dagma_grid(
        h1_dim_list=[10, 20, 50],
        lambda1_list=[0.005, 0.01, 0.02, 0.05, 0.1],
        lambda2_list=[0.005, 0.01, 0.03, 0.05],
        lr_list=[0.005, 0.01, 0.02]
    )
    perform_grid_search(grid, model_func=dagma, output_dir=output_dir)







    

