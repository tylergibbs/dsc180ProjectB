import numpy as np
from src.model.GOLEMTS.model import GolemTS
from  src.model.GOLEMTS.trainer import train as golem_trainer
from src.model.DYNOTEARS import run_dynotears
from src.model.DYNOTEARS import dynotears
from src.model.DAGMATS.dagmats import DagmaTS
from src.model.DAGMATS.dagmats import DagmaLinear


def run_DYNOTEARS(n, d, p, Y, w_thresh=0.01, epochs=100):
    X = Y[:, :d]
    X_lag = Y[:, d:]

    result, w_est, a_est = dynotears.from_numpy_dynamic(X, X_lag, w_threshold=w_thresh, max_iter=epochs)
    A = np.vstack([w_est, a_est])

    return A


def run_GOLEMTS(n, d, p, Y, lambda_1=0.1, lambda_2=1, A_init=None, ev=True, lr=3e-3, lambda_3=9,
               epochs=1000, warmup_epochs=0, log=False, device=None):
    model = GolemTS(n=n, d=d, p=p, Y=Y, device=device, 
                lambda_1=lambda_1, lambda_2=lambda_2, A_init=A_init, ev=True, lr=lr, lambda_3=lambda_3)

    likes, evs = golem_trainer(model, Y, epochs=epochs, warmup_epochs=warmup_epochs, log=log, device=device)

    model_B = model.B.detach().numpy()
    model_B[np.abs(model_B) < 0.2] = 0
    model_A = model_B[:, :d]

    np.fill_diagonal(model_A, 0)

    return model_A


def run_DAGMATS(n, d, p, Y, lambda1=0.01, lambda2=0.03, lr=0.02, w_threshold=0,
                epochs=1000, device=None):
    X = Y[:, :d]

    eq_model = DagmaTS(n=n, p=p, d=d, device=device)
    eq_model = eq_model.to(device)

    model = DagmaLinear(eq_model, verbose=True)
    W_est = model.fit(X, Y, lambda1=0.01, lambda2=0.03, lr=0.02, w_threshold=0, max_iter=epochs)

    return W_est
