import numpy as np

def train(model, Y, epochs=1_000, log=False, warmup_epochs=0, center_data=True, es=True, es_tol=1e-4):
    likes = []
    evs = []
    scores = [float('inf')]

    if center_data:
        Y = Y - Y.mean(axis=0, keepdims=True)

    if warmup_epochs > 0:
        model.ev = True
    i = 0
    while i  < epochs:
        if i > 0 and i == warmup_epochs:
            model.ev = False

        score, likelihood, ev_res = train_step(model, Y,epoch=i)
        if i % 100 == 0:
            # print(score)
            if log:
                print(f'likelihood: {likelihood}')
                print(f'Score: {score}')
            # print(model.B.grad)
            print((score.detach().numpy() - scores[-1]) / scores[-1])
            if es and np.abs(score.detach().numpy() - scores[-1]) / scores[-1] < es_tol:
                # we must either: skip warmup, or end entirely
                if i < warmup_epochs:
                    if log:
                        print(f'Warmup early stop epoch : {i}')
                    i = warmup_epochs
                    

                else:
                    if log:
                        print(f'early stop epoch : {i}')
                    i = epochs
                   


            # print(h)
            likes.append(likelihood.detach().numpy())
            evs.append(ev_res.detach().numpy())
            scores.append(score.detach().numpy())
        i += 1
    return likes, evs
    

def train_step(model , Y, epoch):
    model.train_op.zero_grad()

    score, likelihood, ev_res  = model.run(Y)
    score.backward()
    model.train_op.step()


    return score, likelihood, ev_res