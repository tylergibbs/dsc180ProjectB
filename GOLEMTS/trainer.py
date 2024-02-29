import numpy as np

def train(model, Y, epochs=1_000, log=False, warmup_epochs=0, center_data=True):
    likes = []
    evs = []

    if center_data:
        Y = Y - Y.mean(axis=0, keepdims=True)

    if warmup_epochs > 0:
        model.ev = True
    for i in range(epochs):
        if i > 0 and i == warmup_epochs:
            model.ev = False

        score, likelihood, ev_res = train_step(model, Y,epoch=i)
        if i % 100 == 0:
            # print(score)
            if log:
                print(f'likelihood: {likelihood}')
            # print(model.B.grad)

            # print(h)
            likes.append(likelihood.detach().numpy())
            evs.append(ev_res.detach().numpy())
    return likes, evs
    

def train_step(model , Y, epoch):
    model.train_op.zero_grad()

    score, likelihood, ev_res  = model.run(Y)
    score.backward()
    model.train_op.step()


    return score, likelihood, ev_res