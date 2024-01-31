import numpy as np

def train(model, Y, epochs=1_000):
    for i in range(epochs):
        score, likelihood, h = train_step(model, Y,epoch=i)
        if i % 100 == 0:
            print(score)
            print(likelihood)
            print(h)
    

def train_step(model , Y, epoch):
    model.train_op.zero_grad()

    score, likelihood, h  = model.run(Y)
    score.backward()
    model.train_op.step()


    return score, likelihood, h