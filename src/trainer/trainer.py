import logging

import numpy as np
import math
import torch 


class Trainer:

    def __init__(self, learning_rate=1e-3):
        self.learning_rate=learning_rate

    def train(self, model, X_train, X_val, num_iter, 
              es_threshold=None, checkpoint_iter=None, output_dir=None):
        model.set_learning_rate(self.learning_rate)

        self.model = model
        self.train_score, self.train_info = self.eval_iter(X_train)
        self.val_score, self.val_info = self.eval_iter(X_val)
        
        self.pre_score = None

        for i in range(0, int(num_iter)):
            if checkpoint_iter is not None and i%checkpoint_iter == 0:
               self.checkpoint(i, output_dir)

            if (es_threshold is not None and i > 10 and 
               (self.train_score - self.pre_score)/self.pre_score < es_threshold):
               num_iter = i
               break

            self.pre_score = self.val_score
            self.train_score, self.train_info = self.train_iter(X_train)
            self.val_score, self.val_info = self.eval_iter(X_val)     
        
        if checkpoint_iter is not None:
           self.checkpoint(num_iter, output_dir)

        
    def eval_iter(self, X):
        with torch.no_grad():
           score, info = self.model.run(X)
        return score, info

    def train_iter(self, X):
        score, info  = self.model.update(X)
        return score, info

    def checkpoint(self, iter, output_dir):
        train_str = "TRAINING: [Iter {}] score {}, {}"
        train_str = train_str.format(iter, self.train_score, 
             ", ".join(["{} {}".format(i, self.train_info[i]) for i in list(self.train_info)]))
 
        val_str = "VALIDATION: [Iter {}] score {}, {}"
        val_str = iter, val_str.format(iter, self.val_score, 
             ", ".join(["{} {}".format(i, self.train_info[i]) for i in list(self.train_info)]))

        #TODO probubly return 2 strings for csv and for logger
        #self.model.checkpoint()

        #TODO make logger
        print(train_str)
        print(val_str)

        if output_dir is not None:
           pass#TODO write csv new file



