# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:50:41 2020

@author: kshitij
"""
import torch
import numpy as np

class OLS_pytorch(object):
    def __init__(self):
        self.coefficients = []
        self.X = None
        self.y = None

    def fit(self,X,y):
        if len(X.shape) == 1:
            X = self._reshape_x(X)
        if len(y.shape) == 1:
            y = self._reshape_x(y).T
        #print("X shape   ", X.shape)
        #print("Y shape   ", y.shape)
        X =  self._concatenate_ones(X)
        #print("X shape after concatenation  ", X.shape)
        self.X = torch.from_numpy(X).float().cuda()
        self.y = torch.from_numpy(y).float().cuda()
        XtX = torch.matmul(self.X.T,self.X)
        #print("Xtx shape", XtX.shape)
        #print("y unsqueeze shape", self.y.unsqueeze(2).shape)
        Xty = torch.matmul(self.X.T,self.y.unsqueeze(2))
        #print("Xty shape", Xty.shape)
        XtX = XtX.unsqueeze(0)
        #print("Xtx shape after unsqueezing", XtX.shape)
        XtX = torch.repeat_interleave(XtX, self.y.shape[0], dim=0)
        #print("Xtx shape after tiling", XtX.shape)
        betas_cholesky, _ = torch.solve(Xty, XtX)
        #print("betas shape", betas_cholesky.shape)
        self.coefficients = betas_cholesky

    def predict(self, entry):
        b0 = self.coefficients[0]
        other_betas = self.coefficients[1:]
        prediction = b0

        for xi,bi in zip(entry,other_betas):
            prediction += bi*xi

        return prediction

    def score(self):
        prediction = torch.matmul(self.X,self.coefficients)
        prediction = prediction
        #print("prediction shape ", prediction.shape)

        #print("y mean ", self.y.shape)
        yhat = prediction                        # or [p(z) for z in x]
        ybar = (torch.sum(self.y,dim=1, keepdim=True)/self.y.shape[1]).unsqueeze(2)
        #print("ybar shape, yhat shape", ybar.shape,yhat.shape)# or sum(y)/len(y)
        ssreg = torch.sum((yhat-ybar)**2,dim=1, keepdim=True)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
        sstot = torch.sum((self.y.unsqueeze(2) - ybar)**2,dim=1, keepdim=True)    # or sum([ (yi - ybar)**2 for yi in y])
        score = ssreg / sstot
        #score = ((prediction-y.mean(dim=1, keepdim=True))/(y.mean(dim=1, keepdim=True)-y))**2
        #print("score shape is ", score.shape)
        return score.cpu().numpy().ravel()

    def _reshape_x(self,X):
        return X.reshape(-1,1)

    def _concatenate_ones(self,X):
        ones = np.ones(shape=X.shape[0]).reshape(-1,1)
        #print("Ones shape   ", ones.shape)
        return np.concatenate((ones,X),1)