#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys, os
from matplotlib import pyplot as plt
from pyDOE import lhs

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F

import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel, LinearKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.models import AbstractVariationalGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood, SoftmaxLikelihood
from gpytorch.models.deep_gps import AbstractDeepGPLayer, AbstractDeepGP, DeepLikelihood
from gpytorch.lazy import BlockDiagLazyTensor, lazify
from scipy.special import erf, expit

from pyTrajectoryUtils.pyTrajectoryUtils.utils import *

# Values required for approximating the logistic sigmoid by
# error functions. coefs are obtained via:
# x = np.array([0, 0.6, 2, 3.5, 4.5, np.inf])
# b = logistic(x)
# A = (erf(np.dot(x, self.lambdas)) + 1) / 2
# coefs = lstsq(A, b)[0]
# Approximate \int log(z) * N(z | f_star, var_f_star)
# Approximation is due to Williams & Barber, "Bayesian Classification
# with Gaussian Processes", Appendix A: Approximate the logistic
# sigmoid by a linear combination of 5 error functions.
# For information on how this integral can be computed see
# blitiri.blogspot.de/2012/11/gaussian-integral-of-error-function.html
LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, np.newaxis]
COEFS = np.array([-1854.8214151, 3516.89893646, 221.29346712,
                  128.12323805, -2010.49422654])[:, np.newaxis]

class MFDeepGPLayer(AbstractDeepGPLayer):
    def __init__(self, input_dims, output_dims, prev_dims=0, num_inducing=512, inducing_points=None, prev_layer=None):
        self.prev_dims = prev_dims
        input_all_dims = input_dims + prev_dims
        
        # TODO
        if inducing_points is None:
            if output_dims is None:
                inducing_points = torch.randn(num_inducing, input_all_dims)
            else:
                inducing_points = torch.randn(output_dims, num_inducing, input_all_dims)
        
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([output_dims]) if output_dims is not None else torch.Size([])
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(MFDeepGPLayer, self).__init__(variational_strategy, input_all_dims, output_dims)
        
        self.mean_module = ConstantMean(batch_size=output_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_size=output_dims, ard_num_dims=input_dims),
            batch_size=output_dims, ard_num_dims=None
        )
        
        self.prev_layer = prev_layer
        if prev_dims > 0:
            self.covar_module_corr = ScaleKernel(
                RBFKernel(batch_size=output_dims, ard_num_dims=input_dims),
                batch_size=output_dims, ard_num_dims=None
            )
            self.covar_module_prev = ScaleKernel(
                RBFKernel(batch_size=output_dims, ard_num_dims=None),
                batch_size=output_dims, ard_num_dims=None
            )
            self.covar_module_linear = ScaleKernel(
                LinearKernel(batch_size=output_dims, ard_num_dims=None)
            )
    
    def covar(self, x):
        x_input = torch.index_select(x, -1, torch.arange(self.prev_dims,self.input_dims).long().cuda())
        x_prev = torch.index_select(x, -1, torch.arange(self.prev_dims).long().cuda())
        covar_x = self.covar_module(x_input)
        if self.prev_dims > 0:
            k_corr = self.covar_module_corr(x_input)
            k_prev = self.covar_module_prev(x_prev)
#             k_prev = self.prev_layer.covar(x_input)
            k_linear = self.covar_module_linear(x_prev)
            covar_x += k_corr*(k_prev + k_linear)
#             covar_x = k_corr*(k_prev)
            
        return covar_x

    def forward(self, x):
        # https://github.com/amzn/emukit/blob/master/emukit/examples/multi_fidelity_dgp/multi_fidelity_deep_gp.py
        x_input = torch.index_select(x, -1, torch.arange(self.prev_dims,self.input_dims).long().cuda())
        mean_x = self.mean_module(x_input) # self.linear_layer(x).squeeze(-1)
        covar_x = self.covar(x)
            
        return MultivariateNormal(mean_x, covar_x)
    
    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

                processed_inputs = [
                    inp.unsqueeze(0).expand(x.shape[0], *inp.shape)
                    for inp in other_inputs
                ]
            else:
                processed_inputs = [
                    inp for inp in other_inputs
                ]
            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))

class MFDeepGPC(AbstractDeepGP):
    def __init__(self, train_x, train_y, num_inducing=512, input_uc=0):
        super().__init__()
        
        num_fidelity = len(train_x)
        train_x_shape = train_x[0].shape
        
        # Generate Inducing points - TODO check higher fidelity inducing points
        train_z = []
        
        i_z = torch.randperm(train_x[0].size(0)).cuda()[:num_inducing]
        z_low = train_x[0][i_z, :]
        setattr(self, 'train_z_' + str(0), z_low)
        train_z.append(z_low)
        for i in range(1,num_fidelity):
            i_z_low = torch.randperm(train_x[i-1].size(0)).cuda()[:num_inducing]
            z_high = torch.cat([train_x[i-1][i_z_low, :], train_y[i-1][i_z_low].unsqueeze(-1)], axis=1).unsqueeze(0)
            setattr(self, 'train_z_' + str(i), z_high)
            train_z.append(z_high)
        
        # Generate Multifidelity layers
        self.layers = []
        layer = MFDeepGPLayer(
            input_dims=train_x_shape[-1],
            output_dims=1,
            prev_dims=input_uc,
            num_inducing=num_inducing,
            inducing_points=train_z[0]
        )
        setattr(self, 'layer_' + str(0), layer)
        self.layers.append(layer)
        
        for i in range(1,num_fidelity):
            layer = MFDeepGPLayer(
                input_dims=train_x_shape[-1],
                output_dims=1,
                prev_dims=1,
                num_inducing=num_inducing,
                inducing_points=train_z[i],
                prev_layer=self.layers[i-1]
            )
            setattr(self, 'layer_' + str(i), layer)
            self.layers.append(layer)
        
        self.likelihood = DeepLikelihood(BernoulliLikelihood())
    
    def forward(self, inputs, fidelity=2, eval=False):
        val = self.layers[0](inputs, eval=eval)
        for layer in self.layers[1:fidelity]:
            val = layer(val, inputs, eval=eval)
        val = MultivariateNormal(val.mean.squeeze(-1), val.lazy_covariance_matrix)
        return val
    
    def predict(self, x, fidelity=2):
        with gpytorch.settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            preds = self(x, fidelity=fidelity, eval=True)
        
        val = MultivariateNormal(preds.mean[0], preds.lazy_covariance_matrix[0])
        for i in range(1,preds.mean.shape[0]):
            val += MultivariateNormal(preds.mean[i], preds.lazy_covariance_matrix[i])
        val /= (preds.mean.shape[0])
        
        return self.likelihood.base_likelihood(val).mean.ge(0.5).cpu().numpy()
    
    def predict_proba(self, x, fidelity=2, return_std=False):
        with gpytorch.settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            preds = self(x, fidelity=fidelity, eval=True)

        val = MultivariateNormal(preds.mean[0], preds.lazy_covariance_matrix[0])
        for i in range(1,preds.mean.shape[0]):
            val += MultivariateNormal(preds.mean[i], preds.lazy_covariance_matrix[i])
        val /= (preds.mean.shape[0])

        pred_means = val.mean.cpu().numpy()
        pred_vars = val.variance.cpu().numpy()

        var_f_star = pred_vars
        f_star_min = pred_means
        
        bern = self.likelihood.base_likelihood(val)
        pi_star = bern.probs.cpu().numpy()
        
        if return_std:
            return f_star_min, np.sqrt(var_f_star)
        else:
            return f_star_min, np.sqrt(var_f_star), np.vstack((1 - pi_star, pi_star)).T
        
    def predict_proba_MF(self, x, fidelity=1, C_L=1., C_H=10., beta=0.05, return_std=False, return_all=False):
        with gpytorch.settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            preds = self(x, fidelity=fidelity, eval=True)
            
        val = MultivariateNormal(preds.mean[0], preds.lazy_covariance_matrix[0])
        for i in range(1,preds.mean.shape[0]):
            val += MultivariateNormal(preds.mean[i], preds.lazy_covariance_matrix[i])
        val /= (preds.mean.shape[0])

        pred_means = val.mean.cpu().numpy()
        pred_vars = val.variance.cpu().numpy()

        var_f_star = pred_vars
        f_star_min = pred_means - beta*np.sqrt(pred_vars)
        f_star_min_i = pred_means
        
        val_mf = MultivariateNormal(val.mean-beta*torch.sqrt(val.variance), val.lazy_covariance_matrix)
        bern = self.likelihood.base_likelihood(val_mf)
        pi_star = bern.probs.cpu().numpy()
        
        bern_i = self.likelihood.base_likelihood(val)
        pi_star_i = bern_i.probs.cpu().numpy()
        
        if return_all:
            if return_std:
                return f_star_min, np.sqrt(var_f_star), f_star_min_i, np.sqrt(var_f_star), np.vstack((1 - pi_star_i, pi_star_i)).T
            else:
                return np.vstack((1 - pi_star, pi_star)).T, f_star_min_i, np.sqrt(var_f_star), np.vstack((1 - pi_star_i, pi_star_i)).T
        else:
            if return_std:
                return f_star_min, np.sqrt(var_f_star)
            else:
                return np.vstack((1 - pi_star, pi_star)).T
