import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import pyro
from pyro.distributions import Normal, Uniform, Delta, LogNormal, Exponential
from pyro.optim import Adam
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO
from pyro.contrib.autoguide import AutoDiagonalNormal
from sklearn import preprocessing

from torch.utils.data import TensorDataset, DataLoader
from src.utils.compute_uplift_curve import f_compute_uplift_curve

import time


def get_df_flights(data_path):
    df_flights = pd.read_csv(data_path)
    df_flights = df_flights.loc[~(df_flights.ARRIVAL_DELAY.isna())]
    df_flights = df_flights.loc[~(df_flights.MONTH.isin([10]))]
    # all airlines showup in the 1st month already (i will use this month to encode them)
    # assert len(set(df_flights['AIRLINE']).difference(set(df_flights.loc[df_flights.MONTH==1, 'AIRLINE']))) == 0
    #
    # df_airline_encoding = df_flights.loc[df_flights.MONTH==1, ['AIRLINE', 'ARRIVAL_DELAY']].groupby(by='AIRLINE').mean().rename(columns={'ARRIVAL_DELAY': 'AIRLINE_MEAN_ENCODING'})
    # df_data = pd.merge(df_flights, df_airline_encoding, left_on='AIRLINE', right_index=True)

    col_features = [
        #     'DEP_MINOFDAY', 'ARR_MINOFDAY',
        'DISTANCE', 'DEP_LONGITUDE_SIN', 'DEP_LONGITUDE_COS', 'DEP_LATITUDE_SIN',
        'DEP_LATITUDE_COS', 'DEST_LONGITUDE_SIN', 'DEST_LONGITUDE_COS',
        'DEST_LATITUDE_SIN', 'DEST_LATITUDE_COS', 'DEP_MINOFDAY_SIN',
        'DEP_MINOFDAY_COS', 'ARR_MINOFDAY_SIN', 'ARR_MINOFDAY_COS',
        'DAYWEEK_SIN', 'DAYWEEK_COS', 'DAYMONTH_SIN', 'DAYMONTH_COS',
        'destAirFreq', 'depAirFreq'
        #     ,'AIRLINE_MEAN_ENCODING'
    ]
    col_target = ['ARRIVAL_DELAY']
    cols_to_scale = ['DISTANCE', 'destAirFreq', 'depAirFreq']
    cols_not_scale = [c for c in col_features if c not in cols_to_scale]
    scaler = preprocessing.StandardScaler().fit(df_flights[cols_to_scale].values)

    # month_train = list(range(1, 2))
    # month_test = [2]
    month_train = list(range(1, 11))
    month_test = [11, 12]

    df_train = df_flights.loc[df_flights.MONTH.isin(month_train)]
    df_test = df_flights.loc[df_flights.MONTH.isin(month_test)]


    # get datasets
    x_train = np.hstack(
        [
            df_train[cols_not_scale].values,
            scaler.transform(df_train[cols_to_scale].values)
        ]
    )

    x_test = np.hstack(
        [
            df_test[cols_not_scale].values,
            scaler.transform(df_test[cols_to_scale].values)
        ]
    )

    y_train = df_train[col_target].values
    y_test = df_test[col_target].values

    return x_train, y_train, x_test, y_test


def get_data(data_path):
    x_train, y_train, x_test, y_test = get_df_flights(data_path)
    return x_train, y_train, x_test, y_test


class TwoLayerRegressionModel(nn.Module):

    def __init__(self, use_cuda):
        global n_features, n_hidden, n_out
        super(TwoLayerRegressionModel, self).__init__()
        self.linear_1 = nn.Linear(n_features, n_hidden)
        self.linear_2 = nn.Linear(n_hidden, n_out)
        if use_cuda:
            self.cuda()

    def forward(self, x):
        # x * w + b
        return torch.relu(self.linear_2(torch.tanh(self.linear_1(x)))) + 1e-3


class OneLayerRegressionModel(nn.Module):
    def __init__(self, use_cuda):
        global n_features, n_out
        super(OneLayerRegressionModel, self).__init__()
        self.linear_1 = nn.Linear(n_features, n_out)
        if use_cuda:
            self.cuda()

    def forward(self, x):
        # x * w + b
        return torch.relu(self.linear_1(x)) + 1e-3


# def model(dataset_total_length, batch_size, x_data, y_data):
def model(dataset_total_length, x_data, y_data):
    priors = generate_nnet_priors()
    # scale = pyro.sample('sigma', Uniform(0., 10.))
    lifted_module = pyro.random_module('module', regression_model, priors)

    lifted_module_sample = lifted_module()

    # with pyro.plate('map', x_data.shape[0]): # no subsample
    # with pyro.plate('map', dataset_total_length, subsample_size=x_data.shape[0]): # dont do this.
    with pyro.plate('map', dataset_total_length, subsample=x_data):
        prediction_mean = lifted_module_sample(x_data).squeeze(-1)
        # pyro.sample('observations', LogNormal(prediction_mean, scale), obs=y_data)
        pyro.sample('observations', Exponential(prediction_mean), obs=y_data.squeeze(-1))
        return prediction_mean


def generate_nnet_priors():
    global nn_model_arg, n_features, n_hidden, n_out

    priors = {}

    if nn_model_arg == '1LAYER':
        w1_prior = Normal(torch.zeros(n_out, n_features), torch.ones(n_out, n_features)).to_event(1)
        b1_prior = Normal(torch.tensor([[0.]*n_out]), torch.tensor([[1.]*n_out])).to_event(1)
        priors = {'linear_1.weight': w1_prior, 'linear_1.bias': b1_prior}
    if nn_model_arg == '2LAYER':
        w1_prior = Normal(torch.zeros(n_hidden, n_features), torch.ones(n_hidden, n_features)).to_event(2)
        b1_prior = Normal(torch.tensor([[0.]*n_hidden]), torch.tensor([[1.]*n_hidden])).to_event(2)
        w2_prior = Normal(torch.zeros(n_out, n_hidden), torch.ones(n_out, n_hidden)).to_event(1)
        b2_prior = Normal(torch.tensor([[0.]*n_out]), torch.tensor([[1.]*n_out])).to_event(1)
        priors = {'linear_1.weight': w1_prior, 'linear_1.bias': b1_prior, 'linear_2.weight': w2_prior, 'linear_2.bias': b2_prior}

    return priors


guide = AutoDiagonalNormal(model)

def train(x_data, y_data, dataset_total_length, use_cuda, n_iterations):
    batch_size = 512
    optim = Adam({"lr": 0.005})

    train_ds = TensorDataset(x_data, y_data)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    svi = SVI(model, guide, optim, loss=Trace_ELBO(), num_samples=1000)
    pyro.clear_param_store()
    for j in range(n_iterations):
        start = time.time()
        loss = 0
        for x, y in train_dl:

            if use_cuda:
                x.cuda()
                y.cuda()

            # calculate the loss and take a gradient step
            loss += svi.step(dataset_total_length, x, y)
        elapsed_time = time.time() - start
        if j % 1 == 0:
            print("[iteration %04d] loss: %.4f took: %ds" % (j + 1, loss, elapsed_time))

    return svi


def wrapped_model(x_data, y_data):
    pyro.sample("prediction", Delta(model(x_data, y_data)))


if __name__ == '__main__':

    Normal(torch.zeros([4,5]), torch.ones([4,5])).sample()
    Normal(torch.zeros([4,5]), torch.ones([4,5])).to_event(1).sample()
    data_path = sys.argv[1]
    output_path = sys.argv[2]
    use_cuda_arg = sys.argv[3]
    nn_model_arg = sys.argv[4]
    n_hidden = int(sys.argv[5]) if sys.argv.__len__() > 5 is not None else None
    n_out = 1

    use_cuda = None
    if use_cuda_arg == 'CUDA':
        print('USING CUDA')
        use_cuda = True
    if use_cuda_arg == 'NOCUDA':
        print('NOT USING CUDA')
        use_cuda = False
    if use_cuda is None:
        print('use_cuda_arg [3] must be either "CUDA" or "NOCUDA"')
        exit(1)

    nn_model = None
    if nn_model_arg == '1LAYER':
        print('USING A 1LAYER NNET')
        nn_model = OneLayerRegressionModel
    if nn_model_arg == '2LAYER':
        nn_model = TwoLayerRegressionModel
        if n_hidden is None:
            print('TRYING TO USE A 2LAYER NNET WITH NONE HIDDEN UNITS')
            exit(1)
        print('USING A 2LAYER NNET')
    if nn_model is None:
        print('nn_model_arg [4] must be either "1LAYER" or "2LAYER"')
        exit(1)

    pyro.enable_validation(True)

    n_iterations = 250

    print('LOADING DATA')
    x_train, y_train, x_test, y_test = get_data(data_path)

    dataset_total_length = x_train.shape[0]

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32) + 1e-3

    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32) + 1e-3

    n_features = x_train.shape[1]
    regression_model = nn_model(use_cuda=False)

    print('NNET NAMED PARAMS')
    print([(n, v.shape) for n, v in regression_model.named_parameters()])

    print('RUNNING INFERENCE')
    svi = train(x_train_t, y_train_t, dataset_total_length, use_cuda, n_iterations)

    print('GETTING POSTERIOR TRACES')
    batch_size_test = 512
    test_ds = TensorDataset(x_test_t, y_test_t)
    test_dl = DataLoader(test_ds, batch_size=batch_size_test, shuffle=False)

    posterior = svi.run(dataset_total_length, x_test_t, y_test_t)

    print('COMPUTING POSTERIOR PREDICTIVE STATISTICS')
    sites = ['observations']
    marginal_preds = EmpiricalMarginal(posterior, sites)._get_samples_and_weights()[0].detach().cpu().numpy()

    print('EVALUATING')
    y_probs = pd.DataFrame(marginal_preds.reshape(-1, x_test_t.shape[0])).apply(lambda x: x >= 60).T.mean(axis=1).values
    df_res = pd.DataFrame()
    df_res['ARRIVAL_DELAY'] = y_test_t.view(-1).numpy()
    df_res['Y_PROB'] = y_probs

    df_eval = f_compute_uplift_curve(df_res)
    df_eval.to_csv(output_path, index=False)
