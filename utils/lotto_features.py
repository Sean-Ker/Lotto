#%%
'''
you gimme raw data, I return features with multiple functions
'''
import random
from time import time
import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from talib import abstract
from tqdm import tqdm
import requests
from utils.util import DirInfo, clean_data, plot_ind, split_data
import math
import json
from joblib import Parallel, cpu_count, delayed
from datetime import date, datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import adfuller

# %% Indicator ==============================
class Indicator():
    train_ratio=0.80

    def __init__(self, series, targets, stats_method=np.mean, corr_method=None):
        assert type(series) == pd.Series
        assert corr_method in ['pearson', 'spearman', 'kendall', None]
        assert stats_method in [np.min, np.max, np.mean]

        self.stats_method = stats_method
        self.corr_method = corr_method

        self.data = series.dropna()
        self.targets = targets.loc[self.data.index]
        self.name = self.data.name

        self.train, self.dev, self.test = split_data(self.data.dropna(), self.train_ratio)
        self.get_stats()
        if corr_method != None:
            self.calc_corr()


    def good_attr(self, exclude_items=[]):
        assert type(exclude_items) == list
        exclude = ['data', 'targets','train','dev','test','corr', 'stats_method']
        if exclude_items:
            exclude += exclude_items
        good_attr = dict([(k,v) for k,v in self.__dict__.items() if k not in exclude])
        # good_attr = self.__dict__
        # [good_attr.pop(a, None) for a in ['data', 'targets','train','dev','test']]
        return good_attr

    def __repr__(self):
        return f"IND-{self.name}"

    def __str__(self):
        return str(self.good_attr())

    def get_stats(self):
        '''
        Adfuller.
        N0: THERE IS A UNIT ROOT (data is not stationary)
        Alternate Hyp: Data IS stationary.
        We could only reject the null hypothesis and say that data is stationary iff
        the uncertainty value (pval) lower than (1%-5%).
        Need [0, 0.05). Lower the better.
        '''
        adf_score = ADF_Scores(self.data, sample_size = 100, ntries = 10, method=self.stats_method)
        self.adf_pval = adf_score[0][self.name]
        self.adf_lag = adf_score[1][0]

        '''
        ks_2samp tells us how identical two distributions are.
        Null Hypothesis: The 2 Independent Random Variables are drawn from the same dist.
        Alternate Hyp: Not the same dist.
        We could reject the N0 when our uncertainty (pval) is lower than sig val (1%-5%)
        Need [1, 0.05). Higher the better.
        '''
        res = ks_2samp(self.train, self.dev), ks_2samp(self.train, self.test), ks_2samp(self.dev, self.test)
        # res = [sum(x)/len(x) for x in zip(*res)]
        # print(f'ks_2samp_stats stats: {res}')
        res = self.stats_method(res, axis=0)
        self.ks_stat = res[0]
        self.ks_pval = res[1]

    def to_df(self, include_corr=False):
        base = pd.Series(self.good_attr(['name', 'corr_method']))
        base.name = self.name
        if include_corr:
            base = base.append(self.corr)
        # return pd.concat([base, self.corr], axis=0)
        return base

    def get_plot(self, plot_line=False):
        if plot_line:
            plt.figure(figsize=(16, 5))
            plt.title(f'{self.name} KDE Plot')
            plt.suptitle('ks_2samp: {:0.2f} Adfuller: {:0.2f}'.format(self.ks_pval, self.adf_pval))
            plt.plot(np.arange(self.train.shape[0]), self.train, label=f'Train {self.name}')
            plt.plot(np.arange(self.train.shape[0], self.train.shape[0] + self.dev.shape[0]), self.dev, label=f'Dev {self.name}')
            plt.plot(np.arange(self.train.shape[0]+self.dev.shape[0], self.train.shape[0] + self.dev.shape[0] + self.test.shape[0]), self.test, label=f'Test {self.name}')
            plt.legend(loc='upper left')
            plt.show()

        plt.figure(figsize=(16, 5))
        plt.title(f'{self.name} KDE Plot', size=20)
        plt.suptitle('ks_2samp: {:0.2f} Adfuller: {:0.2f}'.format(self.ks_pval, self.adf_pval))

        sns.kdeplot(self.train, label=f'Train')
        sns.kdeplot(self.dev, label=f'Dev')
        sns.kdeplot(self.test, label=f'Test')

    def calc_corr(self):
        # np.corrcoef(ind.data, y.iloc[:,0][1:])[0][1]
        if hasattr(self, 'corr'): #and self.corr_method == method:
            return self.corr

        data = pd.concat([self.data, self.targets], axis=1)
        self.corr = abs(data.corr(self.corr_method)[self.name][1:])
        if type(self.corr) is pd.DataFrame:
            self.corr = self.corr.iloc[:,0]
        self.corr.index = ['corr_' + i for i in self.corr.index]

        mean, var = self.corr.mean(), self.corr.var()
        self.corr = self.corr.append(pd.Series([mean, var], index=['corr_mean', 'corr_var'])).astype(np.float32)
        # self.corr.name = f'{self.corr_method}_corr'
        self.corr.name = self.name
        self.corr_mean = mean
        return self.corr

def get_all_ind(features, y, stats_method = np.mean, corr_method = None):
    all_ind = []
    for col in tqdm(features.columns):
        all_ind.append(Indicator(features[col], y, stats_method, corr_method))
    return all_ind

def get_all_corr(all_ind):
    all_ind_corr = pd.DataFrame()
    for i in all_ind:
        if all_ind_corr.empty:
            all_ind_corr = pd.DataFrame(i.calc_corr())
        else:
            all_ind_corr = all_ind_corr.join(i.calc_corr())
    return all_ind_corr.transpose()

def get_all_df(all_ind):
    all_ind_df = pd.DataFrame()
    for i in all_ind:
        val = i.to_df(include_corr = False)
        if all_ind_df.empty:
            all_ind_df = pd.DataFrame(val)
        else:
            all_ind_df = all_ind_df.join(val)
    return all_ind_df.transpose()

def get_ind_by_name(all_ind, loi):
    if type(loi) is list:
        return [ind for ind in all_ind if ind.name in loi]
    if type(loi) is str:
        return [ind for ind in all_ind if ind.name == loi][0]

#
# lotto_columns = ['n'+str(i) for i in range(7)]
# all_columns = lotto_columns + ['nb', 'jackpot']

def ADF_Scores(data, sample_size, ntries = 3, verbose=0, method=np.ma, **kwargs):
    assert len(data) > sample_size and sample_size > 0
    assert method in [np.min, np.max, np.mean] # max pessimistic, min optimistic
    if sample_size > 0 and sample_size<=1:
        sample_size = int(len(data) * sample_size)

    adfuller_data = {}
    lag_used = []
    if verbose > 0:
        print('Calculating ADF scores...')

    if type(data) == pd.Series:
        data = pd.DataFrame(data)

    t = time()
    for col_name in data:
        all_scores = []
        for i in range(ntries):
            col = data[col_name]
            if sample_size > 0:
                cut = random.randint(0, int(len(col) - sample_size))
                res = adfuller(data[col_name].iloc[cut:(cut+sample_size)], **kwargs)
                all_scores.append((res[1], res[2]))
        all_scores = method(all_scores, axis=0).astype(np.float32)

        adfuller_data[col_name] = all_scores[0]
        lag_used.append(all_scores[1])
    if verbose>0:
        print(f'Done, took {time() - t} seconds.')
    return adfuller_data, lag_used

# a = ADF_Scores(df['sum'], sample_size = 0, maxlag = 15, autolag='bic')

def get_features(df, extra_features = False):
    data = df.copy()
    jack_features = ('jackpot' in df.columns)

    not_jack = [c for c in data.columns if c != 'jackpot']
    lotto_columns = [x for x in not_jack if x!='nb']
    data[not_jack] = data[not_jack].shift(1)


    if extra_features:
        xtra = get_extra_features(df, jack_features)
        cols = [c for c in xtra if c not in data]
        data = data.join(xtra[cols])
    return data


def get_extra_features(df, jack_features = True):
    data = df.copy()
    not_jack = [c for c in data.columns if c != 'jackpot']
    lotto_columns = [x for x in not_jack if x!='nb']
    data[not_jack] = data[not_jack].shift(1)

    # pct1 = data[lotto_columns + ['nb']].pct_change(1)
    # pct1.columns = [x+'_pct_ch1' for x in pct1.columns]
    # data = data.join(pct1)

    # data['product'] = np.product(data[lotto_columns + ['nb']],axis=1)/math.exp(20)
    data['avg_diff'] = data[lotto_columns].diff(axis=1).mean(axis=1)


    sin = np.sin(data[lotto_columns + ['nb']])
    sin.columns = [x+'_sin' for x in sin.columns]
    data = data.join(sin)

    data[not_jack] = data[not_jack].shift(1)
    data['avg'] = data[lotto_columns].mean(axis=1)

    data['vol'] = data[lotto_columns].var(axis=1)
    data['vol_change'] = data[lotto_columns].var(axis=1).diff()

    data['pct_change1'] = data[lotto_columns].pct_change(1).mean(axis=1)
    data['pct_change3'] = data[lotto_columns].pct_change(3).mean(axis=1)
    data['pct_change5'] = data[lotto_columns].pct_change(5).mean(axis=1)
    data['pct_change10'] = data[lotto_columns].pct_change(10).mean(axis=1)

    pct1 = data[lotto_columns + ['nb']].pct_change(1)
    pct1.columns = [x+'_pct_ch1' for x in pct1.columns]
    data = data.join(pct1)
    pct5 = data[lotto_columns + ['nb']].pct_change(5)
    pct5.columns = [x+'_pct_ch5' for x in pct5.columns]
    data = data.join(pct5)
    pct10 = data[lotto_columns + ['nb']].pct_change(10)
    pct10.columns = [x+'_pct_ch10' for x in pct10.columns]
    data = data.join(pct10)

    data['diff_sum1'] = data[lotto_columns].diff(1).sum(axis=1)
    data['diff_var1'] = data[lotto_columns].diff(1).var(axis=1)
    data['diff_sum3'] = data[lotto_columns].diff(3).sum(axis=1)
    data['diff_var3'] = data[lotto_columns].diff(3).var(axis=1)
    data['diff_sum5'] = data[lotto_columns].diff(5).sum(axis=1)
    data['diff_var5'] = data[lotto_columns].diff(5).var(axis=1)
    data['diff_sum10'] = data[lotto_columns].diff(10).sum(axis=1)
    data['diff_var10'] = data[lotto_columns].diff(10).var(axis=1)

    log_returns = np.log(data[lotto_columns] / data[lotto_columns].shift(1))
    data['lreturn_avg'] = log_returns.mean(axis=1)
    data['lreturn_var'] = log_returns.var(axis=1)

    # =========== Date Feature
    # data['stamp'] = np.diff(pd.DatetimeIndex(data.index).astype(int)/10**11)

    # =========== Jackpot Features
    if jack_features:
        data['jack_diff1'] = data['jackpot'].diff(1)
        data['jack_diff5'] = data['jackpot'].diff(5)
        data['jack_diff10'] = data['jackpot'].diff(10)
        data['jack_lreturn'] = np.log(data['jackpot'] / data['jackpot'].shift(1))

    return data

'''
BAD IND:
    data['sum'] = data[lotto_columns].sum(axis=1)
    data['lreturn_sum'] = log_returns.sum(axis=1)

    data['rol_max'] = data['n6'].rolling(5,min_periods=1).max()
    data['rol_min'] = data['n0'].rolling(5,min_periods=1).min()

    ewm = data[lotto_columns + ['nb']].ewm(5).mean()
    ewm.columns = [x+'_ewm_sum' for x in ewm.columns]
    data = data.join(ewm)

    data['ma3'] = data[lotto_columns].sum(axis=1).rolling(3,min_periods=1).mean()
    data['ma5'] = data[lotto_columns].sum(axis=1).rolling(5,min_periods=1).mean()
    data['ma10'] = data[lotto_columns].sum(axis=1).rolling(10,min_periods=1).mean()
    data['ma15'] = data[lotto_columns].sum(axis=1).rolling(15,min_periods=1).mean()

    data['jack_ma3'] = data['jackpot'].rolling(3,min_periods=1).mean()
    data['jack_ma5'] = data['jackpot'].rolling(5,min_periods=1).mean()
    data['jack_ma10'] = data['jackpot'].rolling(10,min_periods=1).mean()
    data['jack_ma15'] = data['jackpot'].rolling(15,min_periods=1).mean()
'''

def jackpot_features(df, game, save = True):
    data = df.copy()

    if 'jackpot' in data:
        df_index = [i for i, val in data['jackpot'].iteritems() if np.isnan(val)]
    else:
        df_index = data.index
    res = request_jackpot(df_index, game)

    if save:
        res.to_csv(DirInfo.DATA.path + game + "_jack.csv")
    data = data.join(res)
    return data

def request_jackpot(df_index, game):
    earliest_days = {"lmax": date(2009,9,25), "six49": date(1988,8,1), "dgrd": date(2016,10,19)}

    comp_index = [i for i in df_index if i>= earliest_days[game]]
    d = pd.DataFrame(comp_index, columns=['date'])
    d = d.set_index('date')
    d['jackpot'] = 0

    series_dict = dict(Parallel(n_jobs=-1, verbose=2)(delayed(agent_request_jackpot)(i, game) for i in comp_index))
    d['jackpot'].update(series_dict)
    d.sort_index(inplace = True)
    d = d.astype(np.float32)
    return d

def agent_request_jackpot(index, game):
    base_url = "https://www.playnow.com/services2/lotto/draw/{}/{}"
    jackpot_i = 0
    try:
        res = requests.get(base_url.format(game, index)).json()
        jackpot_i = res['gameBreakdown'][0]['prizeAmount'] / 10**6
    except json.JSONDecodeError:
        pass
    # logging.info('Finished index ' + str(index))
    return [index, jackpot_i]


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller


class FractionalDifferentiation:
    """
    FractionalDifferentiation class encapsulates the functions that can
    be used to compute fractionally differentiated series.
    """

    @staticmethod
    def get_weights(diff_amt, size):
        """
        Advances in Financial Machine Learning, Chapter 5, section 5.4.2, page 79.

        The helper function generates weights that are used to compute fractionally
        differentiated series. It computes the weights that get used in the computation
        of  fractionally differentiated series. This generates a non-terminating series
        that approaches zero asymptotically. The side effect of this function is that
        it leads to negative drift "caused by an expanding window's added weights"
        (see page 83 AFML)

        When diff_amt is real (non-integer) positive number then it preserves memory.

        The book does not discuss what should be expected if d is a negative real
        number. Conceptually (from set theory) negative d leads to set of negative
        number of elements. And that translates into a set whose elements can be
        selected more than once or as many times as one chooses (multisets with
        unbounded multiplicity) - see http://faculty.uml.edu/jpropp/msri-up12.pdf.

        :param diff_amt: (float) Differencing amount
        :param size: (int) Length of the series
        :return: (np.ndarray) Weight vector
        """

        # The algorithm below executes the iterative estimation (section 5.4.2, page 78)
        weights = [1.]  # create an empty list and initialize the first element with 1.
        for k in range(1, size):
            weights_ = -weights[-1] * (diff_amt - k + 1) / k  # compute the next weight
            weights.append(weights_)

        # Now, reverse the list, convert into a numpy column vector
        weights = np.array(weights[::-1]).reshape(-1, 1)
        return weights

    @staticmethod
    def frac_diff(series, diff_amt, thresh=0.01):
        """
        Advances in Financial Machine Learning, Chapter 5, section 5.5, page 82.

        References:
        https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
        https://wwwf.imperial.ac.uk/~ejm/M3S8/Problems/hosking81.pdf
        https://en.wikipedia.org/wiki/Fractional_calculus

        The steps are as follows:
        - Compute weights (this is a one-time exercise)
        - Iteratively apply the weights to the price series and generate output points

        This is the expanding window variant of the fracDiff algorithm
        Note 1: For thresh-1, nothing is skipped
        Note 2: diff_amt can be any positive fractional, not necessarility bounded [0, 1]

        :param series: (pd.DataFrame) A time series that needs to be differenced
        :param diff_amt: (float) Differencing amount
        :param thresh: (float) Threshold or epsilon
        :return: (pd.DataFrame) Differenced series
        """

        # 1. Compute weights for the longest series
        weights = get_weights(diff_amt, series.shape[0])

        # 2. Determine initial calculations to be skipped based on weight-loss threshold
        weights_ = np.cumsum(abs(weights))
        weights_ /= weights_[-1]
        skip = weights_[weights_ > thresh].shape[0]

        # 3. Apply weights to values
        output_df = {}
        for name in series.columns:
            series_f = series[[name]].fillna(method='ffill').dropna()
            output_df_ = pd.Series(index=series.index, dtype='float64')

            for iloc in range(skip, series_f.shape[0]):
                loc = series_f.index[iloc]

                # At this point all entries are non-NAs so no need for the following check
                # if np.isfinite(series.loc[loc, name]):
                output_df_[loc] = np.dot(weights[-(iloc + 1):, :].T, series_f.loc[:loc])[0, 0]

            output_df[name] = output_df_.copy(deep=True)
        output_df = pd.concat(output_df, axis=1)
        return output_df

    @staticmethod
    def get_weights_ffd(diff_amt, thresh, lim):
        """
        Advances in Financial Machine Learning, Chapter 5, section 5.4.2, page 83.

        The helper function generates weights that are used to compute fractionally
        differentiate dseries. It computes the weights that get used in the computation
        of fractionally differentiated series. The series is of fixed width and same
        weights (generated by this function) can be used when creating fractional
        differentiated series.
        This makes the process more efficient. But the side-effect is that the
        fractionally differentiated series is skewed and has excess kurtosis. In
        other words, it is not Gaussian any more.

        The discussion of positive and negative d is similar to that in get_weights
        (see the function get_weights)

        :param diff_amt: (float) Differencing amount
        :param thresh: (float) Threshold for minimum weight
        :param lim: (int) Maximum length of the weight vector
        :return: (np.ndarray) Weight vector
        """

        weights = [1.]
        k = 1

        # The algorithm below executes the iterativetive estimation (section 5.4.2, page 78)
        # The output weights array is of the indicated length (specified by lim)
        ctr = 0
        while True:
            # compute the next weight
            weights_ = -weights[-1] * (diff_amt - k + 1) / k

            if thresh > 0 and abs(weights_) < thresh:
                break

            weights.append(weights_)
            k += 1
            ctr += 1
            if ctr == lim - 1:  # if we have reached the size limit, exit the loop
                break

        # Now, reverse the list, convert into a numpy column vector
        weights = np.array(weights[::-1]).reshape(-1, 1)
        return weights

    @staticmethod
    def frac_diff_ffd(series, diff_amt, thresh=1e-5):
        """
        Advances in Financial Machine Learning, Chapter 5, section 5.5, page 83.

        References:

        * https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
        * https://wwwf.imperial.ac.uk/~ejm/M3S8/Problems/hosking81.pdf
        * https://en.wikipedia.org/wiki/Fractional_calculus

        The steps are as follows:

        - Compute weights (this is a one-time exercise)
        - Iteratively apply the weights to the price series and generate output points

        Constant width window (new solution)
        Note 1: thresh determines the cut-off weight for the window
        Note 2: diff_amt can be any positive fractional, not necessarity bounded [0, 1].

        :param series: (pd.DataFrame) A time series that needs to be differenced
        :param diff_amt: (float) Differencing amount
        :param thresh: (float) Threshold for minimum weight
        :return: (pd.DataFrame) A data frame of differenced series
        """

        # 1) Compute weights for the longest series
        weights = get_weights_ffd(diff_amt, thresh, series.shape[0])
        width = len(weights) - 1

        if type(series) == pd.Series:
            series = pd.DataFrame(series.values, index = series.index, columns = [series.name])

        # 2) Apply weights to values
        # 2.1) Start by creating a dictionary to hold all the fractionally differenced series
        output_df = {}

        # 2.2) compute fractionally differenced series for each stock
        for name in series.columns:
            series_f = series[[name]].fillna(method='ffill').dropna()
            temp_df_ = pd.Series(index=series.index, dtype='float64')
            for iloc1 in range(width, series_f.shape[0]):
                loc0 = series_f.index[iloc1 - width]
                loc1 = series.index[iloc1]

                # At this point all entries are non-NAs, hence no need for the following check
                # if np.isfinite(series.loc[loc1, name]):
                temp_df_[loc1] = np.dot(weights.T, series_f.loc[loc0:loc1])[0, 0]

            output_df[name] = temp_df_.copy(deep=True)

        # transform the dictionary into a data frame
        output_df = pd.concat(output_df, axis=1)
        return output_df


def get_weights(diff_amt, size):
    """ This is a pass-through function """
    return FractionalDifferentiation.get_weights(diff_amt, size)


def frac_diff(series, diff_amt, thresh=0.01):
    """ This is a pass-through function """
    return FractionalDifferentiation.frac_diff(series, diff_amt, thresh)


def get_weights_ffd(diff_amt, thresh, lim):
    """ This is a pass-through function """
    return FractionalDifferentiation.get_weights_ffd(diff_amt, thresh, lim)


def frac_diff_ffd(series, diff_amt, thresh=1e-5):
    """
    Advances in Financial Machine Learning, Chapter 5, section 5.5, page 83.

    References:

    * https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
    * https://wwwf.imperial.ac.uk/~ejm/M3S8/Problems/hosking81.pdf
    * https://en.wikipedia.org/wiki/Fractional_calculus

    The steps are as follows:

    - Compute weights (this is a one-time exercise)
    - Iteratively apply the weights to the price series and generate output points

    Constant width window (new solution)
    Note 1: thresh determines the cut-off weight for the window
    Note 2: diff_amt can be any positive fractional, not necessarity bounded [0, 1].

    :param series: (pd.Series) A time series that needs to be differenced
    :param diff_amt: (float) Differencing amount
    :param thresh: (float) Threshold for minimum weight
    :return: (pd.DataFrame) A data frame of differenced series
    """
    return FractionalDifferentiation.frac_diff_ffd(series, diff_amt, thresh)


def plot_min_ffd(series):
    results = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])

    # Iterate through d values with 0.1 step
    data = np.log(series.copy())
    data.index = pd.DatetimeIndex(data.index)
    data = data.resample('1D').last()  # Downcast to daily obs
    data.dropna(inplace=True)
    for d_value in np.linspace(0, 1, 6):
        # Applying fractional differentiation
        differenced_series = frac_diff_ffd(data, diff_amt=d_value, thresh=1e-5).dropna()

        # Correlation between the original and the differentiated series
        corr = np.corrcoef(data.loc[differenced_series.index], differenced_series, rowvar=False)[0, 1]
        # Applying ADF
        differenced_series = adfuller(differenced_series, maxlag=1, regression='c', autolag=None)

        # Results to dataframe
        results.loc[d_value] = list(differenced_series[:4]) + [differenced_series[4]['5%']] + [corr]  # With critical value

    # Plotting
    plot = results[['adfStat', 'corr']].plot(secondary_y='adfStat', figsize=(10, 8))
    plt.axhline(results['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')

    return plot, results

def plot_weights(dRange, nPlots, size):
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_=get_weights(d,size=size)
        w_=pd.DataFrame(w_,index=range(w_.shape[0])[::-1], columns=[d])
        w=w.join(w_,how='outer')
    ax=w.plot()
    ax.legend()
    plt.show()

plot_weights([0,1], nPlots=11, size=100)


# %%

