"""
Tausworthe Random Number Generator with supporting functions as described in Module 6

- Eric Goldberg
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pandas import DataFrame


def tausworthe_gen(n=1, seed=None):
    """
    Generates n Unif(0, 1) pseudo-random numbers using the Tausworthe Method in module 6

    :param n:
        (int) number of PRNs to generate
    :param seed:
        any integer between 0 and 2^37 - 1
    :return:
        numpy array of n Unif(0, 1) PRNs
    """
    assert type(n) is int and n > 0, "n must be a positive integer"

    # set a random seed if none given
    if not seed:
        seed = int(np.random.randint(0, 2 ** 36 - 1, dtype=np.int64))

    assert type(seed) is int and seed < 2 ** 37, "seed must be a positive integer < 2^37"
    # set initial sequence of B_1 to B_q, note that first bit is 1 here which will cause the first PRN to be > 0.5
    bit = bin(2**37 + seed).split('b')[-1]

    # reverse to resolve above issue (though last bit in initial sequence will always be 1)
    bit = bit[::-1]

    # set r, B_1,...B_q, q, and l
    r = 31
    bits = bit
    q = len(bit)
    l = 23

    # generate needed bits depending on number of PRNs requested
    while (len(bits)/l) < n:
        B_i = int(bits[len(bits) - q]) ^ int(bits[len(bits) - r])
        bits += str(B_i)

    # calculate PRNs
    prns = []
    for i in range(n):
        prn = int(bits[i*l:(i+1)*l], 2)/(2**l)
        prns.append(prn)

    return np.array(prns)


def get_frequencies(x, k=1000):
    """
    Tallies how many observations fall into k evenly spaced sub-intervals between 0 and 1 in preparation for chi-square
    goodness-of-fit test.

    :param x:
        numpy array of Unif(0, 1) PRNs
    :param k:
        (int) of sub-intervals, must be less than size of PRN array
    :return:
        numpy array of number of observations in each sub-interval
    """

    assert x.shape[0] > k, "k must be less than the number of observations"
    assert x.shape[0]/k > 5, "expected number of obs in each sub-interval must be > 5, increase observations or " \
                             "decrease k"

    start = 0
    interval = 1/k
    end = start + interval
    freq = []
    while end < 1:
        freq.append(x[(x >= start) & (x < end)].shape[0])
        start = end
        end = start + interval
    return np.array(freq)


def plot_prns(x, ax=None, num_points=None):
    """
    Plots adjacent PRNs on the unit square

    :param x:
        numpy array of PRNs generated from our tausworthe generator!
    :param ax:
        ax to plot on if using subplots
    :param num_points:
        int of number of points to plot, default will plot all
    :return:
        matplotlib plot of adjacent PRNs on the unit square (plt.show() needs to be called if using subplots)
    """
    assert x.shape[0] > 2, "X must be a np array with len > 2"
    z = x[:-1]
    y = x[1:]

    if num_points:
        assert type(num_points) is int and num_points > 0, "num_points must be a positive integer"

        if ax:
            ax.scatter(z[:num_points], y[:num_points])
            ax.set_title('First {} adjacent PRNs'.format(num_points))
        else:
            plt.scatter(z[:num_points], y[:num_points])
            plt.title('First {} adjacent PRNs'.format(num_points))
            plt.show()
    else:
        if ax:
            ax.scatter(z, y)
            ax.set_title('All adjacent PRNs')
        else:
            plt.scatter(z, y)
            plt.title('All adjacent PRNs')
            plt.show()


def runs_up_down(x, alpha=.05):
    """
    Performs the Runs test "Up and Down" for independence as described in Module 6

    :param x:
        numpy array of PRNs
    :param alpha:
        alpha level
    :return:
        data frame of result of Runs test up and down
    """
    assert x.shape[0] > 19, "not enough PRNs for a test"
    assert 0 < alpha < 1, "invalid alpha"

    z = x[:-1]
    y = x[1:]

    # get sign of change between adjacent PRNs
    runs = np.sign(y-z)

    # check for any place which had equal adjacent PRNs and set to + change
    if (np.unique(runs) == 0).any():
        runs = np.where(runs == 0, 1, runs)

    # get number of runs up and down
    num_runs = 0

    for i in range(len(runs)-1):
        if runs[i] != runs[i+1]:
            num_runs += 1

    # calculate expected value and variance of A
    expected_runs = (2*x.shape[0] - 1)/3
    variance_runs = (16*x.shape[0] - 29)/90

    # calculate Z_0 and Z_alpha/2
    z_null = np.abs((num_runs - expected_runs)/(variance_runs**0.5))
    z_alph = stats.norm.ppf(1-alpha/2)
    if z_null > z_alph:
        result = 'reject'
    else:
        result = 'accept'

    return DataFrame(data={r'$|Z_0|$': z_null, r'$z_{\alpha/2}$': z_alph, 'result': result}, index=['results'])


def runs_mean(x, alpha=.05):
    """
    Performs the Runs test "Above and Below the mean" for independence as described in Module 6

    :param x:
        numpy array of PRNs
    :param alpha:
        alpha level
    :return:
        data frame of result of Runs above and below the mean
    """
    assert x.shape[0] > 19, "not enough PRNs for a test"
    assert 0 < alpha < 1, "invalid alpha"

    the_mean = 0.5

    # get sign of PRN - mean
    signs = np.sign(x - the_mean)

    # check for any place which had PRNs equal to the mean, set to + change
    if (np.unique(signs) == 0).any():
        signs = np.where(signs == 0, 1, signs)

    # get number of runs above and below
    num_runs = 0

    for i in range(len(signs)-1):
        if signs[i] != signs[i+1]:
            num_runs += 1

    # calculate expected value and variance of B according to Module 6 equations
    n_1 = signs[signs > 0].shape[0]
    n_2 = signs.shape[0] - n_1

    expected_runs = ((2*n_1*n_2)/signs.shape[0]) + 0.5
    variance_runs = ((2*n_1*n_2*(2*n_1*n_2 - signs.shape[0]))/(signs.shape[0]**2 * (signs.shape[0]-1)))

    # calculate Z_0 and Z_alpha/2
    z_null = np.abs((num_runs - expected_runs)/(variance_runs**0.5))
    z_alph = stats.norm.ppf(1-alpha/2)
    if z_null > z_alph:
        result = 'reject'
    else:
        result = 'accept'

    return DataFrame(data={r'$|Z_0|$': z_null, r'$z_{\alpha/2}$': z_alph, 'result': result}, index=['results'])


def box_muller(x, num_points=None):
    """
    Uses Box-Muller method to transform Unif(0, 1)'s to Norm(0 ,1)'s.

    :param x:
        numpy array of Unif(0, 1)'s
    :param num_points:
        int of number of Norm(0, 1)'s desired. Must be less than the length of input array x
    :return:
        numpy array of Unif(0, 1)'s of full or specified length
    """

    # do box-muller calculations
    if num_points:
        norm_1 = [np.sqrt(-2*np.log(i))*np.cos(2*np.pi*j) for i, j in zip(x[:num_points:2], x[1:num_points:2])]
        norm_2 = [np.sqrt(-2*np.log(i))*np.sin(2*np.pi*j) for i, j in zip(x[:num_points:2], x[1:num_points:2])]

    else:
        norm_1 = [np.sqrt(-2 * np.log(i)) * np.cos(2 * np.pi * j) for i, j in zip(x[::2], x[1::2])]
        norm_2 = [np.sqrt(-2 * np.log(i)) * np.sin(2 * np.pi * j) for i, j in zip(x[::2], x[1::2])]

    # put the outputs back together in original order
    for i, v in enumerate(norm_2):
        norm_1.insert(2*i+1, v)

    return np.array(norm_1)
