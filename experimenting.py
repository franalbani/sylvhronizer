#!/usr/bin/env python3

import numpy as np
from itertools import product
import matplotlib.pyplot as plt


def symbols2samples(symbols, samples_per_symbol):
    '''
    Returns samples of a signal composed by
    constant segments of symbols.

    symbols:            (numpy.int array)
    samples_per_symbol: (float)
    '''
    S = len(symbols)

    # Amount of samples needed to represent all symbols: 
    N = np.floor(S * samples_per_symbol)
    # TODO: floor or ceil?

    symbol_indexes = np.floor(np.arange(N) / samples_per_symbol).astype(np.int64)
    samples = symbols[symbol_indexes]
    return samples
 

def sylvhronizer(signal, samples_per_symbol):
    '''
    Returns the indexes of correct sampling
    for obtaining symbols.

    '''
    # Initial hint:
    one_every = samples_per_symbol
    # TODO: learn from transitions for jittering clocks

    # `phase` should be interpreted in the
    # sampling frequency sense.
    phase = float(one_every)
    assert phase > 1

    previous_sample = None
    ks = []

    for k, sample in enumerate(signal):
        # New sample
        phase -= 1.0

        # Transition
        if sample != previous_sample:
            phase = one_every / 2.0

        # Should we sample?
        if phase < 0.5:
            # We must sample here:
            ks.append(k)

            # and fix the phase for the next:
            phase += one_every

        previous_sample = sample

    return ks


def experiment(samples_per_symbol, block_size):

    symbols = np.random.randint(0, 2, size=block_size)
    signal = symbols2samples(symbols, samples_per_symbol)

    ks = sylvhronizer(signal, samples_per_symbol)

    rec_symbols = signal[ks]
    # FIXME: len(rec_symbols) != len(symbols) by as much as ~175

    # FIXME: this is not a sensible quality measurement for synchronizers.
    error_rate = np.mean(symbols != rec_symbols)

    return error_rate


if __name__ == '__main__':

    spss = np.arange(1.5, 8, 0.21)
    block_sizes = np.array([2])
    block_sizes = np.concatenate([block_sizes, np.arange(20, 100+1, 10)])
    block_sizes = np.concatenate([block_sizes, np.arange(100, 1000+1, 100)])
    block_sizes = block_sizes[1:]

    SER = -np.ones(shape=(len(spss), len(block_sizes)))
    exceptions = []
    M = 10

    # FIXME: this statistic was not carefully thought:
    for (i, sps), (j, block_size) in product(enumerate(spss), enumerate(block_sizes)):
        try:
            SER[i, j] = np.mean([experiment(sps, block_size) for _ in range(M)])
        except Exception as e:
            exceptions.append((sps, block_size, e))
   
    print(exceptions) 

    plt.imshow(SER, cmap='coolwarm')
    plt.colorbar()
    plt.title('Symbol Error rate')
    plt.xlabel('Block size')
    plt.ylabel('Samples per symbol')
    plt.xticks(range(len(block_sizes)), block_sizes, rotation='vertical')
    plt.yticks(range(len(spss)), map(lambda s: '%0.2f' % s, spss))
    plt.savefig('ser.png', bbox_inches='tight')
    plt.show()
