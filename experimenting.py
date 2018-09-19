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

    # `phase` should be interpreted in the
    # sampling frequency sense.
    phase = float(one_every)
    assert phase > 1

    # We will discard all samples before first symbol transition:
    k0 = min(k for k, d in enumerate(np.diff(signal)) if d != 0)

    # First sample is always chosen:
    ks = [k0]
    ps = signal[k0]

    for k, s in enumerate(signal[(k0+1):], start=k0+1):
        # New sample
        phase -= 1.0

        # Transition
        if ps != s:
            phase = one_every / 2.0

        # Should we sample?
        if phase < 0.5:
            # We must sample here:
            ks.append(k)

            # and fix the phase for the next:
            phase += one_every

        ps = s

    return ks


def experiment(samples_per_symbol, block_size):

    symbols = np.random.randint(0, 2, size=block_size)
    signal = symbols2samples(symbols, samples_per_symbol)

    ks = sylvhronizer(signal, samples_per_symbol)

    # First recovered symbol:
    frs = int(np.floor(ks[0] / samples_per_symbol))
    rec_symbols = signal[ks]

    error_rate = np.mean(symbols[frs:] != rec_symbols)

    aux = len(symbols[frs:]) - len(rec_symbols)
    if aux != 0:
        print('Warning: rec_symbols length differ in %d' % aux)

    return error_rate


if __name__ == '__main__':

    spss = np.arange(1.05, 5, 0.11)
    block_sizes = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])

    BER = -np.ones(shape=(len(spss), len(block_sizes)))
    exceptions = []
    M = 10

    for (i, sps), (j, block_size) in product(enumerate(spss), enumerate(block_sizes)):
        try:
            BER[i, j] = np.mean([experiment(sps, block_size) for _ in range(M)])
        except Exception as e:
            exceptions.append((sps, block_size, e))
   
    print(exceptions) 
    plt.imshow(BER, cmap='coolwarm')
    plt.colorbar()
    plt.xlabel('Block size')
    plt.ylabel('Samples per bit')
    plt.xticks(range(len(block_sizes)), block_sizes, rotation='vertical')
    plt.yticks(range(len(spss)), map(lambda s: '%0.2f' % s, spss))
    plt.show()
