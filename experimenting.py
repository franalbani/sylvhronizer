#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def bits2square(bits, samples_per_bit):
    '''
    Return a sampled square signal.
    bits: numpy.int array
    '''
    N = np.floor(len(bits) * samples_per_bit)
    
    dt = 1.0 / samples_per_bit
    
    # Some initial delay:
    t0 = dt * 0.2 # np.random.uniform(0.1, 0.9) * dt

    ts = t0 + np.arange(N) * dt
    
    aux = np.floor(ts / (samples_per_bit*dt)).astype(np.int64)
    aux = aux[aux < len(bits)]
    square = bits[aux] * 2 - 1
    ts = ts[:len(square)]
    return (ts, square)


def sylvhronizer(signal, samples_per_bit):
    '''
    Returns a tuple of indices and corresponding
    samples.

    '''

    # Initial hint:
    one_every = samples_per_bit

    # `phase` should be interpreted in the
    # sampling frequency sense.
    phase = float(one_every)
    assert phase > 1

    # First sample is always chosen:
    ps = signal[0]
    ks = [0]
    output = [signal[0]]

    for k, s in enumerate(signal[1:], start=1):
        # New sample
        phase -= 1.0

        # Transition
        if np.sign(ps) != np.sign(s):
            phase = one_every / 2.0

        # Should we output
        if phase < 0.5:
            # We must output a sample:
            output.append(s)
            ks.append(k)

            # and fix the phase for the next:
            phase += one_every

        ps = s

    output = np.array(output)
    return ks, output


def experiment(samples_per_bit, B=10000):
    '''
    Generates a random signal and measures BER after synch.
    
    '''
    # FIXME: BER is not a proper meassurement of the quality of a synchronizer.

    bits = np.random.randint(0, 2, size=B)
 
    ts, square = bits2square(bits, samples_per_bit)
 
    ks, samples = sylvhronizer(square, samples_per_bit)
 
    recovered_bits = (0.5 * (samples + 1)).astype(np.int64)

    ber = np.mean(bits != recovered_bits)

    return ber


if __name__ == '__main__':

    spbs = np.arange(1.05, 5, 0.11)
    bers = []
    for spb in spbs:
        r = experiment(spb)
        bers.append(r)
    
    plt.plot(spbs, bers, 'o-')
    plt.xlabel('Samples per bit')
    plt.ylabel('BER')
    plt.xticks(spbs)
    plt.grid(True)
    plt.show()
