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

    # We will discard all samples before first transition:
    # FIXME: this requires pre-slicing
    k0 = min(k for k, d in enumerate(np.diff(signal)) if d != 0)

    # First sample is always chosen:
    ks = [k0]
    ps = signal[k0]
    output = [ps]

    for k, s in enumerate(signal[(k0+1):], start=k0+1):
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
 
    ks, _ = sylvhronizer(square, samples_per_bit)
 
    recovered_bits = (0.5 * (square[ks] + 1)).astype(np.int64)

    # first_recovered_bit
    frb = int(np.floor(ks[0] / samples_per_bit))

    ber = np.mean(bits[frb:] != recovered_bits)

    return ber


def aux(samples_per_bit):
    B = 20
    bits = np.random.randint(0, 2, size=B)
 
    ts, square = bits2square(bits, samples_per_bit)
 
    ks, _ = sylvhronizer(square, samples_per_bit)

    recovered_bits = (0.5 * (square[ks] + 1)).astype(np.int64)

    # first_recovered_bit
    frb = int(np.floor(ks[0] / samples_per_bit))

    ber = np.mean(bits[frb:] != recovered_bits)

    plt.plot(ts, square, '.-', color='blue')
    plt.plot(ts[ks], square[ks], 'o', color='red')
    plt.title('%r\n%r' % (list(bits[frb:]), list(recovered_bits)))
    plt.grid(True)
    plt.show()
    

if __name__ == '__main__':

    spbs = np.arange(1.05, 5, 0.11)
    b_sizes = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
    BER = np.empty(shape=(len(spbs), len(b_sizes)))
    for i, spb in enumerate(spbs):
        for j, bs in enumerate(b_sizes):
            BER[i, j] = experiment(spb, B=bs)
    
    plt.imshow(BER)
    plt.colorbar()
#     plt.plot(spbs, bers, 'o-')
    plt.xlabel('Block size')
    plt.ylabel('Samples per bit')
#    plt.grid(True)
    plt.xticks(range(len(b_sizes)), b_sizes)
    plt.yticks(range(len(spbs)), map(lambda s: '%0.2f' % s, spbs))
    plt.show()
