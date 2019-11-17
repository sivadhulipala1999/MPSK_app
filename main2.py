# mpsk modulation
import kivy
from math import *
from numpy import arange
from matplotlib import pyplot as plt
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
import numpy as np

fc = 2  # Carrier freq Hz
A = 2  # V
b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
N = int(np.ceil((4 / b)))
n = np.arange(N)
bit_rate = 512*8

class MyLayout(FloatLayout):
    name = ObjectProperty(None)
    btn = ObjectProperty(None)

    def btnfunc(self):
        print(self.name.text)
        m = int(self.name.text)
        mod = mpskmod(m)
        mpskdemod(m,mod)

class MyNewApp(App):
    def build(self):
        return MyLayout()

def mpskmod(m):
    d = [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0]  # data

    num_bit = int(log2(m))

    extra_bit = m - len(d) % m

    d = d + [0] * extra_bit

    ind = [i for i in range(0, len(d)) if i % num_bit == 0]
    print(ind)

    num_arr = []

    print(d)

    for i in ind:
        temp = ''
        # print('i ={}'.format(i), end = ' ')
        for j in range(i, i + num_bit):
            # print(' j =', j, end = ' ')
            temp += str(d[j])
            # print(' temp =',temp,end=' ')
        num = int(temp, 2)
        num_arr.append(num)
    print(num_arr)

    # carrier generation

    phases = arange(0, 2 * pi, (2 * pi) / m)
    time = arange(0, 1 / bit_rate + 1, 1 / bit_rate)

    plot_val = []
    # x_val = []

    for ind in num_arr:
        angle = phases[ind]  # ind is the symbol formed from the group of bits
        for t in time:
            plot_val.append(A * cos(2 * pi * fc * t + angle))
    plt_time = arange(0, (1 / bit_rate + 1)*len(num_arr), 1 / bit_rate)

    plt.figure("b")
    plot_val = awgn(plot_val)
    plt.plot(plt_time, plot_val)
    plt.grid(True)
    plt.show()
    return plot_val


def awgn(d):
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, len(d))
    noisy = d + gauss
    return noisy

def mpskdemod(m, sig):
    num_bit = int(log2(m))
    sig = np.asarray(sig)
    phases = arange(0, 2 * pi, (2 * pi) / m)
    num_sym_tran = int(len(sig)/bit_rate)
    time = arange(0, (1 / bit_rate + 1), 1 / bit_rate)
    plt_time = arange(0, (1 / bit_rate + 1) *8, 1 / bit_rate)
    h = np.sinc(2 * 4 * (n - (N - 1) / 2))
    # sig = np.convolve(sig, h, 'same')*pow(10,15)
    # plt.figure("a")
    # plt.plot(plt_time,sig)
    # plt.grid(True)
    # plt.show()
    sym_tran = np.split(sig, num_sym_tran)
    demod = []
    for i in sym_tran:
        op = []
        for j in phases:
            temp = np.dot(np.cos(2*pi*fc*time+j), np.asarray(i))
            op.append(temp)
        demod.append(np.argmax(op))
    demod_str = ""
    for i in demod:
        demod_str = demod_str + (np.binary_repr(i,num_bit))
    demod_sig = []

        # plt.plot(s)
        # plt.grid(True)
        # plt.show()
    print("Demodulated "+demod_str)

if __name__ == "__main__":
    MyNewApp().run()
