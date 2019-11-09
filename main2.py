# mpsk modulation
import kivy
from math import *
from numpy import arange
from matplotlib import pyplot as plt
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty

class MyLayout(FloatLayout):
    name = ObjectProperty(None)
    btn = ObjectProperty(None)

    def btnfunc(self):
        print(self.name.text)
        mpskmod(int(self.name.text))

class MyNewApp(App):
    def build(self):
        return MyLayout()

def mpskmod(m):
    d = [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0]  # data

    num_bit = int(log2(m))

    extra_bit = m - len(d) % m

    d = d + [0] * extra_bit

    ind = [i for i in range(0, len(d)) if i % num_bit == 0]

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
    fc = 2  # Hz
    A = 2  # V

    phases = arange(0, 2 * pi, (2 * pi) / m)
    time = arange(0, 1 / (512) + 1, 1 / (512))

    plot_val = []
    # x_val = []

    for ind in num_arr:
        angle = phases[ind]  # ind is the symbol formed from the group of bits
        for t in time:
            plot_val.append(A * cos(2 * pi * fc * t + angle))

    plt.plot(plot_val)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    MyNewApp().run()