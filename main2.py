# mpsk modulation
import cv2
from math import *
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty

fc = 2  # Carrier freq Hz
A = 2  # V
b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
N = int(np.ceil((4 / b)))
n = np.arange(N)
bit_rate = 256

class MyLayout(FloatLayout):
    order = ObjectProperty(None)
    btn = ObjectProperty(None)
    hint = ObjectProperty(None)

    def btnfunc(self):
        print('mpsk calc button pressed')
        o = 0
        try:
            o = int(self.order.text)
            if o % 2 != 0:
                raise ValueError  # app should not work even when the order of modulation is not a power of 2
            image_sig = mpskmod(o)
            self.order.text = ""
            self.hint.text = "Message log: MPSK Modulation successfully performed"
        except ValueError:
            self.hint.text = "Message log: exponents of 2 expected"
            self.order.text = ""
        mpskdemod(o, image_sig)

class MyNewApp(App):
    def build(self):
        return MyLayout()


def mpskmod(m):
    """Performs the MPSK modulation on image data"""
    num_bit = int(log2(m))
    image_data = image_process(num_bit)  # data ready for mpsk modulation

    # carrier generation
    fc = 2  # Hz
    A = 2  # V

    phases = arange(0, 2 * pi, (2 * pi) / m)
    mpsk_arr = [phases[ind] for ind in image_data]  # phase for each data element in the image array
    time = arange(0, 1, 1 / bit_rate)
    cos_input = []
    cos_input = list(map(lambda x: x + 2 * pi * fc * time, mpsk_arr))
    modulated_image = np.cos(cos_input)
    modulated_image = np.reshape(modulated_image, modulated_image.size)
    modulated_image = awgn(modulated_image)
    return modulated_image

def mat_bin(mat):
    """Function to convert the decimal values in a matrix to binary format"""
    s = np.shape(mat)
    temp = np.empty(s[0], dtype='object')  # each object stored in the numpy array would be of type 16 character Unicode string
    for i in range(len(mat)):
        temp[i] = str(bin(mat[i])[2:])
        if len(temp[i]) != 8:  # 8 bits per element are expected in the output matrix
            temp[i] = '0' * (8 - len(temp[i])) + temp[i]
    return temp

def image_process(num_bit):
    """Fetch the image and convert it into an array ready for mpsk modulation"""
    image = cv2.imread('cameraman.tif')
    print(image.shape)
    image = image[:, :, 0]  # reducing the three time repetition to once
    image_size = np.shape(image)

    plt.figure('input image')
    plt.imshow(image, cmap='Greys')
    image = np.reshape(image, (image_size[0] * image_size[1],))
    extra_bit = 0
    if (image.size) % (num_bit) != 0:
        extra_bit = (image.size) % (num_bit)
    image = np.concatenate((image, np.array([0] * extra_bit)))  # padding extra zeros to meet the requirement of mpsk
    image = [int(ele) for ele in image]
    image = mat_bin(image)

    bits_per_row = 8 + extra_bit  # var holding the value of number of bits per row in the image matrix after mat_bin() func call
    temp = np.empty(bits_per_row // num_bit,
                    dtype='object')  # to hold the values of columns extracted with respect to bits required for modulation
    start = 0
    final_image = []  # to hold the final version of the reshape of the image
    indices = []
    for i in range(bits_per_row // num_bit):
        t = [ele[start:start + num_bit] for ele in image]
        temp[i] = t
        for j in range(len(t)):
            if t[j] == '':
                print(j)
                break
        start += num_bit
    for i in range(image.size):
        for j in range(bits_per_row // num_bit):
            final_image.append(int(temp[j][i], base=2))
    return final_image

def awgn(d):
    """Addition of AWGN to the image before transmission"""
    mean = 0
    var = 0.2
    sigma = var ** 0.5
    gauss = 20 * np.random.normal(mean, sigma, len(d))
    noisy = d + gauss
    return noisy

def mpskdemod(m, sig):
    print(sig.size)
    num_bit = int(log2(m))
    phases = arange(0, 2 * pi, (2 * pi) / m)
    num_sym_tran = int(len(sig)/bit_rate)
    time = arange(0, 1, 1 / bit_rate)
    sym_tran = np.split(sig, num_sym_tran)
    demod = []
    for i in sym_tran:
        op = []
        for j in phases:
            temp = np.dot(np.cos(2*pi*fc*time+j), np.asarray(i))
            op.append(temp)
        demod.append(np.argmax(op))
    demod = np.asarray(demod)
    demod_str = ""
    print(demod.size, num_bit)
    for i in demod:
        demod_str = demod_str + (np.binary_repr(i, num_bit))
    print(len(demod_str))
    demod_str = [demod_str[i:i+8] for i in range(0, len(demod_str), 8)]
    print(demod_str)
    demod_dec = np.asarray(list(map(lambda x: int(x, 2), demod_str)))
    print(demod_dec)
    demod_dec = np.reshape(demod_dec, (256, 256))
    print(demod_dec)
    plt.figure('output image')
    plt.imshow(demod_dec, cmap='Greys')
    plt.show()
    return demod_dec

if __name__ == "__main__":
    MyNewApp().run()
