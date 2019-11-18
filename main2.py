# mpsk modulation
import cv2
from math import *
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty, BooleanProperty
from kivy.uix.image import Image
from  kivy.properties import NumericProperty
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
    noise_amp = ObjectProperty(None)
    img_src = StringProperty('cameraman.tif')
    img_noisy_src = StringProperty('cameraman_noisy.tif')
    ipimg = ObjectProperty(None)
    opimg = ObjectProperty(None)
    size_x  = NumericProperty(100)
    size_y = NumericProperty(100)
    size_x_noisy = NumericProperty(0)
    size_y_noisy = NumericProperty(0)
    def btnfunc(self):
        print('mpsk calc button pressed')
        #self.ipimg.size = (100,0)
        self.opimg.size_hint_y = None
        self.opimg.height = '0dp'
        self.size_x_noisy = 0
        self.size_y_noisy = 0
        self.do_layout()
        o = 0
        self.hint.text = "Message log: MPSK modulation started"
        try:
            o = int(self.order.text)
            global noiseamp
            noiseamp = float(self.noise_amp.text)
            if o % 2 != 0:
                raise ValueError  # app should not work even when the order of modulation is not a power of 2
            image_sig = mpskmod(o)
            mpskdemod(o, image_sig)

            self.size_x_noisy = 100
            self.size_y_noisy = 100
            self.opimg.reload()
            self.order.text = ""
            self.hint.text = "Message log: MPSK Modulation and Demodulation successfully performed"
        except ValueError:
            self.hint.text = "Message log: exponents of 2 expected"
            self.order.text = ""


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
    modulated_image = awgn(modulated_image, noiseamp)
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
    image = mat_bin(image)
    extra_bit = 0
    if (image.size) % (num_bit) != 0:
        extra_bit = num_bit - (image.size * 8) % (num_bit)
    image = np.concatenate((image, np.array(['0'] * extra_bit)))  # padding extra zeros to meet the requirement of mpsk
    print(image[len(image)-3:])
    mod_str = ""
    for num in image:
        mod_str = mod_str + num
    temp = [mod_str[i : i+num_bit] for i in range(0, len(mod_str), num_bit)]
    print(temp[len(temp)-3:])
    print('temp length: ', len(temp))
    final_image = [int(x, 2) for x in temp] #to hold the final version of the image with the symbols
    return final_image

def awgn(d, noiseamp):
    """Addition of AWGN to the image before transmission"""
    mean = 0
    var = 0.2
    sigma = var ** 0.5
    gauss = noiseamp * np.random.normal(mean, sigma, len(d))
    noisy = d + gauss
    return noisy

def mpskdemod(m, sig):
    """MPSK Demodulation on the image data"""
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
    print(demod_str[len(demod_str)-3:])
    for i in range(len(demod_str)):
        if len(demod_str[i]) < 8:
            del demod_str[i] #removal of the padded bits
    print("Removed extra bits")
    demod_dec = np.asarray(list(map(lambda x: int(x, 2), demod_str)))
    print(demod_dec.shape)
    demod_dec = np.reshape(demod_dec, (225, 225))
    print("plotting")
    cv2.imwrite('cameraman_noisy.tif', demod_dec)
    # img_noisy_src = "cameraman_noisy.tif"
    # plt.figure('output image')
    # plt.imshow(demod_dec, cmap='Greys')
    # plt.show()
    return demod_dec

if __name__ == "__main__":
    MyNewApp().run()
