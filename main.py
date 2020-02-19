import queue
import win32gui
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tkinter import *
from PIL import Image, ImageGrab

def load_image(direct):
    img = Image.open(direct).convert('L')
    #img = img.resize( (28, 28), Image.ANTIALIAS)
    pix = 255 - np.array(img)
    # outfile = open("tt.txt", "w")
    # for i in range(300):
    #     for j in range(300):
    #         outfile.write("%4d" % pix[i][j])
    #     outfile.write("\n")
    # outfile.close()
    return pix

def show(img):
    fig, ax = plt.subplots()
    ax.imshow(img.reshape( (28, 28) ), cmap = 'gray')
    plt.show()

img = load_image('image/test.png')

model = load_model('ANN_model/offical2_model.h5')

def prediction(img):
    pred = model.predict(img.reshape(1, 28, 28, 1) / 255)
    return np.argmax(pred), np.max(pred)

""" --------- Detecting digit in an image -----------"""
def digit_detected(img):
    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1, 0, 1, -1, 1, -1, 0, 1]
    visited = [[0 for _ in range(img.shape[0])] for __ in range(img.shape[1])]

    def InsideImage(x, y):
        return 0 <= x < img.shape[0] and 0 <= y < img.shape[1]

    def BFS(sx, sy):
        top, left, bot, right = sx, sy, sx, sy
        Q = queue.Queue(maxsize = 300 * 300)
        visited[sx][sy] = 1
        Q.put((sx, sy))
        while (not Q.empty()):
            x, y = Q.get()
            top = min(top, x)
            bot = max(bot, x)
            left = min(left, y)
            right = max(right, y)
            for i in range(8):
                u = x + dx[i]
                v = y + dy[i]
                if (not InsideImage(u, v)): continue
                if (visited[u][v]): continue
                if (not img[u][v]): continue
                visited[u][v] = 1
                Q.put((u, v))
        return top, left, bot + 1, right + 1

    """ ------ main detect function ----- """
    result = []
    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            if (not visited[i][j] and img[i][j]):
                top, left, bottom, right = BFS(i, j)
                tmp = img[top : bottom, left : right]
                ret = np.zeros( (bottom - top + 60, right - left + 80))
                ret[30 : 30 + bottom - top, 40 : 40 + right - left] = tmp
                result.append(ret)
                # print(i, j, top, left, bottom, right)

    return result

""" ----------- Recognition digit in an image -----------"""
def digit_recognition(img):
    result = []
    list_of_digit_image = digit_detected(img)
    # print("There are %d digit in this image" %len(list_of_digit_image))
    for i in range(len(list_of_digit_image)):

        current_image = Image.fromarray(list_of_digit_image[i])
        # current_image.show();
        current_image = current_image.resize( (28, 28), Image.ANTIALIAS)
        #current_image.show();
        current_pix = abs(np.array(current_image))
        # for x in range(28):
        #     for y in range(28):
        #         print("%4d" % current_pix[x][y], end = '')
        #     print()
        result.append(prediction(current_pix)[0])
        # print(result[i])
    return result

def PerformNumber(img, mode = 0):
    digit = digit_recognition(img)
    if (mode == 0):
        print(digit)
    else:
        cur = 0
        for i in range(len(digit)):
            if (0 <= digit[i] <= 9):
                cur = cur * 10 + digit[i]
            else:
                print(cur, end = ' ')
                cur = 0
        print(cur)
# GUI
class Application(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.title("Handwritten Digit Recognition")
        # drawing
        self.canvas = tk.Canvas(self, width = 304, height = 304, bg = "white", cursor = "cross")
        self.label = tk.Label(self, text = "Drawing...", font = ("Helvetica", 40))
        self.classify_btn = tk.Button(self,  text = "Recognise", font = ("Helvetica 15 bold"), command = self.classify_handwritting)
        self.button_clear = tk.Button(self, text = "Clear", font = ("Helvetica 15 bold"), command = self.clear_all)
        # label predict aree

        # button area
        self.canvas.grid(row = 0, column = 0, pady = 2, sticky = W, )
        self.label.grid(row = 0, column = 1, pady = 2, padx = 2)
        self.classify_btn.grid(row = 1, column = 1, pady = 2, padx = 2)
        self.button_clear.grid(row = 1, column = 0, pady = 2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwritting(self):
        self.label.configure(text = 'Recognizing...')
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)
        img = ImageGrab.grab(rect)
        img = img.convert('L')
        # img.show() # this image take border
        img = 255 - np.array(img)
        img = img[2 : 302, 2 : 302]
        # digit, accuracy = prediction(img)
        digit_list = digit_recognition(img)
        result = "There are " + str(len(digit_list)) + " digits:\n";
        for x in digit_list: result += str(x) + ' '
        self.label.configure(text = result)

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill = "black")

# print(prediction(img))
# digit_detected(img)
# print(digit_recognition(img))
# PerformNumber(img, 1)

app = Application()
mainloop()