import tensorflow as tf
import numpy as np
from numpy import zeros, float32
import os
from PIL import Image, ImageFilter, ImageTk
import tkinter as tk
from tkinter import filedialog, Canvas
import tkinter.ttk as ttk
import time
import sys


def display_mnist(image_raw):
    image_2D = (np.reshape(image_raw, (28, 28))).astype(np.uint8)
    image_out = Image.fromarray(image_2D, 'L')
    return image_out
	
def parse_image(argv, size):
	im = Image.open(argv).convert('L')
	width = 28
	height = 28
	newImage = im.resize((width, height), Image.ANTIALIAS)	
	tv = list(newImage.getdata())  # get pixel values
	tva = zeros((size* size), dtype=float32)
	for px in range(size * size):
		if tv[px] >= 150:
			tva[px] = 0
		else:
			tva[px] = 255 		
	return tva

def inference():	
	global image_address
	global image_display
	sess_start_t = int(round(time.time() * 1000))
	image_imported = [parse_image(image_address, 28)]
	image_display = display_mnist(image_imported)
	
	# Set up directory of model
	model_directory = os.path.join(os.getcwd(), 'model')
	output_converted_graph_name = os.path.join(model_directory, 'model_converted.tflite')
	
	# Load TFLite model and allocate tensors.
	interpreter = tf.contrib.lite.Interpreter(model_path=output_converted_graph_name)
	interpreter.allocate_tensors()

	# Get input and output tensors.
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	# Test model with testing dataset
	input_shape = input_details[0]['shape']
	input_data = image_imported
	interpreter.set_tensor(input_details[0]['index'], input_data)
	interpreter.invoke()
	output_data = interpreter.get_tensor(output_details[0]['index'])
	
	# Output prediction with highest probability
	max_pred_index = np.argmax(output_data[0])
	sess_end_t = int(round(time.time() * 1000))
	print("> Predicted value is", max_pred_index)
	print("> Probability:", output_data[0][max_pred_index])
	GUI.show_inference()
	GUI.popup(max_pred_index, output_data[0][max_pred_index], int(sess_end_t-sess_start_t))
class GUI:
	def __init__(self, master):
		global ImgPanel
		
		master.wm_title('T2-CNN-MNIST')
		master.configure(bg = '#f2f7ff')
		master.iconbitmap(master,default='icon.ico')
		width = 500 
		height = 500 
		width_screen = master.winfo_screenwidth() # width of the screen
		height_screen = master.winfo_screenheight() # height of the screen
		x = (width_screen/2) - (width/2)
		y = (height_screen/2) - (height/2)
		master.geometry('%dx%d+%d+%d' % (width, height, x, y))
	    #Icons made by Darius Dan from https://www.flaticon.com/ Flaticon is licensed Creative Commons BY 3.0
		master.resizable(0,0)
		CtrlPanel = tk.Frame(master=master, width=500, height=100)
		CtrlPanel.place(x=250, y=460, anchor="center")
		ImgPanel = tk.Frame(master=master, width=400, height=400, bg="#e6f3ff")
		ImgPanel.place(x=250, y=220, anchor="center")
		ImgPanel.pack_propagate(False) 
		ButtonRun = ttk.Button(master=CtrlPanel, text='Run',
							 command=lambda:inference()).pack(side = "right")
		ButtonOpen = ttk.Button(master=CtrlPanel, text='Open',
							 command=lambda:GUI.open_file(master)).pack(side = "left")
		ImgErr = tk.Label(ImgPanel, text="( no image available )", bg="#e6f3ff")
		ImgErr.place(x=200, y=200, anchor="center")	

	def open_file(self):
		global image_address
		global ImgPanel
		global dummy
		image_address =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("all files","*.*"),("jpeg files","*.jpg"),("png files","*.png")))
		img = ImageTk.PhotoImage(Image.open(image_address).resize((400, 400), Image.ANTIALIAS))
		ImgLabel = tk.Label(ImgPanel, image = img)
		ImgLabel.image = img
		ImgLabel.place(x=200, y=200, anchor="center")
	
	def show_inference():
		global ImgPanel
		global image_display
		img = ImageTk.PhotoImage(image_display)
		ImgInf = tk.Label(ImgPanel, image = img)
		ImgInf.image = img
		ImgInf.place(x=386, y=386, anchor="center")
	def popup(prediction, probability, time):
		popup = tk.Tk()
		popup.wm_title("Output")
		popup.configure(bg = '#f2f7ff')
		popup.iconbitmap(popup,default='icon.ico')
		width = 300 
		height = 120 
		width_screen = popup.winfo_screenwidth() # width of the screen
		height_screen = popup.winfo_screenheight() # height of the screen
		x = (width_screen/2) - (width/2)
		y = (height_screen/2) - (height/2)
		popup.geometry('%dx%d+%d+%d' % (width, height, x, y))
		popup.resizable(0,0)
		text1 = str("Prediction is {}".format(prediction))
		text2 = str("Probability is {}".format(probability))
		text3 = str("Time needed: {} ms".format(time))
		label1 = tk.Label(popup, text=text1, bg="#f2f7ff")
		label1.pack(side="top", fill="x", pady=2)
		label2 = tk.Label(popup, text=text2, bg="#f2f7ff")
		label2.pack(side="top", fill="x", pady=2)
		label3 = tk.Label(popup, text=text3, bg="#f2f7ff")
		label3.pack(side="top", fill="x", pady=2)
		B1 = ttk.Button(popup, text="Thanks", command = popup.destroy)
		B1.place(x=150, y=100, anchor = "center")
		popup.mainloop()
		
if __name__ == '__main__':
	dummy = 0
	root = tk.Tk()
	app = GUI(root)
	root.mainloop()

	
	
	
