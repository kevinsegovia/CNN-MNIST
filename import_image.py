from numpy import zeros, float32
from PIL import Image, ImageFilter

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