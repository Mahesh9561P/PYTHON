from re import findall
import os
import streamlit as st
import pandas as pd 
import numpy as np
import ogdiabret
from PIL import Image 
st.title("Diabetic Retinopathy Detection")
st.subheader("WORLD LOOKS BEAUTIFULL WITH HEALTHY EYES.")

def load_image(image_file):
	img = Image.open(image_file)
	return img

def main():
	st.title("Upload fundus image to be tested")
	ogdiabret.Image_Processing()
	#st.subheader("Home")
	image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
	
	if image_file is not None:	
		file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
		#st.write(file_details)
		
		img = load_image(image_file)
		test_str = image_file.name

		file_no=findall(r'\d+',test_str)
		file_no=int(file_no[0])
		ogdiabret.Preprocessing()
		
		test_image=ogdiabret.scaler_means[file_no]
		#ogdiabret.scaler_means=np.array(ogdiabret.scaler_means)
		ogdiabret.scaler_means.pop(file_no)
		ogdiabret.scaler_means=np.array(ogdiabret.scaler_means)
		ogdiabret.scaler_means=ogdiabret.scaler_means.reshape(-1,1)
		test_result=ogdiabret.y_result[file_no]
		test_image=test_image.reshape(-1,1)
		test_result=np.array(test_result)

		ogdiabret.y_result.pop(file_no)
		ogdiabret.y_result=np.array(ogdiabret.y_result)
		ogdiabret.Model_KNN(test_image,test_result)

if __name__ == '__main__':
	main()

