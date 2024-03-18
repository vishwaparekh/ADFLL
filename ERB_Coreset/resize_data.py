import numpy as np
import pandas as pd
import shutil
import glob
import os
import cv2
import SimpleITK as sitk
from pathlib import Path
import random

def downsamplePatient(patient_CT, resize_factor):
	original_CT = sitk.ReadImage(patient_CT,sitk.sitkInt32)
	dimension = original_CT.GetDimension()
	reference_physical_size = np.zeros(original_CT.GetDimension())
	reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(original_CT.GetSize(), original_CT.GetSpacing(), reference_physical_size)]
	
	reference_origin = original_CT.GetOrigin()
	reference_direction = original_CT.GetDirection()
	reference_size = [round(sz/resize_factor) for sz in original_CT.GetSize()] 
	reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]
	
	reference_image = sitk.Image(reference_size, original_CT.GetPixelIDValue())
	reference_image.SetOrigin(reference_origin)
	reference_image.SetSpacing(reference_spacing)
	reference_image.SetDirection(reference_direction)
	
	reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/3.0))
	transform = sitk.AffineTransform(dimension)
	transform.SetMatrix(original_CT.GetDirection())
	
	transform.SetTranslation(np.array(original_CT.GetOrigin()) - reference_origin)
	centering_transform = sitk.TranslationTransform(dimension)
	img_center = np.array(original_CT.TransformContinuousIndexToPhysicalPoint(np.array(original_CT.GetSize())/3.0))
	centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
	centered_transform = sitk.Transform(transform)
	centered_transform.AddTransform(centering_transform)
	
	return sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0)
    
    

from glob import glob
train=glob("train_sets/*/", recursive = True)
test=glob("test_sets/*/", recursive = True)
all_dir= train+test
for di in all_dir:
	path=Path('small3x/'+di)
	path.mkdir(parents=True,exist_ok=True)
	images=glob(di+"*.gz")
	for image in images:
		img=downsamplePatient(image, 3)
		sitk.WriteImage(img, 'small3x/'+image)
	labels=glob(di+"*.txt")
	for label in labels:
		if 'path' in label:
			shutil.copyfile(label, 'small3x/'+di+'/'+label.split('/')[-1])
			continue
		a=np.loadtxt(label,delimiter=',')
		b=(a/3).astype(int)
		np.savetxt('small3x/'+label,b,fmt='%i', delimiter=",")

	
