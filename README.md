# ADNI_thesis_project
A 3D CNN for the 3-class ADNI problem (AD,MCI,CN) processing brain mri images

## Summary
   I have been given access to ADNI dataset in order to accomplish my thesis project. It was a great honor for me as i was always interested in making projects in favor of healthcare. I was running the experiment through the spock server that was provided by my university (ECE-NTUA) <br />
   The first file created was group.py. In this file i seperated the different MRI images, that were included in ADNI file, into 3 subfolders. The 3 subfolders were the 3 classes of our problem, that are CN, MCI, AD. For this purpose we used the .csv file, that is contained in our repo and includes the Image_data_ID and the group it belongs. <br />
   The second file created was preprocessing.py. In this file, as the name betrays, we have some basic preprocessing in the 3D images, in order to have the same resolution and shape of the images. Furthermore, we use some data augmentation techniques for the purpose of having pluralism in our dataset. <br />
   The third file is model.py. In this file, we run our experiments for a specific dataset and its split into train,test and validation datasets. Now we can evaluate our models. In addition we can detect early stopping and find new ways to delay its appearance to a subsequent epoch.
