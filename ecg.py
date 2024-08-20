# Import the required packages
import numpy as np
import os
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from functools import reduce
from tensorflow.keras.applications import DenseNet121, ResNet50
from tensorflow.keras import Model
from keras.layers import Lambda, Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
import json
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import streamlit as st

st.header("Early Detection of Heart Failure Using ECG")
st.write("  Heart failure is a condition arising from functional or structural abnormality of the Heart.\
        There is a wide range of abnormalities that can lead to heart failure. ECG is a widely used diagnostic tool\
        in healthcare to diagnose heart abnormalities. However, ECG interpretation skills defer among healthcare providers.\
        An AI tool for ECG interpretation will be of great help in augmenting the skills of healthcare providers when diagnosing\
        heart diseases. In this project, we'll attempt \
        to build a model that can assist with early identification of cardiac abnormalities, leading to improvement \
        in timely initiation of treatment, which in turn slows the progression to heart failure. We'll also restrict ourselves\
        to 6 classes( 5 abnormalities and 1 normal class) which include AF (Atrial Flutter), Afib (Atrial Fibrillation), asmi (anteroseptal myocardial infarction)\
        imi (inferior myocardial infarction), sb (Sinus Bradycardia) and nsr (Normal sinus Rythm)\
        ")
st.subheader("Exploratory Analysis and Image Pre-processing")
# load data and create dataframe
folders = ["afib", "af", "sb", "imi", "asmi", "nsr", ]
labels =folders.copy()
image_path = "./ECGSegmentation/data/"
data = []

for folder in folders:
    for filename in os.listdir(os.path.join(image_path,folder)):
        data.append(['{}{}/{}'.format(image_path,folder,filename), folder]) #ECG path and diagnosis

df = pd.DataFrame(data) #all the images of interest
df.columns =['Directory', 'Diagnosis']
st.write("The dataset contains", df['Directory'].count(), " ECG Images")
st.dataframe(df, width=800)

## Divide data into training, validation and test datasets

# AF
df_af = df.loc[df['Diagnosis'] == 'af']
n_af = round(0.70*df_af['Diagnosis'].count().astype(int))
n_af2 = round(0.80*df_af['Diagnosis'].count().astype(int))
df_af_test = df_af.iloc[n_af2:] # 20% data
df_af_validation = df_af.iloc[n_af: n_af2] # 10% data
df_af_train = df_af.iloc[:n_af] #70% data

#AFIB
df_afib = df.loc[df['Diagnosis'] == 'afib']
n_afib = round(0.70*df_afib['Diagnosis'].count().astype(int))
n_afib2 = round(0.80*df_afib['Diagnosis'].count().astype(int))
df_afib_test = df_afib.iloc[n_afib2:]
df_afib_validation = df_afib.iloc[n_afib:n_afib2]
df_afib_train = df_afib.iloc[:n_afib]

#NSR
df_nsr = df.loc[df['Diagnosis'] == 'nsr']
n_nsr = round(0.70*df_nsr['Diagnosis'].count().astype(int))
n_nsr2 = round(0.80*df_nsr['Diagnosis'].count().astype(int))
df_nsr_test = df_nsr.iloc[n_nsr2:]
df_nsr_validation = df_nsr.iloc[n_nsr:n_nsr2]
df_nsr_train = df_nsr.iloc[:n_nsr]

#IMI
df_imi = df.loc[df['Diagnosis'] == 'imi']
n_imi = round(0.70*df_imi['Diagnosis'].count().astype(int))
n_imi2 = round(0.80*df_imi['Diagnosis'].count().astype(int))
df_imi_test = df_imi.iloc[n_imi2:]
df_imi_validation = df_imi.iloc[n_imi:n_imi2]
df_imi_train = df_imi.iloc[:n_imi]

#ASMI
df_asmi = df.loc[df['Diagnosis'] == 'asmi']
n_asmi = round(0.70*df_asmi['Diagnosis'].count().astype(int))
n_asmi2 = round(0.80*df_asmi['Diagnosis'].count().astype(int))
df_asmi_test = df_asmi.iloc[n_asmi2:]
df_asmi_validation = df_asmi.iloc[n_asmi:n_asmi2]
df_asmi_train = df_asmi.iloc[:n_asmi]

#SB
df_sb = df.loc[df['Diagnosis'] == 'sb']
n_sb = round(0.70*df_sb['Diagnosis'].count().astype(int))
n_sb2 = round(0.80*df_sb['Diagnosis'].count().astype(int))
df_sb_test = df_sb.iloc[n_sb2:]
df_sb_validation = df_sb.iloc[n_sb:n_sb2]
df_sb_train = df_sb.iloc[:n_sb]


train_data_frames = [df_afib_train, df_af_train, df_sb_train, df_imi_train, df_asmi_train, df_nsr_train]
validation_data_frames = [df_afib_validation, df_af_validation, df_sb_validation, df_imi_validation, df_asmi_validation, df_nsr_validation]
test_data_frames = [ df_afib_test, df_af_test, df_sb_test, df_imi_test, df_asmi_test, df_nsr_test]

df_train = pd.concat(train_data_frames, ignore_index=True, sort=False) #training data 70%
df_validation = pd.concat(validation_data_frames, ignore_index=True, sort=False) #10% of each class
df_test = pd.concat(test_data_frames, ignore_index=True, sort=False) #20% ofeachclass

# Analysis of classes
df_result = pd.DataFrame(df["Diagnosis"].value_counts())
df_result = df_result.reset_index()
df_result.columns = ["Diagnosis","Total"]
st.text("Total number of images for each class in the dataset")
st.dataframe(
    df_result,
    width=800
)

# Show Graphical Representation
fig = px.pie(df_result,
             values='Total',
             names='Diagnosis')
st.write("The graphical representation of the classes in the dataset",fig)

# Show random images
Images = df["Directory"].values
random_images = [np.random.choice(Images) for i in range(9)]
fig_random = plt.figure(figsize=(20,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    img = plt.imread(os.path.join(random_images[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
st.write("Displaying Random Sample Images from the dataset")
st.pyplot(fig_random)

# Raw image details
sample_img = df.Directory[0]
raw_image = plt.imread(os.path.join(sample_img))
rawImg_fig = plt.figure(figsize=(20,10))
plt.imshow(raw_image)
plt.colorbar()
plt.title('Raw ECG Image')
st.write("Details of A raw image from the dataset")
st.pyplot(rawImg_fig)
st.write(f"The dimensions of the image are {raw_image.shape[1]} pixels width and {raw_image.shape[0]} pixels height")
st.write(f"The maximum pixel value is {raw_image.max(): 4f} and the minimum is {raw_image.min(): 4f}")
st.write(f"The mean value of the pixels is {raw_image.mean(): 4f} and the standard deviation is {raw_image.std(): 4f}")

# Normalization 
Image_generator = ImageDataGenerator(
    samplewise_center = True,
    samplewise_std_normalization=True)

#Standardize the images
train_generator=Image_generator.flow_from_dataframe(
    dataframe=df_train,
    directory= None,
    x_col="Directory",
    y_col="Diagnosis",
    class_mode = "categorical",
    subset="training",
    batch_size=1,
    seed=42,
    shuffle=False,
    target_size=(512,920))
valid_generator=Image_generator.flow_from_dataframe(
    dataframe=df_validation,
    directory= None,
    x_col="Directory",
    y_col="Diagnosis",
    class_mode = "categorical",
    ubset="validation",
    batch_size=1,
    seed=42,
    shuffle=False,
    target_size=(512,920))
#test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=Image_generator.flow_from_dataframe(
    dataframe=df_test,
    directory= None,
    x_col="Directory",
    y_col="Diagnosis",
    class_mode = "categorical",
    #subset="validation",
    batch_size=1,
    seed=42,
    shuffle=False,
    target_size=(512,920))

#Visualize Standardized Image
sns.set_style("white")
generated_image, label = train_generator.__getitem__(0)
stdImg_fig = plt.figure(figsize=(20,10))
plt.imshow(generated_image[0], cmap='gray')
st.write("Sample Pre-processed Image")
st.pyplot(stdImg_fig)

# Comparison between raw and standardized image pixels
sns.set()
comp_fig = plt.figure(figsize=(20,10))
sns.distplot(raw_image.ravel(), label=f"original image mean {np.mean(raw_image): .4f} and standard deviation {np.std(raw_image): 4f}"
             f" min pixel value {np.mean(raw_image):.4f}, max pixel value {np.max(raw_image):.4f}",
            color="blue",
            kde=False)
sns.distplot(generated_image.ravel(), label=f"generated image mean {np.mean(generated_image): .4f} and standard deviation {np.std(generated_image): 4f}"
             f" min pixel value {np.mean(generated_image):.4f}, max pixel value {np.max(generated_image):.4f}",
            color="red",
            kde=False)
plt.legend(loc='upper center')
st.write("Comparison between the pixel distribution of raw and standardized images")
st.pyplot(comp_fig)

plt.title('Comparison of Distribution of pixels intensities between raw and generated images')
plt.xlabel('pixel intensity')
plt.ylabel('# pixels in image')

st.subheader("Training, Validation & Test")
st.text("Results of data division into training test and validation")
st.write("Total number of training images",df_train["Directory"].count())
st.write("Total number of validation images",df_validation["Directory"].count())
st.write("Total number of test images",df_test["Directory"].count())