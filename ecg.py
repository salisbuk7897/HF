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
# from tkinter import *
# import pandas as pd
# from tkinter import filedialog
import json
import cv2

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
# Images = df["Directory"].values
# random_images = [np.random.choice(Images) for i in range(9)]
# fig_random = plt.figure(figsize=(20,10))
# for i in range(9):
#     plt.subplot(3,3,i+1)
#     img = plt.imread(os.path.join(random_images[i]))
#     plt.imshow(img, cmap='gray')
#     plt.axis('off')
#     plt.tight_layout()
# st.write("Displaying Random Sample Images from the dataset")
# st.pyplot(fig_random)

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

# Saved Model
savedModel=load_model('ecgResnet2.h5')
savedModel.summary(print_fn=lambda x: st.text(x))

#Saved History
savedHistory=np.load('resnet_history.npy',allow_pickle='TRUE').item()

# Accuracy
st.write("")
st.text("Accuracy Plot")
fig_acc = plt.gcf()
plt.plot(savedHistory['accuracy'])
plt.plot(savedHistory['val_accuracy'])
plt.axis(ymin=0.4, ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
st.pyplot(fig_acc)

# Loss
st.write("")
st.text("Loss Plot")
fig_loss = plt.figure(figsize=(20,10))
plt.plot(savedHistory['loss'])
plt.plot(savedHistory['val_loss'])
plt.axis(ymin=0.4, ymax=1)
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
st.pyplot(fig_loss)

# AUC
st.write("")
st.text("AUC Plot")
fig_auc = plt.figure(figsize=(20,10))
plt.plot(savedHistory['AUC'])
plt.plot(savedHistory['val_AUC'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
st.pyplot(fig_auc)

# use save models to make prediction on test data
#model_pred=savedModel.predict(test_generator)
# Use saved model prediction on test data
modelPred = np.load('savedModelTestPredictions.npy', allow_pickle=True).item()
model_pred = modelPred['pred']

test_labels=[]
for i in range(294):
    img, label = test_generator.__getitem__(i)
    test_labels.append(list(label[0]))

pred_list=[]
for z in range(len(model_pred)):
    pred_list.append(list(model_pred[z]))

img_pred_list_z = []
for i in range(294):
    img_dictz = {}
    pred_listz = pred_list[i]
    # start here
    max_pred =  max(pred_listz)
    pred_index = pred_listz.index(max_pred)
    label_listzz = test_labels[i]
    max_label = max(label_listzz)
    label_index = label_listzz.index(max_label)
    img_dictz[f'ECG Test {i}'] = (int(round(max_pred)) == int(max_label) and pred_index == label_index) #correct this arrangement to ECG Test: number
    img_dictz['status'] = f"{"Correct" if (int(round(max_pred)) == int(max_label) and pred_index == label_index) else "Incorrect"}"
    img_dictz['pred'] = f"class {pred_index}"
    img_dictz['class'] = f"class {label_index}"
    img_pred_list_z.append(img_dictz)

class_dctz= [] 
for i in img_pred_list_z:
    class_dctz.append(i['class'])

class_dfz = pd.DataFrame(class_dctz, columns=['classes'])
#class_dfz.head()

class_df_resultz = pd.DataFrame(class_dfz['classes'].value_counts())
class_df_resultz = class_df_resultz.reset_index()
class_df_resultz.columns = ["Classes","Total"]
#class_df_resultz.head()

test_df_resultz = pd.DataFrame(df_test['Diagnosis'].value_counts())
test_df_resultz = test_df_resultz.reset_index()
test_df_resultz.columns = ["Diagnosis","Total"]
#test_df_resultz.head()

merged_dfz = test_df_resultz.merge(class_df_resultz, how='right')

img_pred_listz2 = []
full_namez = {'af': 'Atrial Flutter', 'afib':'Atrial Fibrillation', 'asmi':'Anteroseptal Myocardial Infarction', 'imi':'Inferior myocardial Infarction', 'sb':'Sinus Bradycardia', 'nsr':'Normal Sinus Rythm'}
for i in img_pred_list_z:
    Diagnosisz = f"{merged_dfz[merged_dfz['Classes']==i['class']]['Diagnosis'].item()}"
    i['real_class'] = Diagnosisz
    i['real_classname'] = full_namez[Diagnosisz]
    Diagnosisz2 = f"{merged_dfz[merged_dfz['Classes']==i['pred']]['Diagnosis'].item()}"
    i['pred_class'] = Diagnosisz2
    i['pred_classname'] = full_namez[Diagnosisz2]
    img_pred_listz2.append(i)

rows = [] 
for a in img_pred_listz2:
    rows.append(list(a.values()))
columns = list(img_pred_listz2[0].keys())
#columns

tst_result = pd.DataFrame(rows, columns= columns)
tst_result = tst_result.drop(columns[0], axis=1)

afib_res = tst_result[tst_result['real_class']=='afib']
af_res = tst_result[tst_result['real_class']=='af']
asmi_res = tst_result[tst_result['real_class']=='asmi']
imi_res = tst_result[tst_result['real_class']=='imi']
nsr_res = tst_result[tst_result['real_class']=='nsr']
sb_res = tst_result[tst_result['real_class']=='sb']

x = test_df_resultz['Diagnosis'].to_numpy()
y = test_df_resultz['Total'].to_numpy()

colours = ["Blue","green","Red","Yellow", "Purple", "cyan"]

data_fig = go.Figure(data=[go.Bar(x=y,
                                y=x, 
                                text=y,
                                textposition="outside",
                                marker_color=colours,
                                 orientation='h')])

data_fig.update_layout(width=1000, height=500)
data_fig.update_layout(title_text="Test Data")
st.write(data_fig)


df_res_afib = pd.DataFrame(afib_res['status'].value_counts())
df_res_afib = df_res_afib.reset_index()
df_res_afib.columns = ["Status","Total"]

# af
df_res_af = pd.DataFrame(af_res['status'].value_counts())
df_res_af = df_res_af.reset_index()
df_res_af.columns = ["Status","Total"]

#asmi
df_res_asmi = pd.DataFrame(asmi_res['status'].value_counts())
df_res_asmi = df_res_asmi.reset_index()
df_res_asmi.columns = ["Status","Total"]

#imi
df_res_imi = pd.DataFrame(imi_res['status'].value_counts())
df_res_imi = df_res_imi.reset_index()
df_res_imi.columns = ["Status","Total"]

# nsr
df_res_nsr = pd.DataFrame(nsr_res['status'].value_counts())
df_res_nsr = df_res_nsr.reset_index()
df_res_nsr.columns = ["Status","Total"]

# sb
df_res_sb = pd.DataFrame(sb_res['status'].value_counts())
df_res_sb = df_res_sb.reset_index()
df_res_sb.columns = ["Status","Total"]

fig = make_subplots(
    rows=2, cols=3, 
    specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}], [{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]], 
    subplot_titles=("Atrial Fibrillation Test", "Atrial Flutter Test", "Anteroseptal Myocardial Infarction","Inferior Myocardial Infarction", "Normal Sinus Rythm", "Sinus Bradycardia"))
fig.add_trace(go.Pie(labels=df_res_afib['Status'].to_numpy(), values=df_res_afib['Total'].to_numpy()), 1, 1)
fig.add_trace(go.Pie(labels=df_res_af['Status'].to_numpy(), values=df_res_af['Total'].to_numpy()), 1, 2)
fig.add_trace(go.Pie(labels=df_res_asmi['Status'].to_numpy(), values=df_res_asmi['Total'].to_numpy()), 1, 3)
fig.add_trace(go.Pie(labels=df_res_imi['Status'].to_numpy(), values=df_res_imi['Total'].to_numpy()), 2, 1)
fig.add_trace(go.Pie(labels=df_res_nsr['Status'].to_numpy(), values=df_res_nsr['Total'].to_numpy()), 2, 2)
fig.add_trace(go.Pie(labels=df_res_sb['Status'].to_numpy(), values=df_res_sb['Total'].to_numpy()), 2, 3)


fig.update_layout(width=1000, height=700, title_text="Prediction Results for All 6 Classes")
fig.update_traces(marker=dict(colors=['purple', 'aquamarine']))
st.write(fig)

uploaded_imgDir = ''
uploaded_imgDiagnosis = ''
st.subheader("Model Prediction Testing")
st.text('Please Upload an ECG Image to Predict')
## file upload for prediction
img_byte = ''
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    img_byte = bytes_data
    #st.write(bytes_data)
st.text("please select diagnosis")
Diagnosis_option = st.selectbox(
    "What is the diagnosis of the image you uploaded?",
    ('Atrial Flutter', 'Atrial Fibrillation', 'Anteroseptal Myocardial Infarction', 'Inferior myocardial Infarction', 'Sinus Bradycardia', 'Normal Sinus Rythm'),
)

#st.write("You selected:", Diagnosis_option)
img_path2 = "predict"
def prepare_pred(a, b):
    # not expecting many people to test, hence the next few lines of code have not been throughly engineered
    # Opening JSON file
    f = open(os.path.join(os.getcwd(),'data.json'))

    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    datacopy = data.copy()
    if (len(datacopy["data"]) == 0):
        img_new = os.path.join(os.getcwd(),f"{img_path2}", f"{1}.jpg")
        with open(img_new, 'wb') as file:
            file.write(a)
            dict_data = [img_new, b]
            datacopy["data"].append(dict_data)
            # Closing file
            f.close()
            with open(os.path.join(os.getcwd(),"data.json"), 'w') as file1:
                json.dump(datacopy, file1)
        return dict_data
    else:
        img_new = os.path.join(os.getcwd(),f"{img_path2}", f"{len(datacopy["data"])+1}.jpg")
        with open(img_new, 'wb') as file:
            file.write(a)
            dict_data = [img_new, b]
            datacopy["data"].append(dict_data)
            # Closing file
            f.close()
            with open(os.path.join(os.getcwd(),"data.json"), 'w') as file1:
                json.dump(datacopy, file1)
        return dict_data

def start_pred(imagedata):
    image_record = [imagedata]
    dfp =pd.DataFrame(image_record, columns=["img","p"])

    transformed_img = Image_generator.flow_from_dataframe(
        dataframe=dfp,
        directory= None,
        x_col="img",
        y_col="p",
        class_mode = "categorical",
        batch_size=1,
        seed=42,
        shuffle=False,
        target_size=(512,920))

    img, label = transformed_img.__getitem__(0) #cv2.imread(infiles[0])
    image_resized= cv2.resize(img[0], (920,512))
    #cv2.imshow('ECG', img[0])
    #cv2.waitKey(0)
    img=np.expand_dims(image_resized,axis=0)
    #img.shape

    pred_img=savedModel.predict(img)
    #print(pred_img[0])
    img_rec = {}
    pred_list_img = list(pred_img[0])
    max_pred_img =  max(pred_list_img)
    pred_index_img = pred_list_img.index(max_pred_img)
    img_rec['pred'] = f"class {pred_index_img}"
    img_Diagnosis = f"{merged_dfz[merged_dfz['Classes']==img_rec['pred']]['Diagnosis'].item()}"
    img_rec['pred_class'] = img_Diagnosis
    img_rec['pred_classname'] = full_namez[img_Diagnosis]
    img_rec['score'] = str(max_pred_img)
    img_rec['img'] = imagedata[0]
    img_rec['real_Diagnosis'] = imagedata[1]
    statuss = f'{"Correct" if img_rec["pred_classname"] == img_rec["real_Diagnosis"] else "Incorrect"}'
    img_rec['status'] = statuss

    f = open(os.path.join(os.getcwd(),'result.json'))
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    datacopy = data.copy()
    f.close()
    datacopy["result"].append(img_rec)
    with open(os.path.join(os.getcwd(),'result.json'), 'w') as file1:
                json.dump(datacopy, file1)
    #print(img_rec)
    st.text("Prediction Results")
    
    st.dataframe(
        pd.DataFrame({
            "field": ["Predicted Diagnosis", "Provided Diagnosis", "Prediction Status"],
            "values": [img_rec["pred_classname"], img_rec["real_Diagnosis"], statuss],
        }), width=800
    )
    # st.write("Predicted Diagnosis: ",img_rec["pred_classname"])
    # st.write("Provided Diagnosis: ",img_rec["real_Diagnosis"])
    #st.write("Probability score of Diagnosis: ",img_rec["score"])
    float_score = float("{:.1f}".format(float(img_rec["score"])*100))
    #st.write("Prediction Status: ",img_rec["pred_classname"] == img_rec["real_Diagnosis"])
    rct = f'{  "😄" if img_rec["pred_classname"] == img_rec["real_Diagnosis"] else '😓' }'
    st.markdown("""
        <style>
            table {
                width: 100%;
            }
            td {
                text-align: center;
            }
            .big-font {
                font-size:100px !important;
            }
        </style>
        """, unsafe_allow_html=True)
    # st.markdown(f'<div class="container"><p class="big-font">{rct}</p><div>', unsafe_allow_html=True)
    # st.markdown(f'<div class="container">\
    #                 <div>\
    #                     <div class="big-font">{float_score}%</div>\
    #                     <div class="ui-labels">Pobability Score</div>\
    #                 </div>\
    #             </div>', unsafe_allow_html=True)
    st.markdown(f'<table>\
                    <tr>\
                        <th> Probability Score </th>\
                        <th> Model Status </th>\
                    </tr>\
                    <tr>\
                        <td class="big-font">\
                            {float_score}%\
                        </td>\
                        <td class="big-font">\
                            {rct}\
                        </td>\
                    </tr>\
                </table>', unsafe_allow_html=True)

# if showPredict:
#     if st.button("Start prediction"):
#         start_pred()
# else:
if st.button("Predict"):
    cx = prepare_pred(img_byte, Diagnosis_option)
    start_pred(cx)
