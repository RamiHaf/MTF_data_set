# MTF_data_sets
We present two image datasets:
### The curated Multi-Task Faces (curated MTF) image data set:
Is a meticulously curated collection of face images designed for various classification tasks, including face recognition, as well as race, gender, and age
classification. This dataset was automatically and manually annotated to train robust classification models. 
### The Non-curated Multi-Task Faces (Non-curated MTF) image data set:
Is a large collection of human faces that have been copied and annotated automatically. The main objective of the Non-curated dataset is to aid in training generative adversarial networks (GANs)

### Both the curated and the Non-curated MTF data sets have been ethically gathered by leveraging publicly available images of celebrities and strictly adhering to copyright regulations.
## Data distribution
### The curated MTF data set:
Contains 5,246 images with a distinct distribution of 240 celebrity image faces across different labels.  While our initial efforts were aimed at creating a balanced dataset across tasks and labels, the actual distribution of data available online led to an imbalance within the dataset. This imbalance can be attributed to various factors. I) Celebrities from different regions of the world publish their images at varying rates and under different copyright licenses. II) Young celebrities tend to publish their images more frequently than elderly celebrities. III) Elderly celebrities often have more images from their younger days than images from their current age.

### The Non-curated MTF data set:
Contains 132,816 images of 640 celebrity faces. The automatic annotation and cropping kept the distribution balanced accross all the diferent tasks, as intended.






## Get the data

The curated MTF data set can be accessed through the following: https://sobigdata.d4science.org/catalogue-sobigdata?path=/dataset/multi-task_faces_mtf_dataset

The non-curated MTF data set can be accessed through the following link: https://drive.google.com/drive/folders/1u-5COZPG1n28SiIBbbNjMjGTPyV-UlOJ?usp=sharing

Moreover, we have also made available all the trained models that we have evaluated on the curated and non-curated MTF datasets. Since these models provide baseline results for the various tasks supported by the dataset, our goal in releasing them is to facilitate future research on this dataset.
https://drive.google.com/drive/folders/1M0vKn2AeKNj-Ta0M6VHNHHznJYbLobtU?usp=sharing

The second version of the curated MTF data set for single tasks and the trained models are available on Google Drive through the following link: 
https://drive.google.com/drive/folders/1FCSCaBMkGZ6GFcOHmfbFGPcgucRaeCrf?usp=sharing.

Researchers can conveniently access the data for each task from this version, which streamlines their workflow and simplifies experimentation.

In addition to the data set release, we have made available the Python code we used for evaluating the data with the various DL models. This code can serve as a baseline for other researchers to compare their work with the results reported here. The code is available on Github through the following link: 
https://github.com/RamiHaf/MTF_data_set/blob/main/MTF_classification_five_models.ipynb
