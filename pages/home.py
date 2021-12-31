import streamlit as st
from PIL import Image



def home():

    st.title('EasyML')

    image = Image.open('pages/easymlpic.png')

    st.image(image)

    st.markdown('''

    ## This is the **homepage** of the EasyML App :rocket:

    Machine learning is the study of computer algorithms that can improve automatically through experience and by the use of data.
    With EasyML I aim to give an introduction to AutoML systems and bto show that machine learning is now easier than ever!

    Try out the **IdleEDA**, **AutoML**, and **LazyCompare** features!

    ## Usage

    Your data needs to be in a certain format for the exploratory data analysis and autoML features to work correctly. The minimum requirement for the EDA is that your data file is a CSV. For autoML the requirement is a CSV and your last value should be the target value while the rest of the values are data values.
    The AutoML feature is a regressor, so input the data appropriately.

    ***IdleEDA***

    For the exploratory data analysis part of the app, most CSVs will work. Just input the CSV and see the results!
    *Note that larger datasets can take upwards of 3 minutes

    ***AutoML***

    Datasets in CSV format with numerical values are perfect, just make sure to have the target value in the last column. Based on the model you choose
    you will be able to change hyperparameters which will change the performance of the model. Testing different model hyperparameters can change the performance of the model greatly.

    ***LazyCompare***

    Similar to AutoML datasets will need to be in CSV format with numerical values with the target value in the last column. 20+ Machine learning models will be tested on the data given
    and the user will be able to see the results.

    Built by [Nalin Nagar](https://github.com/NNDEV1/)

    ''')
