import os
import streamlit as st
import pandas as pd
import warnings 
import tensorflow as tf
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
warnings.filterwarnings('ignore')


## new stuff
from textblob import TextBlob
##

from tensorflow.keras.models import load_model
#model = tf.keras.models.load_model(r'C:\Users\User\Desktop\SA\LSTM-Sentiment')
model = tf.keras.models.load_model(r'bestModel\bestmodel.h5')

#tokenizing

tokenizer = Tokenizer()
with open(r'C:\Users\User\Desktop\SA\tokenizer.pkl', 'rb') as tokenizer_file:
    loaded_tokenizer = pickle.load(tokenizer_file)


def preprocess_text2(text):
    sequence = loaded_tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=60)
    return padded_sequence

def predict_sentiment(text):
    preprocessed_text = preprocess_text2(text)
    # Ensure the input has three dimensions (batch_size, sequence_length, features)
    preprocessed_text = preprocessed_text.reshape(1, preprocessed_text.shape[1], 1)
    # Predict sentiment
    prediction = model.predict(preprocessed_text)
    return prediction

#def analyze(x):
    if x>= 0.3:
        return 'Positive'
    elif x <= -0.3:
        return 'Negative'
    else:
        return 'Neutral'


## Streamlit starts here ##
st.title('Senty!')

# Insert containers separated into tabs:
tab1, tab2 = st.tabs(["Welcome", "Instructions"])

with tab1:
    tab1.write("Welcome to my Sentiment Analysis webapp powered by Streamlit! Please read the instructions.")


with tab2:
    st.markdown(
        """
        <div style="text-align: center; padding: 10px;">
            <h3> How To Use </h3>
        </div>
        <div style="text-align:left; padding:0px;">
        <li> Prepare your CSV file. Ensure the first column contains your desired text. <span><a href= https://support.microsoft.com/en-us/office/rearrange-the-order-of-columns-in-a-table-d1701654-fe43-4ae3-adbc-29ee03a97054 target='_blank'> How? </a></span></li>
        <li> Upload your CSV file using the Analyze CSV dropdown </li>
        <li> After a few seconds, your processed dataset should be ready for download</li>
        <li> Processing time depends on the size of your dataset </li>
        </div>
        """,
        unsafe_allow_html=True
    )


with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload CSV file')    

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity
    def analyze(text):
        if text >= 0.5:
            return 'Positive'
        elif text <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

with st.spinner():

    if upl:
        df = pd.read_csv(upl)
        #df['Processed_Text'] = df.iloc[:,0].apply(preprocess_apply)
        #input_data = np.array([processed_text]) 
        #input_data = np.array(df['Processed_Text']) #tuple index out of range >>#this is probably the wrong error
        
        #sentiment = model.predict(input_data)
        #prediction = predict_sentiment(input_data) 
        #input 0 of LSTM layer is incompatible with the layer, expected ndim3 but found ndim2 

        #df['Analysis'] = df['Reviews'].apply(preprocess_apply) #cant rememeber why I did this

        df['Sentiment'] = np.nan
             
       
        df['Processed'] = df.iloc[:,0]
        df['Value'] = df['Processed'].apply(predict_sentiment)
        
        df['Value'] = df['Value'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
        df['Sentiment'] = df['Value'].apply(analyze)
        #.apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)

        #df['Processed'] = df['Processed'].apply(lambda x: [round(x[0], 2)] if isinstance(x, np.ndarray) else x)
            
        #df['Processed Text'] = df.iloc[:,0].apply(predict_sentiment)

        
        #st.write(input.shape)
        #input = input.reshape(6,100) #tried to reshape the array
        #sentiment = model.predict(input) 


        ### TextBlob ###

        #df['Processed_Text'] = df.iloc[:,0].apply(preprocess_apply)
        #df['Score'] = df['Processed_Text'].apply(score)
        #df['Analysis'] = df['Score'].apply(analyze)

        ###---###           

        st.markdown("""---""")
        st.subheader('Sample')

        #sample_df = df.sample(10)

        # Display the updated DataFrame with sentiment predictions
        st.write('**Sample of 10 Rows with Sentiment Predictions:**')
        
        st.write(df.sample(10))
       
        st.markdown("""---""")
        st.markdown(
            """
            <div style="max-width=20px, background-color='red';"> 
            </div>
            """,
            unsafe_allow_html=True
        )
        st.title('Dataset Summary')
        
        avg_sentiment = np.mean(df['Value'])
        
        count = df['Sentiment'].value_counts()
        
        st.write(f'Average Sentiment Score: {avg_sentiment[0]:.2f}')
        st.write(count)
    
        
        width = 5
        height = 5
        fig,ax = plt.subplots(figsize=(width,height))
        ax.pie(count,labels=count.index,autopct='%1.1f%%',colors=['red', 'green', 'grey'])
        plt.title('Percentage of Sentiments')
        st.pyplot(fig)
        ### ###

        st.markdown("""---""")
        st.write('Click the button below, your dataset is available for download.')

@st.cache_data
def cacheDF(df):
        #return df.to_csv().encode('utf-8')
        return df.to_csv(index=False).encode('utf-8')
csv = cacheDF(df)

cached_csv = cacheDF(df)

# Displaying the download button with a dynamically generated file_name
if cached_csv is not None:
    # Get the name of the uploaded file without the extension
    file_name = upl.name.split('.')[0] if upl else "sentiments"

    # Set the file_name dynamically
    file_name = f'{file_name}_processed.csv'

    # Display the download button
    st.download_button(
        label='Download processed data',
        data=cached_csv,
        file_name=file_name,
        mime='text/csv'
    )




st.markdown(
    """
    <div style="text-align: center; padding: 10px;">
        Developed by <a href="#" target="_blank">Jerry Ade</a>
    </div>
    """,
    unsafe_allow_html=True
)

