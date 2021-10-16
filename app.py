import pandas as pd
import os, re, json, string, random, threading, time
#import tqdm.notebook as tqdm
import pickle
import numpy as np

import nltk

try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    pass

# import tag_helper as helper
# import tag_helper2 as helper2
#import paraphrase2 as paraphrase
import st_helper as sth

import datetime
import streamlit as st
import streamlit.components.v1 as components

pd.set_option('max_colwidth', 1500)
#pd.options.display.max_rows = 1000
pd.set_option('display.max_rows', 1000)

eula_name_dict = {'C0': 'audit_rights',
 'C1': 'licensee_indemnity',
 'C2': 'licensor_indemnity',
 'C3': 'license_grant',
 'C4': 'eula_others',
 'C5': 'licensee_infringement_indemnity',
 'C6': 'licensor_exemption_liability',
 'C7': 'licensor_limit_liabilty',
 'C8': 'software_warranty'}

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

#read in the 100 Eulas
all_eulas_100 = pd.read_csv(r'./st_eula_df15Oct21.csv', encoding="utf-8")

#convert all to string types - fix pyarrow/numpy bug.
all_eulas_100['len'] = all_eulas_100['len'].astype(str)

st.image(r"./sunrise2.jpg")

#print out EULAs
st.markdown(
""" 
###### This App demonstrates the use of machine learning technology in legal practice. 
###### Predict the presence of 8 key clauses in a standard end-user software license agreement (EULA) using AI technology. 
###### These are:  Class 0 – Licensor's audit rights; Class 1 – Licensee's indemnity to Licensor; Class 2 – Licensor's indemnity to Licensee; Class 3 – License grant; Class 4 – Others (control class); Class 5 – Licensee's IP infringement indemnity to Licensor; Class 6 – Licensor's exemptions of liability; Class 7 – Licensor's limitation of liability; Class 8 – Software warranty.
###### Specifically, the App uses a distilled BERT pre-trained model published by Google [State of the art language model for NLP](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270 ). 
###### Step 1 – View 3 randomly selected EULAs at a time. 
###### Step 2 – Choose the EULA for prediction.
###### Step 3 – Load Model 
###### Step 4 – Run Predict. 
###### Please note that it may take between 30-90 seconds for the predictions as the process is quite computationally intensive.   
_Copyright ML Kok, Oct 2021_
"""
)


sample_eula_pd = None
def get_sample_eulas():
    del st.session_state['sample_index']

#Get random EULAs X 3
st.warning("Step 1: Click Button to view randomly selected EULAs, 3 at a time (hover mouse over eula_text field to see the entire EULA).")
st.button("Sample EULAs", on_click=get_sample_eulas)
st.markdown(
r""" 
__Samples (note hover mouse over eula_text field to see the entire EULA):__
""", unsafe_allow_html=False)
#st.write(all_eulas_100.sample(n=3)['eula_text'])
sample_eula_pd = all_eulas_100.sample(n=3)
if not 'sample_index' in st.session_state:
    st.session_state['sample_index'] = sample_eula_pd.index
    st.session_state['sample_eula_pd'] = sample_eula_pd

    
st.write(st.session_state['sample_eula_pd'])
#%%
def get_selection_idx(option):
    st.session_state['selected_eula_index'] = option

#select EULA to classify
st.warning("Step 2: Select the EULA by index to analyse using the Drop-Down Box.")
option = None
option = st.selectbox('Select EULA by index', st.session_state['sample_index'], index=0, on_change=get_selection_idx, args=(option,))
st.session_state['selected_eula_index'] = option


st.write('You selected:')
all_eulas_100.loc[option]
if 'selected_eula_index' in st.session_state:
    st.write("Selected index: ", st.session_state['selected_eula_index'] )

#%%
#Get the 1-grams for each eula and re-arrange the columns.
all_eulas_100['1_gram'] = all_eulas_100['eula_text'].apply(sth.create_ngrams, args=(1,))
all_eulas_100 = all_eulas_100[['eula_name', 'eula_text', 'len', '1_gram']]

#Get the namse and text columns into a list.
#all_eulas_100_list = [row['eula_name'] + " : " + row['eula_text'][:]   for idx, row in all_eulas_100.iterrows()]
#st.write(all_eulas_100.head())

#extract sentences from the test EULA
if 'selected_eula_index' in st.session_state and st.session_state['selected_eula_index']:
    st.write("Printing first three sentences from selected EULA (hover mouse over 'eula_text' field to see more): ")
    eula_sentences_test = [sent for sent in all_eulas_100.loc[st.session_state['selected_eula_index']]['1_gram'] ]
    st.write(eula_sentences_test[:3])
    st.session_state['eula_sentences_test'] = eula_sentences_test

#Get tokenizer
# @st.cache
# def get_model_tokenizer():
#     model = AutoModelForSequenceClassification.from_pretrained("./model")
#     tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased', use_fast=False)
#     return tokenizer, model

#tokenizer and model
#@st.cache
def fetch_model():
    #global tokenizer, model
    if not 'model_loaded' in st.session_state:
        with st.spinner(text='Model loading in progress ...'):
            # model = AutoModelForSequenceClassification.from_pretrained("./model")
            # tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased', use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained("kloon99/KML_Software_License_v1")
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased', use_fast=False)
            #tokenizer, model = get_model_tokenizer()
            st.session_state['model'] = model
            st.session_state['tokenizer'] = tokenizer
            st.session_state['model_loaded'] = True


st.markdown("""
###### Before running the prediction, we fetch the pre-trained model and compatible tokenizer.
""")
st.warning("Step 3: Click Buttton to fetch the pre-trained model and tokenizer.")
st.button("Fetch model", on_click=fetch_model)

if 'model_loaded' in st.session_state and st.session_state['model_loaded'] == True:
        st.success('Done, model loaded')
        st.write( st.session_state['model'].__class__, st.session_state['tokenizer'].__class__)


####################### Prediction #######################################

# st.write(all_eulas_100.loc[SELETED_EULA_IDX]['1_gram'][:5] )
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

#Create the test dataset - pytorch requirement.
class TestDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

#Get predictions
def get_predictions():
    with st.spinner(text='Prediction in progress ...'):
        #global model, test_dataset, eula_sentences_test, predict_pd_top, tokenizer, predict_pd, eula_sentences_test
        global model, tokenizer, predict_pd, eula_sentences_test
        if 'model' in st.session_state:
            model = st.session_state['model'] 
            tokenizer = st.session_state['tokenizer']
        else:
            st.error("Model not loaded, please click get model button to load model.")
            st.stop()
        ######
        if 'eula_sentences_test' in st.session_state and len(st.session_state['eula_sentences_test']) > 0:
            
            eula_sentences_test = [sent for sent in st.session_state['eula_sentences_test'] ]
        else:
            st.error("Sample EULA not selected!")
            st.stop()
        test_encodings = tokenizer(eula_sentences_test, truncation=True, padding=True)
        test_dataset = TestDataset(test_encodings)
        trainer = Trainer(model=model)
        predictions = trainer.predict(test_dataset)
        #predictions.predictions
        #############
        ##get predicted class and scores
        from scipy.special import softmax
        import numpy as np
        pred_list = []
        for _pred in predictions.predictions:
            x=np.argmax(_pred)
            y=np.max(softmax(_pred))
            pred_list.append([x,y]) 
        ################
        x, y = zip(*pred_list)
        #################
        #create dataframe of eula, predicted class and scores
        eulas = eula_sentences_test
        predict_dict = {'eulas': eulas, "class": x, "score" : y}
        predict_pd = pd.DataFrame({'eulas': eulas, "class": x, "score" : y}, columns=['eulas','class','score' ])
        st.session_state['predict_pd'] = predict_pd
        #predict_pd.head(3)
        ##########
        #preds = np.argmax(predictions.predictions, axis=-1)
        ############

st.warning("Step 4: Click Button to predict, may take between 1-2 minutes to run.")
st.button("Model Predict", on_click=get_predictions)
st.markdown(
""" ###### Please note that it may take between 1-2 minutes for the predictions as the process is quite computationally intensive.   
""")

#globals
# predict_pd_top = None
# predict_pd = None

if 'predict_pd' in st.session_state:
    #st.write(st.session_state['predict_pd'].sort_values(['class', 'score'],ascending=False).groupby('class').head(1).to_html())
    # components.html(
    #     '''<html>
    #     <head>Results</head>
    #     <body>
    #         {}
    #     </body>
    #     </html>'''.format(st.session_state['predict_pd'].sort_values(['class', 'score'],ascending=False).groupby('class').head(1).to_html()),
    #     width=900, height=400, scrolling=True
    #    )
    #st.write(st.session_state['predict_pd'].sort_values(['class', 'score'],ascending=False).groupby('class').head(1).to_html())
    raw_results = st.session_state['predict_pd'].sort_values(['class', 'score'],ascending=False).groupby('class').head(1).to_html()
    raw_results = re.sub(r'dataframe', 'table', raw_results, 0, re.VERBOSE | re.MULTILINE)
    results_htm = sth.get_results_html(raw_results)
    components.html(results_htm,
        width=700, height=600, scrolling=True)


def show_class_details():
    st.session_state["show_details_class"] = 1
    

st.button("Click for a detailed description of class categories", on_click=show_class_details)

if "show_details_class"  in st.session_state:
    st.markdown(
        """
        *	Class 0 – Licensor's audit rights: Often the software license granted to the licensee is subject to limitations and restrictions. This right of audit permits the licensor to inspect the licensor's systems and environment which run the software in order to determine whether or not the licensor has kept within those limits and restrictions. This provision can be of concern to the licensee as its systems and environment may contain sensitive information. The licensor may need to limit the scope of such audits so that the licensee's sensitive information is not exposed to the licensor during such audits.  
        *	Class 1 – Licensee's indemnity: Under this provision, the licensee agrees to fully indemnify the licensor for the losses it suffers arising out of the licensee's use of the software. Some indemnities are not fault-based, i.e., the licensee is obliged to indemnify the licensor even if the licensee is not in any breach of the license terms; these need to be avoided or at least appropriately circumscribed.  
        *	Class 2 – Licensor's indemnity: Under this category, it is expected that the licensor should undertake to indemnify the licensee against claims by third parties the licensor's software infringes a third party's intellectual property rights. Generally, it is a concern if this clause is not present.
        *	Class 3 – License grant: This provision contains the terms of the license grant to the licensee, setting out the permitted scope of use and also the usage limits. Ensure that commercial usage is permitted, including use of the software in customer centric applications, as is necessary.
        *	Class 4 – Others: This is a control class; it represents other clauses of a EULA not identified by the other classes.  
        *	Class 5 – Licensee's IP infringement indemnity: This provision generally applies when the licensee may upload it’s content onto the licensor’s platform or portal. If such content infringes any third party’s intellectual property, then the licensee is expected to indemnify the licensor from any ensuing third party claims against the licensor.  
        *	Class 6 – Licensor's exemptions of liability: Other than bespoke software, most software licensors will exempt their liability for indirect losses, loss of profits, loss of business or goodwill etc. of the licensee arising from the use of standard software offerings.  
        *	Class 7 – Licensor's limitation of liability: Apart from the outright exemptions of liability under class 6, under this provision, the licensor seeks to limit its liability to the licensee for direct losses which exceeds a pre-defined limit, which in many cases will be the fees paid for the software over a period of 6 to 12 months.  
        *	Class 8 – Software warranty – This sets out the limited warranty for the software, in that it will function according to its published specifications for a limited period, usually between 1 to 6 months.   
        (***Please note that the above is for informational purposes only and not meant as legal advice.)
        """
    )