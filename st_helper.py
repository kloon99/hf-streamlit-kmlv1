import streamlit as st
import streamlit.components.v1 as components
def get_results_html(raw_results):
    return """
        <!doctype html>
        <html lang="en">
        <head>
            <!-- Required meta tags -->
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">

            <!-- Bootstrap CSS -->
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">

            <title></title>
        </head>
        <body>
            <p class="text-primary">
                ‘score’: The score indicates how close that clause matches the relevant class of the 8 EULA provisions (ranges from 0.1 to 1.0).<br/>
                **<b>Generally, the score needs to be 0.995 and above to be a close match, however in some instances even a lower score indicates an interesting variation of the clause under consideration.</b><br/>
                ‘class’: These are the key categories of clauses/provisions in a standard EULA. <br/>
                Class 0 – Licensor's audit rights &nbsp;&nbsp; Class 1 – Licensee's indemnity to Licensee &nbsp;&nbsp; Class 2 – Licensor's indemnity to Licensee<br/>
                Class 3 – License grant &nbsp;&nbsp; Class 4 – Others (control class) &nbsp;&nbsp; Class 5 – Licensee's IP infringement indemnity to Licensor &nbsp;&nbsp; Class 6 – Licensor's exemptions of liability  <br/>
                Class 7 – Licensor's limitation of liability  &nbsp;&nbsp;  Class 8 – Software warranty &nbsp;&nbsp; (click button below for a more detailed description)
            </p>
            <h5>Prediction Results </h5>
            {}
        </body>
        </html>
    """.format(raw_results)

#updated 10 Aug 21
def create_ngrams(_text, n=1):
    from nltk import sent_tokenize, word_tokenize
    all_sents = sent_tokenize(_text)
    _sents_list = []
    for i in range(len(all_sents)):
        _sents = all_sents[i:i+n]
        n_sents = ' '.join(_sents)
        #_sents_list += _sents
        n_sents = ' '.join(n_sents.split())
        _sents_list.append(n_sents)      
    return _sents_list

# "this is a call back test"

# def form_callback():
#     st.write(st.session_state.my_slider)
#     st.write(st.session_state.my_checkbox)

# with st.form(key='my_form'):
#     slider_input = st.slider('My slider', 0, 10, 5, key='my_slider')
#     checkbox_input = st.checkbox('Yes or No', key='my_checkbox')
#     submit_button = st.form_submit_button(label='Submit', on_click=form_callback)
# #%%


# # text_ph = None
# # def change_text():
# #     global text_ph
# #     text_ph.text(st.session_state.name)
# #     st.write('your nameL', st.session_state.name)


# with st.container():
#     st.text_input('tell me your name',  key="name", on_change=None)
#     text_ph = st.empty()
#     if  st.session_state.name:
#         text_ph.text("Your name is {}".format(st.session_state.name))

