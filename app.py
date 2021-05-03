import streamlit as st
from tensorflow_model.model import LyricsGeneratorModel

# from tensorflow_model_mini.model import LyricsGeneratorModel
MODEL = LyricsGeneratorModel()

#===========================================#
#              Streamlit Code               #
#===========================================#
desc = "This project is part of the final project in NLP_SYS class (2110594)"

st.title('Thai Lyrics Generator v3.1')
st.write(desc)

inpt_seed = st.text_input('Seed Text', value='เพราะ', max_chars=10)
# inpt_artist = st.selectbox(
#     'Artist Name', ['ANY', 'BNK48', 'POTATO', 'CARABAO', 'SEK_LOSO', 'GRASSHOPPER'])

# ARTIST_MAP = {
#     'ANY': 'default',
#     'BNK48': 'bnk48',
#     'POTATO': 'potato',
#     'CARABAO': 'carabao',
#     'SEK_LOSO': 'sekloso',
#     'GRASSHOPPER': 'grasshopper'
# }
inpt_artist = st.selectbox(
    'Artist Name', ['ANY', 'BNK48'])

ARTIST_MAP = {
    'ANY': 'default',
    'BNK48': 'bnk48'
}

if st.button('Generate'):
    if len(str(inpt_seed)) == 0:
        st.markdown(
            "<h3 style='text-align: center; color: red;'>EMPTY_STRING_NOT_ALLOWED</h3>", unsafe_allow_html=True)
    else:
        generated_text = '<br>'.join(MODEL.predict(
            inpt_seed, max_gen_length=50, artist=ARTIST_MAP[inpt_artist])[:10])
        # generated_text = '<br>'.join(MODEL.predict(
        #     inpt_seed, max_gen_length=80)[:10])
        st.markdown(f"<center>{generated_text}</center>",
                    unsafe_allow_html=True)
