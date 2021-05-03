import os
import pickle
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences


class LyricsGeneratorModel:
    def __init__(self):
        self._model = self._init_model()
        self._model.load_weights(f'{os.path.dirname(os.path.abspath(__file__))}/model_weight.h5')
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/model_meta.pickle', 'rb') as f:
            _, self._word_to_idx, self._idx_to_word = pickle.load(f)
        
    def _init_model(self):
        model = Sequential()
        model.add(Embedding(5002 , 8, input_length=5))
        model.add(Dropout(0.1))
        model.add(LSTM(512))
        model.add(Dropout(0.1))
        model.add(Dense(5002, activation='softmax'))
        model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def _texts_to_sequences(self, text, word_to_index):
        text = text.strip().split(' ')
        token_list = [word_to_index[x] for x in text]
        return token_list
    
    
    def _sample_output(self, probs, temperature=1.0):
        probs = [p**(1/temperature) for p in probs]
        sum_probs = sum(probs)
        probs = [p/sum_probs for p in probs]
        r = random.uniform(0, 1)
        acp = 0.
        for i, p in enumerate(probs):
            acp += p
            if acp >= r:
                return i
        return len(probs)-1

    def _temperature_sampling_decode(self, seed_text, max_gen_length, input_len, temperature):
        current_text = seed_text
        probs = []
        for _ in range(max_gen_length):
            current_idx = pad_sequences([self._texts_to_sequences(current_text, self._word_to_idx)], maxlen=input_len, padding='pre')
            next_token_idx_probs = self._model.predict(current_idx)[0]
            next_token_idx = self._sample_output(next_token_idx_probs, temperature)
            output_word = self._idx_to_word[next_token_idx]
            probs.append(next_token_idx_probs[next_token_idx])
            current_text += ' ' + output_word
        return current_text, probs
    
    def _post_process(self, inpt):
        return [e for e in ''.join([e.replace('<eos>', '\n') for e in inpt.split()]).split('\n') if e]

    def predict(self, seed_text, max_gen_length=50, temp=0.90):
        return self._post_process(self._temperature_sampling_decode(str(seed_text), max_gen_length, 5, temp)[0])