import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
import numpy as np

def LoadText():
    with open("C:\\Users\\Kevin\\Documents\\shakespeare.txt", "r") as text_file:
        data = text_file.read()
    text = list(data)
    inputSize = len(text)
    data = sorted(list(set(text)))
    dataSize = len(data)
    return text, inputSize, dataSize, data

text, textSize, uniqueChars, chars = LoadText()
ci = dict((c,i) for i, c in enumerate(chars))
ic = dict((i,c) for i, c in enumerate(chars))

seqLen = 80
sentences = []
nextChars= []
for i in range(0,textSize-seqLen, 3):
    sentences.append(text[i:i+seqLen])
    nextChars.append(text[i+seqLen])

x_train = np.zeros((len(sentences), seqLen, uniqueChars), dtype=np.bool)
y_train = np.zeros((len(sentences), uniqueChars), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for j, char in enumerate(sentence):
        x_train[i, j, ci[char]] = 1
    y_train[i, ci[nextChars[i]]] = 1
    
model = Sequential()
model.add(LSTM(150, input_shape=(seqLen, uniqueChars), return_sequences=True, implementation=1))
model.add(Dropout(0.25))
model.add(LSTM(150, input_shape=(seqLen, uniqueChars), implementation=1))
model.add(Dropout(0.5))
model.add(Dense(uniqueChars, activation='softmax'))
model.compile(optimizer=keras.optimizers.rmsprop(lr=0.007), loss='categorical_crossentropy')

with open("C:\\Users\\Kevin\\Documents\\output.txt", "w") as text_file:
    for it in range(50):
        print('=' * 50)
        print('iteration: ', it)
        model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1)
        startIndex = np.random.randint(0,textSize-seqLen-1)
        generated = ''
        sentence = text[startIndex:startIndex+seqLen]
        generated = generated.join(sentence)
        print("generating with seed: ", generated)

        for i in range(4000):
            x = np.zeros((1, seqLen, uniqueChars))
            for j, char in enumerate(sentence):
                x[0, j, ci[char]] = 1

            prediction = model.predict(x, batch_size=1, verbose=0)[0]
            index = np.random.choice(uniqueChars, p=prediction)
            char = ic[index]
            generated += char
            sentence.append(char)
            sentence.pop(0) 
        text_file.write(generated)
