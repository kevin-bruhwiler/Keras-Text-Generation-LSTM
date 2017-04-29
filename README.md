# Keras-Text-Generation-LSTM
An character level LSTM neural network that generates text, written in python and utilizing keras

Very straightforward implementation. Loads a text file and divides it up into segments of length 'seqLen'. Attempts to predict the character following each sequence. After every training epoch it generates several thousand characters of text and writes them to 'output.txt'

Several megabytes of text should be supplied to get good results. Model struggles to converge with less than one megabyte.
