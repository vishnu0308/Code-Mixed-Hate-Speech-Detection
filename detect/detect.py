
from googletrans import Translator
from keras import backend as K
from tensorflow.keras.layers import GlobalAveragePooling1D, Dense, multiply, GlobalMaxPooling1D, Lambda
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import re
import pandas as pd
import regex
from library import *
import googletrans
import dill
import json
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate, Dropout, GlobalMaxPooling1D, concatenate, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint

print("--------Initialising Model---------------")

dill.load_session('translated.db')


def trac1_dataset_preprocess():
    df1 = load_dataset("../Dataset/TRAC1/agr_hi_train.csv")
    df2 = load_dataset("../Dataset/TRAC1/agr_en_train.csv")
    df = pd.concat([df1, df2])
    df = preprocess_text(df)
    df["message"].fillna('', inplace=True)
    df1 = load_dataset("../Dataset/TRAC1/agr_hi_dev.csv")
    df2 = load_dataset("../Dataset/TRAC1/agr_en_dev.csv")
    val_df = pd.concat([df1, df2])
    val_df = preprocess_text(val_df)
    y_train = df["class"]
    y_test = val_df["class"]
    x_train = df["message"]
    x_test = val_df["message"]
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = trac1_dataset_preprocess()


def concat_normal_translate(lst, trans_lst):
    lst = list(lst)
    trans_lst = list(trans_lst)
    conc_lst = []
    for i in range(len(lst)):
        conc_lst.append(lst[i] + " "+trans_lst[i])
    return conc_lst


x_train_concat = concat_normal_translate(x_train, x_train_translated)
x_test_concat = concat_normal_translate(x_test, x_test_translated)


def get_max_text_len(msgs):
    return max(list(map(lambda msg: len(msg), msgs)))


def convert_classes_to_nums(y_train, y_test):
    classes = y_train
    le = LabelEncoder()
    integer_labels = le.fit_transform(classes)
    y_train = integer_labels
    y_test = le.transform(y_test)
    return y_train, y_test, le


def print_classification_metrics(y_pred, y_test):
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
    print("F1-score: {:.2f}".format(f1))


def infer_class(model, tokenizer, max_len, label_encoder, text):
    # Tokenize the input text
    text_sequence = tokenizer.texts_to_sequences([text])
    # Pad the sequence to the maximum length
    padded_sequence = pad_sequences(text_sequence, maxlen=max_len)
    # Make the prediction
    prediction = model.predict(padded_sequence, verbose=0)[0]
    # Convert the prediction to the actual label
    predicted_label = np.argmax(prediction)
#     predicted_class = label_encoder.inverse_transform([predicted_label])[0]
    return predicted_label


def predict_all(model, tokenizer, max_len, label_encoder, x_test):
    y_pred = []
    for msg in list(x_test):
        pred_class = infer_class(model, tokenizer, max_len, label_encoder, msg)
        y_pred.append(pred_class)
    return y_pred


def additional_metrics(model, tokenizer, max_len, le):
    x_train, x_test, y_train, y_test = trac1_dataset_preprocess()
    y_test = le.transform(y_test)
    y_pred = predict_all(model, tokenizer, max_len, le, x_test)
    print_classification_metrics(y_pred, y_test)


def se_block(in_block, ch, ratio=16):
    x = GlobalAveragePooling1D()(in_block)
    x = Dense(ch//ratio, activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
    x = Reshape((1, num_filters))(x)
    tile_layer = Lambda(lambda x: K.tile(x, [1, in_block.shape[1], 1]))(x)
    x = multiply([in_block, tile_layer])
    return x


####
x_train, x_test, y_train, y_test = trac1_dataset_preprocess()


text_data = x_train_concat
val_text_data = x_test_concat
# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
word_index = tokenizer.word_index

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(text_data)
val_sequences = tokenizer.texts_to_sequences(val_text_data)
# Pad sequences to a fixed length
max_len = get_max_text_len(x_train)  # Set the maximum sequence length
data = pad_sequences(sequences, maxlen=max_len)
val_data = pad_sequences(val_sequences, maxlen=max_len)

# Convert labels to one-hot encoding
y_train, y_test, le = convert_classes_to_nums(y_train, y_test)
labels = to_categorical(y_train)
test_labels = to_categorical(y_test)

# embedding matrix
embedding_dim = 200
num_words = len(tokenizer.word_index) + 1
embedding_matrix = np.zeros((num_words, embedding_dim))

embedding_layer = Embedding(num_words, embedding_dim, weights=[
                            embedding_matrix], trainable=True)

x_train, y_train = data, labels
x_test, y_test = val_data, test_labels


# Define hyperparameters
embed_dim = embedding_dim
# embed_dim=100
num_filters = 64
filter_sizes = [2, 3, 4]
dropout_rate = 0.5
batch_size = 64
epochs = 5

print(max_len)
# Define input layer
input_layer = Input(shape=(max_len,))

# Add embedding layer
embedding = embedding_layer(input_layer)
print(embedding.shape)

# Add parallel convolutional layers with max pooling and global max pooling
conv_layers = []
for filter_size in filter_sizes:
    conv_layer = Conv1D(filters=num_filters,
                        kernel_size=filter_size, activation='relu')(embedding)
    print(conv_layer.shape)
    se_layer = se_block(conv_layer, num_filters)
    pool_layer = MaxPooling1D(pool_size=max_len - filter_size + 1)(se_layer)
    conv_layers.append(GlobalMaxPooling1D()(pool_layer))
concat_layer = concatenate(conv_layers, axis=1)

# Add dropout layer
dropout_layer = Dropout(dropout_rate)(concat_layer)

# Add output layer
output_layer = Dense(3, activation='softmax')(dropout_layer)

# Define model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile model with binary cross-entropy loss and Adam optimizer
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])


#########

# Define input layer
input_layer = Input(shape=(max_len,))

# Add embedding layer
embedding = embedding_layer(input_layer)
print(embedding.shape)

# Add parallel convolutional layers with max pooling and global max pooling
conv_layers = []
for filter_size in filter_sizes:
    conv_layer = Conv1D(filters=num_filters,
                        kernel_size=filter_size, activation='relu')(embedding)
    print(conv_layer.shape)
    se_layer = se_block(conv_layer, num_filters)
    pool_layer = MaxPooling1D(pool_size=max_len - filter_size + 1)(se_layer)
    conv_layers.append(GlobalMaxPooling1D()(pool_layer))
concat_layer = concatenate(conv_layers, axis=1)

# Add dropout layer
dropout_layer = Dropout(dropout_rate)(concat_layer)

# Add output layer
output_layer = Dense(3, activation='softmax')(dropout_layer)

# Define model
best_model = Model(inputs=input_layer, outputs=output_layer)

# Compile model with binary cross-entropy loss and Adam optimizer
best_model.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])

best_model.load_weights('concat_glove_200_cnn1d_seblock_orig.h5')

print("---------------Loaded Model-------------------------")


def translate(msg):
    translator = Translator()
    tr = translator.translate(msg).text
    return tr


def infer_class(model, tokenizer, max_len, label_encoder, text):
    text += " " + translate(text)
    # Tokenize the input text
    text_sequence = tokenizer.texts_to_sequences([text])
    # Pad the sequence to the maximum length
    padded_sequence = pad_sequences(text_sequence, maxlen=max_len)
    # Make the prediction
    prediction = model.predict(padded_sequence, verbose=0)[0]
    # Convert the prediction to the actual label
    predicted_label = np.argmax(prediction)
    predicted_class = label_encoder.inverse_transform([predicted_label])[0]
    return predicted_class


while True:
    text = input("Text to classify (Press q to exit) : ")
    text = text.strip().lower()
    if text == "q":
        break
    print(infer_class(best_model, tokenizer, max_len, le, text))
