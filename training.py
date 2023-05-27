import random
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
import numpy as np
import pickle
import json
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Create empty variables
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']
data_file = open('intents.json').read()

# Load the data from 'intents.json'
intents = json.loads(data_file)

# Process each intent and pattern in the intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize the pattern into words
        w = nltk.word_tokenize(pattern)

        # Extend the 'words' list with the tokenized words
        words.extend(w)

        # Append the tuple (words, tag) to 'documents' list
        documents.append((w, intent['tag']))

        # Add the intent's tag to 'classes' if it's not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Preprocess the words
words = [lemmatizer.lemmatize(w.lower())
         for w in words if w not in ignore_words]

# Remove duplicates and sort the words list
words = sorted(list(set(words)))

# Save 'words' and 'classes' as pickle files for later use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create the training data
training = []
output_empty = [0] * len(classes)

# Process each document
for doc in documents:
    bag = []
    pattern_words = doc[0]

    # Lemmatize and lowercase the pattern words
    pattern_words = [lemmatizer.lemmatize(
        word.lower()) for word in pattern_words]

    # Create a bag-of-words representation for the current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Create the output row (one-hot encoded) for the current document's class
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    # Append the bag-of-words representation and output row to 'training'
    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)

# Convert 'training' to a NumPy array
training = np.array(training)

# Extract the input data ('train_x') and output data ('train_y') from 'training'
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Print information about the training data
print("Training data created")

# Create a sequential model
model = Sequential()

# Add layers to the model
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Configure the optimizer and compile the model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=200, batch_size=5, verbose=1)

# Save the model to a file
model.save('chatbotmodel.h5', hist)

# Print a message indicating that the model has been created
print("Model created")