import random
# Import the necessary libraries
from keras.models import load_model
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import json

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the trained model
model = load_model('chatbotmodel.h5')

# Load the intents JSON file
intents = json.loads(open('intents.json').read())

# Load the preprocessed words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Function to clean up the user's sentence


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words

# Function to convert the cleaned sentence into a bag of words representation


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

# Function to predict the class of a user's sentence


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    # Set an error threshold
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get a response based on the predicted class


def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


while True:
    # create a gui for chatbot using tkinter
    import pyttsx3
    import tkinter
    from tkinter import *
    root = Tk()
    root.title("Pixer Chatbot")
    root.geometry("400x500")
    root.resizable(width=FALSE, height=FALSE)
    root.configure(bg="#0045ff")
    # create main menu
    main_menu = Menu(root)
    root.config(menu=main_menu)
    # create chat window
    chat_window = Text(root, bd=0, bg="#E6E6FA", width=50,
                       height=8, font=("Arial", 12), state=DISABLED)
    chat_window.place(x=6, y=6, height=385, width=370)
    # create scroll bar
    scrollbar = Scrollbar(root, command=chat_window.yview, cursor="arrow")
    scrollbar.place(x=375, y=6, height=385)
    chat_window.configure(yscrollcommand=scrollbar.set)
    # create send button to send message
    send_button = Button(root, text="Send", bg="#000000", activebackground="#004080", fg="white",
                         width=12, height=5, font=("Arial", 12), cursor="hand2")
    send_button.place(x=270, y=410, height=45)
    # create voice button to send message
    voice_button = Button(root, text="Voice", bg="#000000", activebackground="#004080", fg="white",
                          width=12, height=5, font=("Arial", 12), cursor="hand2")
    voice_button.place(x=270, y=465, height=45)
    # create input field for message
    message_field = Entry(root, bg="white", width=32, font=("Arial", 12))
    message_field.place(x=6, y=410, height=100, width=260)

    # create function to scroll down automatically
    def scroll_down():
        chat_window.yview(END)

    # create function to send message take message from message field
    def send():
        import webbrowser
        message = message_field.get()
        message_field.delete(0, END)
        # if message is quit then exit
        if message == "Bye" or message == "bye":
            res = "See you later";
            engine = pyttsx3.init()
            engine.say(res)
            engine.runAndWait()
            root.destroy()
        # if message is out of intent then print sorry
        if message != '':
            chat_window.config(state=NORMAL)
            chat_window.insert(END, "You: " + message + '\n\n')
            chat_window.config(foreground="#442265", font=("Purple", 12))
            ints = predict_class(message, model)
            res = get_response(ints, intents)
            chat_window.insert(END, "Pixer: " + res + '\n\n')
            chat_window.config(state=DISABLED)

            # if message search in google search engine then open google
            if message == "Search":
                res = 'opening google search engine'
                webbrowser.open('https://www.google.com/')
            # if message search in youtube search engine then open youtube
            if message == "Youtube":
                res = 'opening youtube'
                webbrowser.open('https://www.youtube.com/')
            # if message search in facebook search engine then open facebook
            if message == "Facebook":
                res = 'opening facebook'
                webbrowser.open('https://www.facebook.com/')
            # if message search in instagram search engine then open instagram
            if message == "Instagram":
                res = 'opening instagram'
                webbrowser.open('https://www.instagram.com/')

              # convert text to speech
            engine = pyttsx3.init()
            engine.say(res)
            engine.runAndWait()
            scroll_down()

    send_button.config(command=send)

    # click on voice button to get voice input
    def voice():
        import speech_recognition as sr
        import pyttsx3
        import pyaudio

        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        r = sr.Recognizer()
        with sr.Microphone() as source:
            engine.say("Pixer is listening...")
            engine.runAndWait()
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio)
                # display text in message field
                message_field.insert(0, text)
            except:
                engine.say("Sorry could not recognize what you said")
                engine.runAndWait()

    voice_button.config(command=voice)

    # search from google if input is not availabe in patterns
    def search():
        import webbrowser
        import speech_recognition as sr
        import pyttsx3
        import pyaudio

        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        r = sr.Recognizer()
        with sr.Microphone() as source:
            engine.say("Pixer is listening...")
            engine.runAndWait()
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio)
                # display text in message field
                webbrowser.open('https://www.google.com/search?q='+text)
            except:
                engine.say("Sorry could not recognize what you said")
                engine.runAndWait()

    def enter_function(event):
        send()
    # bind main window with enter key
    root.bind('<Return>', enter_function)
    # create function to exit

    def exit_function():
        exit()
    # bind main window with exit key
    root.bind('<Escape>', exit_function)
    root.mainloop()
    # create function to esc key

    def esc_function():
        root.destroy()
    # bind main window with esc key
    root.bind('<Escape>', esc_function)

    root.mainloop()