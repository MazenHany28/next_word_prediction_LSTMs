import tkinter as tk
from tkinter import ttk
from keras.models import load_model
import pickle
import numpy as np

class TextPredictorGUI:
    def __init__(self, master, model, tokenizer):
        self.master = master
        master.title("Text Predictor")

        # Define styles
        self.style = ttk.Style(master)
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 12))
        self.style.configure('TButton', background='#4caf50', foreground='white', font=('Arial', 12))

        self.frame = ttk.Frame(master)
        self.frame.grid(row=0, column=0, padx=10, pady=10)

        self.label = ttk.Label(self.frame, text="Enter text:")
        self.label.grid(row=0, column=0, padx=5, pady=5, sticky='w')

        self.entry = ttk.Entry(self.frame, width=50)
        self.entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        self.predict_button = ttk.Button(self.frame, text="Predict Next Word", command=self.predict)
        self.predict_button.grid(row=1, column=1, padx=5, pady=5, sticky='ew')

        self.result_label = ttk.Label(master, text="", font=('Arial', 14, 'bold'))
        self.result_label.grid(row=1, column=0, padx=10, pady=10)

        # Pass the pre-trained text generation pipeline to the GUI
        self.model = model
        self.tokenizer = tokenizer

        # Make the entry and button responsive
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)

    def predict(self):
        text = self.entry.get()
        text = text.split(" ")
        text = text[-3:]
        text = " ".join(text)
        sequence = self.tokenizer.texts_to_sequences([text])
        sequence = np.array(sequence)
        preds = np.argmax(self.model.predict(sequence))
        predicted_word = ""

        for key, value in self.tokenizer.word_index.items():
            if value == preds:
                predicted_word = key
                break

        self.result_label.config(text="Predicted Next Word: " + predicted_word)

if __name__ == "__main__":
    # Load the model and tokenizer
    model = load_model('next_words.h5')
    tokenizer = pickle.load(open('token.pkl', 'rb'))

    root = tk.Tk()
    root.geometry("500x200")
    root.resizable(False, False)  # Disable resizing
    gui = TextPredictorGUI(root, model, tokenizer)
    root.mainloop()
