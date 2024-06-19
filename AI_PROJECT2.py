import tkinter as tk #For GUI
from tkinter import simpledialog #For GUI
import pandas as pd #For data manupilation
from sklearn.model_selection import train_test_split #for splitting data into training and testing sets
from sklearn.linear_model import Perceptron #for creating perceptron model
from sklearn.metrics import accuracy_score #for calculating the accuracy of the model
from io import StringIO #for reading the data from a string

#Data
data = """Math,Science,English,Pass/Fail
85,78,90,Pass
60,65,70,Fail
95,88,92,Pass
55,58,60,Fail
100,100,100,Pass
59,59,59,Fail
70,70,70,Pass
69,69,69,Fail
85,93,74,Pass
40,30,20,Fail
98,78,90,Pass
21,57,90,Fail
90,90,90,Pass
10,10,10,Fail
93,92,97,Pass
45,59,73,Fail
88,73,73,Pass
16,32,80,Fail
83,75,93,Pass
13,42,53,Fail
85,83,99,Pass
21,23,28,Fail
84,70,91,Pass
53,48,47,Fail
83,89,92,Pass
31,39,42,Fail
80,74,97,Pass
49,35,50,Fail
86,94,100,Pass
82,74,98,Pass
80,70,22,Fail
72,34,62,Fail
90,77,88,Pass
94,100,73,Pass
33,44,55,Fail
12,45,69,Fail
94,93,92,Pass
21,23,42,Fail
12,18,55,Fail
10,20,100,Fail
73,95,78,Pass
12,13,14,Fail
21,25,30,Fail
47,48,50,Fail
38,37,30,Fail
"""
data = pd.read_csv(StringIO(data)) #to read the data file as string
data['Pass/Fail'] = data['Pass/Fail'].map({'Pass': 1, 'Fail': 0}) #pass =1 and fail =0 to make it easier
X = data[['Math', 'Science', 'English']] #x contains the math and science and english columns
y = data['Pass/Fail'] # y contains the pass or fail column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #split the data,20% data for testing,80% for training,to split it the same everytime

perceptron = Perceptron(eta0=0.01, max_iter=1000, tol=0.001) #initialize the perceptron model learning rate=0.01, max ephoc=1000 and goal=0.001


perceptron.fit(X_train, y_train) #to train the perceptron
accuracy = accuracy_score(y_test, perceptron.predict(X_test)) #calculates the accuracy of the model comapred to the tests


root = tk.Tk() #GUI Frame
root.geometry("400x300") #Size of the Frame

def get_user_input():
    window = tk.Toplevel(root) #Frame
    window.geometry("400x300") # frame size
    tk.Label(window, text="Enter Math score:").pack() #label for math score
    math_entry = tk.Entry(window) #textbox
    math_entry.pack()
    tk.Label(window, text="Enter Science score:").pack() #label for Science score
    science_entry = tk.Entry(window)#textbox
    science_entry.pack()
    tk.Label(window, text="Enter English score:").pack() #label for English score
    english_entry = tk.Entry(window)#textbox
    english_entry.pack()

    def on_submit():
        math = int(math_entry.get()) # get the math score
        science = int(science_entry.get()) # get the science score
        english = int(english_entry.get()) # get the english score
        input_df = pd.DataFrame([[math, science, english]], columns=['Math', 'Science', 'English']) #create a dataframe that has the marks, it is similar to the data file
        result = perceptron.predict(input_df) #predict the result depending on the trained perceptron
        result_text = "Pass" if result[0] == 1 else "Fail" #show the result
        tk.Label(window, text=f"Prediction: {result_text}").pack() #label for the result

    tk.Button(window, text="Predict", command=on_submit).pack() #predict button

def update_model_parameters():
    param_window = tk.Toplevel(root) #Frame
    param_window.geometry("300x200") #frame size
    tk.Label(param_window, text="Enter Learning Rate:").pack() #label for LR
    lr_entry = tk.Entry(param_window)
    lr_entry.pack()
    tk.Label(param_window, text="Enter Max Epoch:").pack() # Label for Epoch
    max_iter_entry = tk.Entry(param_window)
    max_iter_entry.pack()
    tk.Label(param_window, text="Enter Goal:").pack() #Label for Goal
    tol_entry = tk.Entry(param_window)
    tol_entry.pack()

    def on_update():
        lr = float(lr_entry.get()) #get the LR
        max_iter = int(max_iter_entry.get()) #get the # of Epochs
        tol = float(tol_entry.get()) #get the goal
        perceptron.set_params(eta0=lr, max_iter=max_iter, tol=tol) #update the perceptron model
        perceptron.fit(X_train, y_train)  # retrain the model with new parameters
        updated_accuracy = accuracy_score(y_test, perceptron.predict(X_test)) #update the accuracy
        tk.Label(param_window, text=f"Model updated! New accuracy: {updated_accuracy*100:.2f}%").pack() #label for accuracy

    tk.Button(param_window, text="Update", command=on_update).pack() #update button

def show_weights_and_threshold():
    weights = perceptron.coef_ #retrieve the weight coeffcients of the trained perceptron
    threshold = perceptron.intercept_ #retrieve the thresolhd
    weights_str = f"Weights: {weights[0]}" #string for the model
    threshold_str = f"Threshold: {threshold[0]}" # threshold of the model
    weights_window = tk.Toplevel(root)
    weights_window.geometry("400x200") #frame size
    tk.Label(weights_window, text=weights_str).pack() #label for weight
    tk.Label(weights_window, text=threshold_str).pack() #label for threshold

tk.Label(root, text=f"Initial Model Accuracy: {accuracy*100:.2f}%").pack() #label for acc
tk.Button(root, text="Enter Test Data", command=get_user_input).pack() #enter data button
tk.Button(root, text="Update Model Parameters", command=update_model_parameters).pack()# update button
tk.Button(root, text="Show Weights and Threshold", command=show_weights_and_threshold).pack() #show weight and threshold button
root.mainloop()
