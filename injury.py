import tkinter as tk
from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from PIL import Image, ImageTk
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier  # Import BaggingClassifier directly

# Load your dataset
df = pd.read_csv('C:/Users/sainath/PycharmProjects/injuryPrediction/week_approach4.csv')

df = df.drop(['avg training success', 'min training success', 'max training success', 'avg training success.1', 'max training success.1', 'min training success.1'], axis = 1)
df = df.drop(['avg training success.2', 'max training success.2', 'min training success.2', 'avg exertion', 'min exertion', 'max exertion'], axis = 1)
df = df.drop(['avg exertion.1', 'min exertion.1', 'max exertion.1', 'avg exertion.2', 'min exertion.2', 'max exertion.2', 'max km one day'], axis = 1)
df = df.drop(['avg recovery', 'min recovery', 'max recovery', 'avg recovery.1', 'min recovery.1', 'max recovery.1', 'avg recovery.2', 'min recovery.2', 'max recovery.2'], axis = 1)
df = df.drop(['rel total kms week 0_1', 'rel total kms week 0_2', 'rel total kms week 1_2'], axis = 1)

df1 = df.sort_values(by = 'Athlete ID');

shuffled_df1 = df.sample(frac=1,random_state=4)

# Put all the fraud class in a separate dataset.
injury_df1 = shuffled_df1.loc[shuffled_df1['injury'] == 1]
#Randomly select 492 observations from the non-fraud (majority class)
non_injured_df1 = shuffled_df1.loc[shuffled_df1['injury'] == 0].sample(n=2875)

# Concatenate both dataframes again
normalized_df = pd.concat([injury_df1, non_injured_df1])

# Define X and Y
X = normalized_df.drop('injury', axis = 1)
Y = normalized_df['injury']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, stratify = Y)
rus = RandomUnderSampler(random_state=0)
X_train, Y_train =rus.fit_resample(X_train,Y_train)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
bag = BaggingClassifier(n_estimators = 35)  # Use BaggingClassifier directly
bag.fit(X_train, Y_train)
Y_pred = bag.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

# Function to predict injury based on input features
def predict_injury():
    # Extract input values from entry widgets
    nr_sessions = float(entry1.get())
    nr_rest_days = float(entry2.get())
    total_kms = float(entry3.get())
    total_km_Z3_Z4_Z5_T1_T2 = float(entry4.get())
    nr_tough_sessions = float(entry5.get())
    nr_days_with_interval_session = float(entry6.get())
    total_km_Z3_4 = float(entry7.get())
    max_km_Z3_4_one_day = float(entry8.get())
    total_km_Z5_T1_T2 = float(entry9.get())
    max_km_Z5_T1_T2_one_day = float(entry10.get())
    total_hours_alternative_training = float(entry11.get())
    nr_strength_trainings = float(entry12.get())
    nr_sessions_1 = float(entry13.get())
    nr_rest_days_1 = float(entry14.get())
    total_kms_1 = float(entry15.get())
    max_km_one_day_1 = float(entry16.get())
    total_km_Z3_Z4_Z5_T1_T2_1 = float(entry17.get())
    nr_tough_sessions_1 = float(entry18.get())
    nr_days_with_interval_session_1 = float(entry19.get())
    total_km_Z3_4_1 = float(entry20.get())
    max_km_Z3_4_one_day_1 = float(entry21.get())
    total_km_Z5_T1_T2_1 = float(entry22.get())
    max_km_Z5_T1_T2_one_day_1 = float(entry23.get())
    total_hours_alternative_training_1 = float(entry24.get())
    nr_strength_trainings_1 = float(entry25.get())
    nr_sessions_2 = float(entry26.get())
    nr_rest_days_2 = float(entry27.get())
    total_kms_2 = float(entry28.get())
    max_km_one_day_2 = float(entry29.get())
    total_km_Z3_Z4_Z5_T1_T2_2 = float(entry30.get())
    nr_tough_sessions_2 = float(entry31.get())
    nr_days_with_interval_session_2 = float(entry32.get())
    total_km_Z3_4_2 = float(entry33.get())
    max_km_Z3_4_one_day_2 = float(entry34.get())
    total_km_Z5_T1_T2_2 = float(entry35.get())
    max_km_Z5_T1_T2_one_day_2 = float(entry36.get())
    total_hours_alternative_training_2 = float(entry37.get())
    nr_strength_trainings_2 = float(entry38.get())
    athlete_ID = int(entry39.get())
    date = int(entry40.get())

    # Create a numpy array of input features
    input_features = np.array([[nr_sessions, nr_rest_days, total_kms, total_km_Z3_Z4_Z5_T1_T2,
                                nr_tough_sessions, nr_days_with_interval_session, total_km_Z3_4,
                                max_km_Z3_4_one_day, total_km_Z5_T1_T2, max_km_Z5_T1_T2_one_day,
                                total_hours_alternative_training, nr_strength_trainings, nr_sessions_1,
                                nr_rest_days_1, total_kms_1, max_km_one_day_1, total_km_Z3_Z4_Z5_T1_T2_1,
                                nr_tough_sessions_1,nr_days_with_interval_session_1, total_km_Z3_4_1,
                                max_km_Z3_4_one_day_1, total_km_Z5_T1_T2_1, max_km_Z5_T1_T2_one_day_1,
                                total_hours_alternative_training_1, nr_strength_trainings_1, nr_sessions_2,
                                nr_rest_days_2, total_kms_2, max_km_one_day_2, total_km_Z3_Z4_Z5_T1_T2_2,
                                nr_tough_sessions_2, nr_days_with_interval_session_2, total_km_Z3_4_2,
                                max_km_Z3_4_one_day_2, total_km_Z5_T1_T2_2, max_km_Z5_T1_T2_one_day_2,
                                total_hours_alternative_training_2, nr_strength_trainings_2, athlete_ID, date]])
    predicted_injury = bag.predict(input_features)
    # Predict injury
    if predicted_injury == 1:
        prediction_result = "Injury"
    else:
        prediction_result = "Injury"

    accuracy_percentage = accuracy * 100

    messagebox.showinfo("Prediction Result", f"{accuracy_percentage:.2f}% chances of {prediction_result}")


root = tk.Tk()
root.title("Injury Prediction System")
 # Change title to "Injury Prediction System"

# Load background image
background_image = Image.open("Background.jpg")  # Change "background_image.png" to the path of your image file
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

header_frame = tk.Frame(root, bg='white', bd=5)
header_frame.place(relx=0.5, rely=0.05, relwidth=0.9, relheight=0.1, anchor='n')

# Label for the header
header_label = tk.Label(header_frame, text="Athletic Runner Injury Prediction System", font=('Helvetica', 20))
header_label.pack(fill='both', expand=True)

frame1 = tk.Frame(root, bg='white', bd=5)
frame1.place(relx=0.05, rely=0.2, relwidth=0.2, relheight=0.5)

frame2 = tk.Frame(root, bg='white', bd=5)
frame2.place(relx=0.4, rely=0.2, relwidth=0.2, relheight=0.5)

frame3 = tk.Frame(root, bg='white', bd=5)
frame3.place(relx=0.75, rely=0.2, relwidth=0.2, relheight=0.5)


# Create entry widgets for input features
label1 = tk.Label(frame1, text="Number of Sessions:")
label1.grid(row=0, column=0, sticky="w")
entry1 = tk.Entry(frame1, width=5, justify="left")
entry1.grid(row=0, column=1)

label2 = tk.Label(frame1, text="Number of Rest Days:")
label2.grid(row=1, column=0, sticky="w")
entry2 = tk.Entry(frame1, width=5, justify="left")
entry2.grid(row=1, column=1)

label3 = tk.Label(frame1, text="Total Kilometers:")
label3.grid(row=2, column=0, sticky="w")
entry3 = tk.Entry(frame1, width=5, justify="left")
entry3.grid(row=2, column=1)

label4 = tk.Label(frame1, text="Total KM Z3-Z4-Z5-T1-T2:")
label4.grid(row=3, column=0, sticky="w")
entry4 = tk.Entry(frame1, width=5, justify="left")
entry4.grid(row=3, column=1)

label5 = tk.Label(frame1, text="Number of Tough Sessions:")
label5.grid(row=4, column=0, sticky="w")
entry5 = tk.Entry(frame1, width=5, justify="left")
entry5.grid(row=4, column=1)

label6 = tk.Label(frame1, text="No. of Days with Interval Session:")
label6.grid(row=5, column=0, sticky="w")
entry6 = tk.Entry(frame1, width=5, justify="left")
entry6.grid(row=5, column=1)

label7 = tk.Label(frame1, text="Total KM Z3-4:")
label7.grid(row=6, column=0, sticky="w")
entry7 = tk.Entry(frame1,width=5, justify="left")
entry7.grid(row=6, column=1)

label8 = tk.Label(frame1, text="Max KM Z3-4 One Day:")
label8.grid(row=7, column=0, sticky="w")
entry8 = tk.Entry(frame1, width=5, justify="left")
entry8.grid(row=7, column=1)

label9 = tk.Label(frame1, text="Total KM Z5-T1-T2:")
label9.grid(row=8, column=0, sticky="w")
entry9 = tk.Entry(frame1, width=5, justify="left")
entry9.grid(row=8, column=1)

label10 = tk.Label(frame1, text="Max KM Z5-T1-T2 One Day:")
label10.grid(row=9, column=0, sticky="w")
entry10 = tk.Entry(frame1, width=5, justify="left")
entry10.grid(row=9, column=1)

label11 = tk.Label(frame1, text="Total Hours Alternative Training:")
label11.grid(row=10, column=0, sticky="w")
entry11 = tk.Entry(frame1, width=5, justify="left")
entry11.grid(row=10, column=1)

label12 = tk.Label(frame1, text="Number of Strength Trainings:")
label12.grid(row=11, column=0, sticky="w")
entry12 = tk.Entry(frame1, width=5, justify="left")
entry12.grid(row=11, column=1)

label13 = tk.Label(frame2, text="Number of sessions.1:")
label13.grid(row=0, column=3, sticky="w")
entry13 = tk.Entry(frame2, width=5, justify="left")
entry13.grid(row=0, column=4)

label14 = tk.Label(frame2, text="Number of rest days.1:")
label14.grid(row=1, column=3, sticky="w")
entry14 = tk.Entry(frame2, width=5, justify="left")
entry14.grid(row=1, column=4)

label15 = tk.Label(frame2, text="total kms.1:")
label15.grid(row=2, column=3, sticky="w")
entry15 = tk.Entry(frame2, width=5, justify="left")
entry15.grid(row=2, column=4)

label16 = tk.Label(frame2, text="max km one day.1:")
label16.grid(row=3, column=3, sticky="w")
entry16 = tk.Entry(frame2, width=5, justify="left")
entry16.grid(row=3, column=4)

label17 = tk.Label(frame2, text="total km Z3-Z4-Z5-T1-T2.1:")
label17.grid(row=4, column=3, sticky="w")
entry17 = tk.Entry(frame2, width=5, justify="left")
entry17.grid(row=4, column=4)

label18 = tk.Label(frame2, text="Number of tough sessions.1:")
label18.grid(row=5, column=3, sticky="w")
entry18 = tk.Entry(frame2, width=5, justify="left")
entry18.grid(row=5, column=4)

label19 = tk.Label(frame2, text="No. of days with interval session.1:")
label19.grid(row=6, column=3, sticky="w")
entry19 = tk.Entry(frame2, width=5, justify="left")
entry19.grid(row=6, column=4)

label20 = tk.Label(frame2, text="total km Z3-4.1:")
label20.grid(row=7, column=3, sticky="w")
entry20 = tk.Entry(frame2, width=5, justify="left")
entry20.grid(row=7, column=4)

label21 = tk.Label(frame2, text="max km Z3-4 one day.1:")
label21.grid(row=8, column=3, sticky="w")
entry21 = tk.Entry(frame2, width=5, justify="left")
entry21.grid(row=8, column=4)

label22 = tk.Label(frame2, text="total km Z5-T1-T2.1:")
label22.grid(row=9, column=3, sticky="w")
entry22 = tk.Entry(frame2, width=5, justify="left")
entry22.grid(row=9, column=4)

label23 = tk.Label(frame2, text="max km Z5-T1-T2 one day.1:")
label23.grid(row=10, column=3, sticky="w")
entry23 = tk.Entry(frame2, width=5, justify="left")
entry23.grid(row=10, column=4)

label24 = tk.Label(frame2, text="total hours alternative training.1:")
label24.grid(row=11, column=3, sticky="w")
entry24 = tk.Entry(frame2, width=5, justify="left")
entry24.grid(row=11, column=4)

label25 = tk.Label(frame2, text="Number of Strength Trainings.1:")
label25.grid(row=12, column=3, sticky="w")
entry25 = tk.Entry(frame2, width=5, justify="left")
entry25.grid(row=12, column=4)

label26 = tk.Label(frame3, text="Number of sessions.2:")
label26.grid(row=0, column=5, sticky="w")
entry26 = tk.Entry(frame3, width=5, justify="left")
entry26.grid(row=0, column=6)

label27 = tk.Label(frame3, text="Number of rest days.2:")
label27.grid(row=1, column=5, sticky="w")
entry27 = tk.Entry(frame3,width=5, justify="left")
entry27.grid(row=1, column=6)

label28 = tk.Label(frame3, text="total kms.2:")
label28.grid(row=2, column=5, sticky="w")
entry28 = tk.Entry(frame3,width=5, justify="left")
entry28.grid(row=2, column=6)

label29 = tk.Label(frame3, text="max km one day.2:")
label29.grid(row=3, column=5, sticky="w")
entry29 = tk.Entry(frame3,width=5, justify="left")
entry29.grid(row=3, column=6)

label30 = tk.Label(frame3, text="total km Z3-Z4-Z5-T1-T2.2:")
label30.grid(row=4, column=5, sticky="w")
entry30 = tk.Entry(frame3,width=5, justify="left")
entry30.grid(row=4, column=6)

label31 = tk.Label(frame3, text="Number of tough sessions.2:")
label31.grid(row=5, column=5, sticky="w")
entry31 = tk.Entry(frame3,width=5, justify="left")
entry31.grid(row=5, column=6)

label32 = tk.Label(frame3, text="No. of days with interval session.2:")
label32.grid(row=6, column=5, sticky="w")
entry32 = tk.Entry(frame3,width=5, justify="left")
entry32.grid(row=6, column=6)

label33 = tk.Label(frame3, text="total km Z3-4.2:")
label33.grid(row=7, column=5, sticky="w")
entry33 = tk.Entry(frame3,width=5, justify="left")
entry33.grid(row=7, column=6)

label34 = tk.Label(frame3, text="max km Z3-4 one day.2:")
label34.grid(row=8, column=5, sticky="w")
entry34 = tk.Entry(frame3,width=5, justify="left")
entry34.grid(row=8, column=6)

label35 = tk.Label(frame3, text="total km Z5-T1-T2.2:")
label35.grid(row=9, column=5, sticky="w")
entry35 = tk.Entry(frame3,width=5, justify="left")
entry35.grid(row=9, column=6)

label36 = tk.Label(frame3, text="max km Z5-T1-T2 one day.2:")
label36.grid(row=10, column=5, sticky="w")
entry36 = tk.Entry(frame3,width=5, justify="left")
entry36.grid(row=10, column=6)

label37 = tk.Label(frame3, text="total hours alternative training.2:")
label37.grid(row=11, column=5, sticky="w")
entry37 = tk.Entry(frame3,width=5, justify="left")
entry37.grid(row=11, column=6)

label38 = tk.Label(frame3, text="Number of strength trainings.2:")
label38.grid(row=12, column=5, sticky="w")
entry38 = tk.Entry(frame3,width=5, justify="left")
entry38.grid(row=12, column=6)

label39 = tk.Label(frame3, text="Athlete ID:")
label39.grid(row=13, column=5, sticky="w")
entry39 = tk.Entry(frame3,width=5, justify="left")
entry39.grid(row=13, column=6)

label40 = tk.Label(frame3, text="Date:")
label40.grid(row=14, column=5, sticky="w")
entry40 = tk.Entry(frame3,width=5, justify="left")
entry40.grid(row=14, column=6)

# Create a button to trigger prediction
predict_button = tk.Button(root, text="Predict Injury", command=predict_injury, fg="Black", bg="Red")
predict_button.place(relx=0.5, rely=0.95, relwidth=0.3, relheight=0.1, anchor='s')


# Run the main event loop
root.mainloop()