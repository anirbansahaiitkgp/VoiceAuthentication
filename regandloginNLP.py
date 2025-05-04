import os
import pyaudio
#import sounddevice as sd
import wave
import numpy as np
import time
import scipy.io.wavfile as wav
import pyttsx3
import torch
from time import sleep
from speechbrain.inference.speaker import SpeakerRecognition
import joblib
from googletrans import Translator


from flask import Flask, request
import warnings
import os
import numpy as np

# import scipy.io.wavfile as wav
from speechbrain.inference.speaker import SpeakerRecognition
import speech_recognition as sr
from googletrans import Translator
import joblib
import pandas as pd
import subprocess
import json
from pydub import AudioSegment
# import threading
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
import torch
import tensorflow as tf



#os.environ['ALSA_LOGLEVEL'] = 'none'


#app/Server/pretrained_models/spkrec-ecapa-voxceleb
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="app/Server/pretrained_models/spkrec-ecapa-voxceleb")


# Initialize the speech engine
engine = pyttsx3.init()


########################################################################
def commandClassify_S2(class_label_to_predict, text_result):
    # pipeline = joblib.load('Models/ClassSvm.joblib')
    command_to_classify = text_result
    data = {'Commands': [command_to_classify]}
    single_command_df = pd.DataFrame(data)
    # predicted_class = pipeline.predict(single_command_df['Commands'])
    # class_label_to_predict = predicted_class[0]
    if class_label_to_predict == 'Miscellaneous':
        predicted_subclass = ['Temporary']
    elif class_label_to_predict == 'Not needed':
        predicted_subclass = ['NC']
    else:
        subClassModel = joblib.load(f'app/Server/Models/SubClassModel_{class_label_to_predict}.joblib')
        predicted_subclass = subClassModel.predict(single_command_df['Commands'])
    return [class_label_to_predict, predicted_subclass[0]]
    
########################################################################
recognizer = sr.Recognizer()
# Anirban Models - 
# Load the model from file
#model = joblib.load('app/Server/anirbanModels/Main/RNN_Multiclass_Subclass.joblib')
###################################################################################
# === Load the TFLite models ===
interpreter_main = tf.lite.Interpreter(model_path='./app/Server/anirbanModels/Main/RNN_Multiclass_Subclass.tflite')
interpreter_main.allocate_tensors()
input_details_main = interpreter_main.get_input_details()
output_details_main = interpreter_main.get_output_details()

#===== Load the TFLite subclass category models ====================================
##Cloud
interpreter_C= tf.lite.Interpreter(model_path='./app/Server/anirbanModels/Cloud/RNN_Multiclass_SubclassCategory.tflite')
interpreter_C.allocate_tensors()
input_details_C = interpreter_C.get_input_details()
output_details_C = interpreter_C.get_output_details()

##Edge
interpreter_E= tf.lite.Interpreter(model_path='./app/Server/anirbanModels/Edge/RNN_Multiclass_SubclassCategory.tflite')
interpreter_E.allocate_tensors()
input_details_E = interpreter_E.get_input_details()
output_details_E = interpreter_E.get_output_details()

##Update
interpreter_U= tf.lite.Interpreter(model_path='./app/Server/anirbanModels/Update/RNN_Multiclass_SubclassCategory.tflite')
interpreter_U.allocate_tensors()
input_details_U = interpreter_U.get_input_details()
output_details_U = interpreter_U.get_output_details()
####################################################################################

# === Preprocess a text input ===
def preprocess_input(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_seq_len, padding='post')
    return np.array(padded, dtype=np.float32)  # FIX: convert to float32

 
# === Run inference ===
def predict(text):
    input_data = preprocess_input(text)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data
  
# === Class label mapping ===
class_labels = {
    0: 'Edge',
    1: 'Cloud',
    2: 'Update'
}

###################################################################################
tokenizer = joblib.load('app/Server/anirbanModels/Main/tokenizer.joblib')
max_sequence_length=joblib.load('app/Server/anirbanModels/Main/max_sequence_length_mainModel.joblib')

#############################################################################################
# Load the model from and tokenizer
#model_E = joblib.load('app/Server/anirbanModels/Edge/RNN_Multiclass_SubclassCategory.joblib')
tokenizer_E = joblib.load('app/Server/anirbanModels/Edge/tokenizer.joblib')
max_sequence_length_E=joblib.load('app/Server/anirbanModels/Edge/max_sequence_length.joblib')

# Load the model from and tokenizer
#model_C = joblib.load('app/Server/anirbanModels/Cloud/RNN_Multiclass_SubclassCategory.joblib')
tokenizer_C = joblib.load('app/Server/anirbanModels/Cloud/tokenizer.joblib')
max_sequence_length_C=joblib.load('app/Server/anirbanModels/Cloud/max_sequence_length.joblib')

# Load the model from and tokenizer
#model_U = joblib.load('app/Server/anirbanModels/Update/RNN_Multiclass_SubclassCategory.joblib')
tokenizer_U = joblib.load('app/Server/anirbanModels/Update/tokenizer.joblib')
max_sequence_length_U=joblib.load('app/Server/anirbanModels/Update/max_sequence_length.joblib')



# Function to convert speech to text
def speech_to_text(audio_file_path):
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source) 
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")


def predict_class2(command,tokenizer,model,max_sequence_length):
    sequence = tokenizer.texts_to_sequences([command])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
    prediction = model.predict(padded_sequence)
    predicted_class_index = np.argmax(prediction)
    return predicted_class_index


def predict_class(command, tokenizer, interpreter, max_seq_len, input_details):
    # Preprocess
    sequence = tokenizer.texts_to_sequences([command])
    padded = pad_sequences(sequence, maxlen=max_seq_len, padding='post')
    input_data = np.array(padded, dtype=np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details_main[0]['index'], input_data)
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details_main[0]['index'])
    prediction = np.argmax(output_data, axis=-1)[0]
    return prediction
#########
'''
def predict_class(command, tokenizer, model_or_interpreter, max_seq_len):
    sequence = tokenizer.texts_to_sequences([command])
    padded = pad_sequences(sequence, maxlen=max_seq_len, padding='post')
    
    input_data = np.array(padded, dtype=np.float32)

    # TFLite path
    if isinstance(model_or_interpreter, tf.lite.Interpreter):
        input_details = model_or_interpreter.get_input_details()
        output_details = model_or_interpreter.get_output_details()

        model_or_interpreter.set_tensor(input_details[0]['index'], input_data)
        model_or_interpreter.invoke()
        output_data = model_or_interpreter.get_tensor(output_details[0]['index'])
        return np.argmax(output_data, axis=-1)[0]
    
    # Keras path
    else:
        prediction = model_or_interpreter.predict(input_data)
        return np.argmax(prediction, axis=-1)[0]
'''
#########


def Edge_Model(subclasscat,command,interpreter,tokenizer,max_sequence_length, input_details):
    #prediction = predict_class2(command, tokenizer,model, max_sequence_length)
    prediction = predict_class(command, tokenizer,interpreter, max_sequence_length, input_details)
    if(prediction==0):
        print("Edge -> Battery Fuel:", prediction)
        return "Battery Fuel"
    elif(prediction==1):
        print("Edge -> Tires:", prediction)
        return "Tires"
    elif(prediction==2):
        print("Edge -> Basic :",prediction)
        return "Basic"
    else:
        print("Dont know")
        return "-"

def Cloud_Model(subclasscat,command,interpreter,tokenizer,max_sequence_length, input_details):
    print(command)
    #prediction = predict_class2(command, tokenizer,model, max_sequence_length)
    prediction = predict_class(command, tokenizer,interpreter, max_sequence_length, input_details)
    if(prediction==0):
        print("Cloud -> Song Media:", prediction)
        return "Song Media"
    elif(prediction==1):
        print("Cloud -> News Notification:", prediction)
        return "News Notification"
    elif(prediction==2):
        print("Cloud -> Weather :",prediction)
        return "Weather"
    elif(prediction==3):
        print("Cloud -> Traffic Maps :",prediction)
        return "Traffic Maps"
    else:
        print("Dont know")
        return "-"

def Update_Model(subclasscat,command,interpreter,tokenizer,max_sequence_length, input_details): 
    print(command)
    #prediction = predict_class2(command, tokenizer,model, max_sequence_length)
    prediction = predict_class(command, tokenizer,interpreter, max_sequence_length, input_details)
    if(prediction==0):
        print("Update -> Cancel:", prediction)
        return "Cancel"
    elif(prediction==1):
        print("Update -> Perform:", prediction)
        return "Perform"
    elif(prediction==2):
        print("Update -> Check:",prediction)
        return "Check"
    else:
        print("Dont know")
        return "-"

def Main_model(subclasscat,command, input_details_main):  
    prediction = predict_class(command, tokenizer,interpreter_main, max_sequence_length, input_details_main)
    if(prediction==0):
        print("Edge :", prediction)
        subclasscat="EDGE"
        subClass_cat = Edge_Model(subclasscat,command,interpreter_E,tokenizer_E,max_sequence_length_E,input_details_E)
        return ["Edge", subClass_cat]
    elif(prediction==1):
        print("Cloud :", prediction)
        subclasscat="CLOUD"
        #subClass_cat = Cloud_Model(subclasscat,command,model_C,tokenizer_C,max_sequence_length_C)
        subClass_cat = Cloud_Model(subclasscat,command,interpreter_C,tokenizer_C,max_sequence_length_C,input_details_C)
        return ["Cloud", subClass_cat]
    elif(prediction==2):
        print("Update :",prediction)
        subclasscat="UPDATE"
        subClass_cat = Update_Model(subclasscat,command,interpreter_U,tokenizer_U,max_sequence_length_U,input_details_U)
        return ["Update", subClass_cat]
    elif(prediction==3):
        print("Miscellaneous :",prediction)
        return ["Miscellaneous", "Temporary"]
    else:
        print("Dont know")
        return ["-","-"]
########################################################################
def commandClassifier(command):
    start_time_lstm = time.time()
    subclasscat="MAINMODEL"
    opLst = Main_model(subclasscat,command,input_details_main)
    end_time_lstm = time.time() 
    execution_time_lstm = end_time_lstm - start_time_lstm  # Calculate the duration
    

    #start_time_ml = time.time()
    subclass_cat = commandClassify_S2(opLst[0], command)
    #end_time_ml = time.time() 
    execution_time_ml=0
    #execution_time_ml = end_time_ml - start_time_ml  # Calculate the duration
    #data = {
    #            "status": opLst[0],
    #            "user": subclass_cat[1]
    #    }
    
    #print(f"Anirban Only - {opLst}")
    print(f"Model 1 -{opLst}")
    print(f"Model 2 -{subclass_cat}")
    print (f"Time taken for LSTM Execution (SubClass) :{execution_time_lstm} seconds")
    #speak(f"The command classified as subclass {opLst[0]} and subclass category as {subclass_cat[1]} .") 
    return [opLst[0],opLst[1],execution_time_lstm,execution_time_ml]
    
    
def speak(text):
    print(text)
    engine.say(text)
    engine.runAndWait()
def record_audio(file_path, duration=10, audioName="None"):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 48000
    RECORD_SECONDS = duration
    WAVE_OUTPUT_FILENAME = file_path
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
   
       
    
    if(audioName=="HeyTvs"):
        speak("Please say the wake word 'Hey TVS'")
        sleep(2)
    print("Recording...")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()



def register():
    # Base paths
    base_path = "./app/Server/dynamic-content"
    users_file_path = os.path.join(base_path, "users.txt")
    names_file_path = os.path.join(base_path, "names.txt")
    standard_path = os.path.join(base_path, "final_standard")
    predict_temp_path = os.path.join(base_path, "predictTemp")

    # Wake word paths
    wake_path = os.path.join(base_path, "heyTVS")
    wake_file_path = os.path.join(base_path, "heyTVS.txt")

    # Command paths
    command_folder_path = os.path.join(base_path, "commands")
    command_text_file = os.path.join(base_path, "commands.txt")

    # Threshold log path
    threshold_file_path = os.path.join(base_path, "dynamicThreshold.txt")

    # Get username
    speak("Please say your name")
    username = input("Enter username: ").strip()

    if not username:
        speak("Username cannot be empty")
        return

    # Check if name already exists
    if os.path.exists(names_file_path):
        with open(names_file_path, "r") as file:
            existing_names = [line.strip().lower() for line in file]
        if username.lower() in existing_names:
            speak("Username already exists. Please try a different name.")
            return

    # File paths
    user_audio_path = os.path.join(standard_path, f"{username}.wav")
    user_audio_predict_path = os.path.join(predict_temp_path, f"{username}_predict.wav")
    wake_audio_filename = f"{username}_heyTVS.wav"
    wake_audio_path = os.path.join(wake_path, wake_audio_filename)
    command_audio_filename = f"{username}_command.wav"
    command_audio_path = os.path.join(command_folder_path, command_audio_filename)
    relative_wake_path = f"dynamic-content/heyTVS/{wake_audio_filename}"
    relative_command_path = f"app/Server/dynamic-content/commands/{command_audio_filename}"

    # Record reference and verification audio
    speak("Recording reference audio")
    sleep(1)
    record_audio(user_audio_path)

    speak("Recording verification audio")
    sleep(1)
    record_audio(user_audio_predict_path)

    try:
        score, _ = verification.verify_files(user_audio_path, user_audio_predict_path)
        score = score.item()
        print(f"Verification score: {score:.4f}")

        if score >= 0.7:
            # Save reference and name info
            with open(users_file_path, "a") as ufile:
                ufile.write(f"{username}: {user_audio_path}\n")
            with open(names_file_path, "a") as nfile:
                nfile.write(f"{username}\n")

            # Record wake word
            record_audio(wake_audio_path, duration=5, audioName="HeyTvs")
            with open(wake_file_path, "a") as wfile:
                wfile.write(f"{username}: {relative_wake_path}\n")

            # Record and verify standard command
            command_attempts = 0
            max_attempts = 8
            command_verified = False

            while command_attempts < max_attempts and not command_verified:
                speak("Please say the standard command: 'Current weather in goa ?'")
                sleep(1)
                record_audio(command_audio_path, duration=5)

                try:
                    cmd_score, _ = verification.verify_files(user_audio_path, command_audio_path)
                    cmd_score = cmd_score.item()
                    print(f"Command match score: {cmd_score:.4f}")

                    if 0.475 <= cmd_score <= 0.6:
                        with open(command_text_file, "a") as cfile:
                            cfile.write(f"{username}: {relative_command_path}\n")

                        # Save threshold score
                        with open(threshold_file_path, "a") as tfile:
                            tfile.write(f"{username}: tensor([{cmd_score:.4f}])\n")

                        command_verified = True
                        speak(f"Registration successful for {username}")
                        break
                    else:
                        speak("Command does not match. Please try again.")
                        command_attempts += 1

                except Exception as e:
                    speak(f"Error verifying standard command: {str(e)}")
                    command_attempts += 1

            if not command_verified:
                speak("Registration failed due to repeated command mismatch. Cleaning up.")
                cleanup_user_data(username, [
                    user_audio_path,
                    user_audio_predict_path,
                    wake_audio_path,
                    command_audio_path
                ], [
                    users_file_path,
                    names_file_path,
                    wake_file_path,
                    command_text_file,
                    threshold_file_path
                ])

        else:
            speak("Voice match failed. Registration unsuccessful.")
            if os.path.exists(user_audio_path):
                os.remove(user_audio_path)
            if os.path.exists(user_audio_predict_path):
                os.remove(user_audio_predict_path)

    except Exception as e:
        speak(f"Error during registration: {str(e)}")
        if os.path.exists(user_audio_path):
            os.remove(user_audio_path)
        if os.path.exists(user_audio_predict_path):
            os.remove(user_audio_predict_path)

# Helper: Remove user entries and files
def cleanup_user_data(username, file_paths, text_files):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)

    def remove_line(file, key):
        if os.path.exists(file):
            with open(file, "r") as f:
                lines = f.readlines()
            with open(file, "w") as f:
                for line in lines:
                    if not line.lower().startswith(key.lower()):
                        f.write(line)

    for txt_file in text_files:
        remove_line(txt_file, username)


def login():
    # Path to the user reference list
    user_file_path = "./app/Server/dynamic-content/users.txt"
   
    # Record login attempt
    speak("Recording login audio")
    sleep(1)
    record_audio("./app/Server/dynamic-content/unknownVoices/unknown.wav")

    # Dictionary to hold scores
    user_scores = {}

    # Read user reference data
    if not os.path.exists(user_file_path):
        speak("User reference file not found.")
        return

    with open(user_file_path, "r") as file:
        for line in file:
            if ':' in line:
                name, ref_path = line.strip().split(':', 1)
                name = name.strip()
                ref_path = ref_path.strip()
                print(ref_path)
               
                if os.path.exists(ref_path):
                    try:
                        score, _ = verification.verify_files(ref_path, "./app/Server/dynamic-content/unknownVoices/unknown.wav")
                       
                        # Convert tensor([value]) to float
                        score = score.item()

                        user_scores[name] = score
                        print(f"Score for {name}: {score:.4f}")
                    except Exception as e:
                        print(f"Error processing {name}: {e}")
                else:
                    print(f"Reference audio not found for {name}: {ref_path}")

    # Determine the best match
    if user_scores:
        best_match = max(user_scores, key=user_scores.get)
        max_score = user_scores[best_match]

        if max_score >= 0.6:
            speak(f"Welcome {best_match} to OTA Voice Assistant")
            CommandClassifyNLP()
            
        else:
            speak("Unauthorized user")
    else:
        speak("No valid users found for verification")

def CommandClassifyNLP():

    
    while True:
        speak(f"Waiting for your command..")
        sleep(1)
        record_audio("./app/Server/dynamic-content/inputCmd/inputCmd.wav",5, "CommandNLP")
        translator = Translator()
        start_time_ts = time.time()
        text_result = speech_to_text("./app/Server/dynamic-content/inputCmd/inputCmd.wav")
        print("Text result:", text_result)
        end_time_ts = time.time() 
        execution_time_ts = end_time_ts - start_time_ts  # Calculate the duration
        sleep(1)
        if text_result == None:
            print("No commands detected. Re-enter command")
            speak("No commands detected. Re-enter command")
            sleep(2)
            
        elif text_result =="logout":
            print("logging out")
            speak(f"Please wait... Logging out. Logout successfully.")
            sleep(3)
            
            break;
        else:
            print(text_result)
            start_time_nlp = time.time()
            op1 = commandClassifier(text_result)
            end_time_nlp = time.time() 
            execution_time_nlp = end_time_nlp - start_time_nlp  # Calculate the duration
            speak(f"The command classified as subclass {op1[0]} and subclass category as {op1[1]} .") 
            
    
    return

def main():
    print("Select an option:")
    print("1. Register")
    print("2. Login")
    print("3. Exit")
    
    choice = input("Enter your choice (1/2/3): ")

    switch = {
        "1": register,
        "2": login,
        "3": lambda: exit(0)
    }

    action = switch.get(choice)
    
    if action:
        action()
    else:
        print("Invalid choice. Please select 1 or 2.")

if __name__ == "__main__":
    while(1):
        main()
