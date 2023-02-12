from tkinter import *
import numpy as np
import os
import shutil
import csv
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
from pathlib import Path
import cv2

root = Tk()
root.geometry("620x485")
root.title("College Attendance System")
root.minsize(900, 640)
root.maxsize(900, 640)

message = Label(root, text="Intelligent-Attendence-System", bg="green", fg="white", width=40, height=3,
                font=('times', 30, 'bold'))
message.pack(fill=X, pady=15)


# The function beow is used for checking
# whether the text below is number or not ?
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def TakeImages():
    # Both ID and Name is used for recognising the Image
    Id = (roll_entry.get())
    name = (name_entry.get())

    # Checking if the ID is numeric and name is Alphabetical
    if is_number(Id) and name.isalpha():
        # Opening the primary camera if you want to access
        # the secondary camera you can mention the number
        # as 1 inside the parenthesis
        cam = cv2.VideoCapture(0)
        # Specifying the path to haarcascade file
        harcascadePath = "data\haarcascade_frontalface_default.xml"
        # Creating the classier based on the haarcascade file.
        detector = cv2.CascadeClassifier(harcascadePath)
        # Initializing the sample number(No. of images) as 0
        sampleNum = 0
        while True:
            # Reading the video captures by camera frame by frame
            ret, img = cam.read()
            # Converting the image into grayscale as most of
            # the the processing is done in gray scale format
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # It converts the images in different sizes
            # (decreases by 1.3 times) and 5 specifies the
            # number of times scaling happens
            faces = detector.detectMultiScale(gray, 1.3, 5)

            # For creating a rectangle around the image
            for (x, y, w, h) in faces:
                # Specifying the coordinates of the image as well
                # as color and thickness of the rectangle.
                # incrementing sample number for each image
                cv2.rectangle(img, (x, y), (
                    x + w, y + h), (255, 0, 0), 2)
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder
                # TrainingImage as the image needs to be trained
                # are saved in this folder
                cv2.imwrite(
                    "TrainingImage\ " + name + "." + Id + '.' + str(
                        sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                # display the frame that has been captured
                # and drawn rectangle around it.
                cv2.imshow('frame', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is more than 60
            elif sampleNum > 60:
                break
        # releasing the resources
        cam.release()
        # closing all the windows
        cv2.destroyAllWindows()
        # Displaying message for the user
        res = "Images Saved for ID : " + Id + " Name : " + name
        # Creating the entry for the user in a csv file
        row = [Id, name]
        with open('UserDetails\\UserDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            # Entry of the row in csv file
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if is_number(Id):
            res = "Enter Alphabetical Name"
            message.configure(text=res)
        if name.isalpha():
            res = "Enter Numeric Id"
            message.configure(text=res)


def getRollAndName():
    global roll_entry, name_entry
    window = Tk()
    window.geometry("620x485")
    window.title("Collecting Sample Images")
    window.minsize(700, 500)
    window.maxsize(800, 400)
    msg = Label(window, text="Collecting Sample Images", bg="green", fg="white", width=40, height=2,
                font=('times', 30, 'bold'))
    msg.pack(fill=X, pady=10)

    roll = Label(window, text="Roll No.", width=13, height=2, fg="white", bg="green", font=('times', 20, ' bold '))
    roll.place(x=10, y=150)
    roll_entry = Entry(window, width=25, bg="white", fg="black", font=('times', 20, ' bold '))
    roll_entry.place(x=280, y=170)

    name = Label(window, text="Name", width=13, fg="white", bg="green", height=2, font=('times', 20, ' bold '))
    name.place(x=10, y=240)
    name_entry = Entry(window, width=25, bg="white", fg="black", font=('times', 20, ' bold '))
    name_entry.place(x=280, y=260)

    Button(window, text="Exit", fg="white", bg="green", font=('times', 20, ' bold '), width=15, height=2,
           command=window.destroy, activebackground="Red").place(x=385, y=350)
    Button(window, text="Start", fg="white", bg="green", font=('times', 20, ' bold '), width=15, height=2,
           command=TakeImages, activebackground="Red").place(x=20, y=350)

    window.mainloop()


def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # creating detector for faces
    detector = cv2.CascadeClassifier("data\haarcascade_frontalface_default.xml")
    # Saving the detected faces in variables
    faces, Id = getImagesAndLabels("TrainingImage")
    # Saving the trained faces and their respective ID's
    # in a model named as "trainner.yml".
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    # Displaying the message
    res = "Image Trained"
    message.configure(text=res)


def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    # creating empty ID list
    Ids = []
    # now looping through all the image paths and loading the
    # Ids and the images saved in the folder
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


faceCascade = cv2.CascadeClassifier("data\haarcascade_frontalface_default.xml")


def face_dector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        image, face = face_dector(frame)
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = recognizer.predict(face)
            if result[1] < 500:
                confidence = int(100 * (1 - (result[1]) / 300))
                display_string = str(confidence) + '% confidence it is user'
            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)
            if confidence > 75:
                cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', image)

            else:
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('Face Cropper', image)

        except:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)
            pass

        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()


sample_collect = Button(root, text="Collect Sample Images", fg="white", bg="green", font=('times', 20, ' bold '),
                        width=20, height=2, activebackground="Red", command=getRollAndName).place(x=280, y=200)
training_model = Button(root, text="Training Model", fg="white", bg="green", font=('times', 20, ' bold '), width=20,
                        height=2, activebackground="Red", command=TrainImages).place(x=280, y=300)
Attendance = Button(root, text="Attendance", fg="white", bg="green", font=('times', 20, ' bold '), width=20, height=2,
                    activebackground="Red", command=TrackImages).place(x=280, y=400)
Quit = Button(root, text="Quit", fg="white", bg="green", font=('times', 20, ' bold '), width=20, height=2,
              command=root.destroy, activebackground="Red").place(x=280, y=500)
root.mainloop()
