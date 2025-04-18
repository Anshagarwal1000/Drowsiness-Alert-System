﻿# 😴 Drowsiness Detection System using OpenCV, Dlib & Python

A real-time drowsiness detection system that uses computer vision to monitor eye aspect ratio and alert the user with an alarm when signs of drowsiness or sleep are detected.

##  Features

- Detects face and eyes using dlib’s 68 facial landmarks.
- Calculates eye aspect ratio to determine eye closure.
- Differentiates between **Active**, **Drowsy**, and **Sleeping** states.
- Plays an alarm sound when drowsiness or sleep is detected.
- Real-time monitoring via webcam.


##  Technologies Used

- Python
- OpenCV
- Dlib
- Imutils
- Playsound
- NumPy
- Threading

##  How It Works

The system calculates the eye aspect ratio (EAR) using specific facial landmarks. Based on the EAR value:

- **Active (EAR > 0.25)** – Eyes are open.
- **Drowsy (0.21 < EAR <= 0.25)** – Eyes partially closed.
- **Sleeping (EAR ≤ 0.21)** – Eyes are closed.

If the drowsiness/sleep condition persists for more than a few ( 6 ) frames, an alarm is triggered using a sound file.
