# Colorization App

This Python application provides a graphical user interface (GUI) for colorizing black-and-white images and videos. It features multiple colorization methods including automatic colorization, dot-based colorization, and dynamic palette-based colorization using an external API to generate color palettes. The app utilizes OpenCV for image processing and Tkinter for the GUI.

## Features

- **Automatic Colorization**: Automatically colorizes black-and-white images and videos using a pre-trained deep learning model.
- **Dot-Based Colorization**: Allows users to manually place colored dots on the image, which influence the colorization of nearby areas.
- **Dynamic Palette-Based Colorization**: Users can generate color palettes dynamically via an API and apply these colors based on the grayscale intensity of the image.
- **Image and Video Support**: Supports both images and videos for colorization. Videos can be colorized only with automatic colorization.

## Prerequisites

Before running this application, you will need the following:
- Python 3.6 or newer
- OpenCV library
- NumPy
- PIL (Pillow)
- Tkinter
- Requests


  Install the required Python libraries by running: pip install numpy opencv-python pillow requests


Download the model from: https://drive.google.com/drive/folders/1Qh9GuAE7k1jxrQcofhuW3o8y8Gan35BP?usp=sharing

# Demo

- **Initialization**: Run `Main.py` to initiate the application.
-  Upload a file, choose a colorization mode and wait for the result.
