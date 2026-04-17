# MindType Mobile

Mental State Detection Using Keystroke Dynamics — Android Application

## Project Overview
MindType Mobile is an Android application that passively monitors a user's mental stress state by analyzing how they type on their smartphone — specifically the timing patterns of keystrokes. It operates as a custom Android keyboard (Input Method Editor) and runs silently in the background during normal phone usage.

## Features
- **Privacy First**: No typed text content is ever recorded or stored. Only timing metadata is captured.
- **Mental State Detection**: Features are extracted (dwell time, flight time, typing speed, touch pressure) and fed into an on-device TensorFlow Lite model.
- **Real-Time Classification**: Classifies mental states as Calm, Mild Stress, or High Stress.
- **Dataset Collection**: Built for research contexts to collect keystroke features mapped with self-reported stress scores.

## Setup Requirements
- Android API Level 26+ (Android 8.0 Oreo and above)
- Android Studio for development
- Python & TensorFlow Lite for ML pipeline

## Target Accuracy Objectives
The target metric for the ML pipeline is a weighted F1-Score of 85-90%. We cap the target accuracy to actively avoid model overfitting, ensuring the mobile models generalize better in real production environments.

## Research Context
This application is part of an academic research project at VIT-AP University, Department of Networking and Security, under the supervision of Dr. Udit Narayana Kar.
