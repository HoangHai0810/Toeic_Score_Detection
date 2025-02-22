# Toeic Score Detection

## Introduction

Toeic Score Detection is a project that utilizes Optical Character Recognition (OCR) and the YOLOv8 model to detect and recognize TOEIC scores from images. This project automates the score extraction process, minimizing errors and saving time compared to manual data entry.

## Features

- **Score Region Detection**: Uses YOLOv8 to identify the location of the TOEIC score on the score sheet.
- **Character Recognition**: Applies OCR to convert the detected score image into numerical text.
- **Web Interface**: Provides a user-friendly web interface for uploading images and viewing recognition results.

## Example
Image:
![BL1](https://github.com/user-attachments/assets/e1c966bc-dca4-4322-aba3-3cc389e7d15e)
Result:
![BL](https://github.com/user-attachments/assets/e150e163-4242-4b00-ab71-0cc010a11bef)

## Installation

1. **Clone the repository**:
   
   ```bash
   git clone https://github.com/HoangHai0810/Toeic_Score_Detection.git
   cd Toeic_Score_Detection
   ```

2. **Create and activate a virtual environment (optional but recommended)**:
   
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the trained YOLOv8 model**:
   - Download the `yolov8x.pt` file from the provided link and place it in the project root directory.

5. **Run the application**:
   
   ```bash
   python webOCR.py
   ```

   The application will run on `http://localhost:5000`. Open a web browser and navigate to this address to use it.

## Usage

1. Open a web browser and go to `http://localhost:5000`.
2. Upload an image of the TOEIC score sheet.
3. View the recognized score displayed on the interface.

## Contact
For any questions or suggestions, please contact:

Author: Anh Hoang-Hai Email: hoanghaianh0810@gmail.com

