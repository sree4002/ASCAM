# ASCAM

**Axonal Swelling Classification and Analysis Model (ASCAM)**  
Automated detection and quantification of axonal swellings in histological brain sections using deep learning.

[![Language: Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)  
![Platform](https://img.shields.io/badge/platform-Google%20Colab-lightgrey)  
![Model Type](https://img.shields.io/badge/model-YOLOv5%20%2B%20CNN-orange)

---

## 🧠 Overview

ASCAM is a two-part deep learning pipeline built for the **automated detection and quantification of axonal swellings**, a pathological marker found in neurodegenerative diseases like Alzheimer's. It consists of:

1. **Image Classification Model** – Filters images as swelling-positive or swelling-negative.
2. **Object Detection Model** – Detects and quantifies axonal swellings using YOLO, optimized for small objects.

ASCAM significantly reduces analysis time and variability, offering a scalable alternative to manual quantification in histology workflows.

---

## 📁 Repository Structure

| Folder                  | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `Classification_Model`  | CNN-based binary image classifier to detect presence of swellings.          |
| `Object_Detection_Model`| YOLOv5-based model for bounding-box localization of swellings.              |
| `Running_Code`          | Scripts for inference, prediction, and result visualization.                |
| `test_images`           | Sample test images from 5xFAD mouse pons brain sections (DAB stained).       |

---
## Usage

ASCAM is composed of three components: the Classification Model, Object Detection Model, and Full Pipeline. Each can be run independently or as an integrated workflow.

### Classification Model

Download all files in the `Classification_Model/` folder and run the following command locally:

```bash
# Run the classification model
python classify_images.py --input_dir test_images
```
Classification results will be printed to the terminal.

### Object Detection Model

Download all files in the `Object_Detection_Model/` folder and run the following command locally:

```bash
# Run the object detection model
python detect_swellings.py --weights weights.pt --source test_images --output results/
```
Output images with bounding boxes will be saved to the results/ folder.

### Full Pipeline (Running Code)
1. Download the `Running_Code/` folder
2. Add your test images to running_code_images folder
3. Run code locally
4. Classifcation model results will be in terminal and Object Detection model results will be in running_code_results folder.

## Contributing

Thank you for wanting to improve ASCAM! If you encounter any bugs, have questions, or would like to contribute new features, feel free to open an issue or submit a pull request.

