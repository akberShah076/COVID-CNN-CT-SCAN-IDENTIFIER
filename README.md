# Transfer Learning for COVID-19 Diagnosis using CT Images

## Project Overview
This project involves building a 2D Convolutional Neural Network (CNN) for diagnosing COVID-19 using CT images. The goal is to classify images into two categories: infected vs. non-infected. 

The project is divided into several tasks, including training the CNN from scratch, implementing transfer learning, visualizing the models using CAM methods, and comparing the results.

---

## Tasks and Objectives

### Task 1: Model Construction and Training
1. **Constructing a CNN:**
   - Modify a ResNet-18 or ResNet-50 model for binary classification.
   
2. **Training from Scratch:**
   - Train the CNN with random initialization (no pre-trained weights) on the training set.
   - Evaluate the trained model on the test set.

3. **Transfer Learning:**
   - Train the CNN using pre-trained weights.
   - Evaluate the model on the test set.

4. **Visualization using CAM:**
   - Use two CAM methods (e.g., GradCAM and EigenCAM) to visualize results.
   - Randomly select 10 COVID-positive and 10 non-COVID images from the test set for visualization.
   
5. **Discussion:**
   - Write a discussion comparing the benefits of transfer learning vs. training from scratch.

---

## Implementation Details

### Requirements
- **Frameworks and Tools:**
  - PyTorch
  - Grad-CAM (or similar visualization libraries)
- **Loss Functions:**
  - Use `BCEWithLogitsLoss` for binary classification and `BinaryClassifierOutputTarget` for CAM visualization.
  - Alternatively, use `CrossEntropyLoss` and `ClassifierOutputTarget`.
- **DataLoader:**
  - Training, validation, and test sets are provided.

### Steps to Run the Code
1. **Install Dependencies:**
   Install all required packages, including PyTorch and Grad-CAM libraries. For example:
   ```bash
   pip install torch torchvision pytorch-grad-cam
   ```

2. **Train the Model from Scratch:**
   - Modify the ResNet-18 or ResNet-50 architecture for binary classification.
   - Train the model using random weights and evaluate it.

3. **Train the Model Using Transfer Learning:**
   - Load pre-trained weights for the ResNet model.
   - Train the model and evaluate its performance.

4. **Visualize Results Using CAM:**
   - Use GradCAM or EigenCAM to visualize results for both models.
   - Ensure correct class labels (`category`) are used for visualization.

5. **Compare Results:**
   - Discuss the benefits and drawbacks of transfer learning compared to training from scratch.

---

## References
- [PyTorch Grad-CAM Repository](https://github.com/jacobgil/pytorch-grad-cam)
- [Keras Grad-CAM Example](https://github.com/jacobgil/keras-grad-cam)
- [Kaggle Notebook: Dog-Cat Classifier with Grad-CAM](https://www.kaggle.com/code/nguyenhoa/dog-cat-classifier-gradcam-with-tensorflow-2-0/notebook)

---

## Notes
- Ensure proper preprocessing of the CT images before training.
- Always verify the correctness of the `category` label in CAM visualizations to avoid inaccuracies.
