# Diabetic Retinopathy Detection

This project focuses on building a deep learning model to detect diabetic retinopathy (DR) using retinal fundus images. The goal is to classify images into five categories indicating the severity of DR: No_DR, Mild, Moderate, Severe, and Proliferate_DR. Early detection of DR is crucial in preventing vision impairment, and deep learning can offer preliminary diagnostic support.

The dataset was organized manually from image folders, and a DataFrame was created linking image paths to their corresponding labels. The data was split into training, validation, and test sets using stratified sampling to preserve class distribution. Extensive data augmentation, including rescaling, shearing, and vertical flipping, was applied to the training set to improve generalization. Only rescaling was applied to validation and test sets.

The model was constructed using Keras and TensorFlow, with a custom ResNet-inspired architecture built from scratch. The architecture included an initial Conv2D block followed by three stacked residual blocks and average pooling. The final output layer was a softmax activation with five units for multiclass classification. The model was compiled with the Adam optimizer and categorical cross-entropy loss.

Training was conducted with early stopping and model checkpointing to prevent overfitting and to preserve the best-performing model. Accuracy and loss trends were monitored over epochs. While training loss steadily decreased, the validation loss was more erratic, suggesting that the validation set might be under-representative or noisy.

Performance was evaluated on the test set using accuracy, confusion matrix, and classification report. The model performed well for the No_DR class and reasonably for the Moderate and Mild classes. However, it struggled with distinguishing Severe and Proliferate_DR cases, which are likely underrepresented and visually similar to adjacent classes. This model can serve as a useful first screening tool but is not intended for final clinical diagnosis.

