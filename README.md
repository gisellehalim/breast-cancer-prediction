# Machine Learning Project Report - Giselle Halim

## Project Domain

Breast cancer is a cancer that occurs in the breast. It occurs when breast cells grow uncontrollably and take over healthy and surrounding breast tissue.

Breast cancer is more common in women, but can also affect men. The most common symptom of breast cancer is a lump in the breast or armpit. Other symptoms include breast skin changes, pain, inward retraction of the nipple, and unusual discharge from the nipple.

The cause of breast cancer is unknown, but it is thought to be caused by factors such as age, family history, genetic factors, and lifestyle factors. Treatment of breast cancer depends on the stage of cancer, age, and health condition of the patient. Breast cancer treatment may include surgery, chemotherapy, radiation therapy, and hormone therapy.

There are many imaging tests that can be used to diagnose breast cancer, such as mammography, ultrasound, MRI, and PET-CT scan. The results of these imaging tests will be measured and scrutinized from various assessments to help diagnose breast cancer. Some of these assessments include:

* Tumor size and shape: Breast cancer tumors are usually larger and more irregular than benign tumors.
* Tumor density: Breast cancer tumors are usually denser than normal breast tissue.

* Tumor texture: Breast cancer tumors usually have an uneven and rough texture.

* Blood flow to the tumor: Breast cancer tumors usually have higher blood flow than normal breast tissue.
* Tumor spread: Breast cancer tumors may spread to other breast tissues or to lymph nodes in the armpit.

Based on data from GLOBOCAN 2022, breast cancer is the number 1 cancer suffered by Indonesian women.A total of 66,271 new cases of breast cancer occurred in Indonesia in 2022. In addition, breast cancer also has the highest cases in Indonesia when calculated from the overall cancer cases that occur in all genders."[1]

![GLOBOCAN 2022](https://i.ibb.co/HCwKptQ/GLOBOCAN-2022.png)

In the context of health/healthcare, machine learning can analyze large amounts of patient data, identify certain patterns, and extract valuable insights that can aid accurate medical diagnosis. Machine learning technology can also help improve the consistency and accuracy of diagnosis by minimizing the incidence of human error. Given this, machine learning is a suitable method to help diagnose breast cancer based on patient imaging results. [2]

Especially for breast cancer, machine learning can help the diagnosis process through imaging images (in this case dataset, images that have been processed by computers to obtain size numbers) by recognizing early symptoms in images that are difficult for humans to recognize. In addition, machine learning is able to recognize complex patterns in data that may not be detected by conventional methods such as genetic, molecular, or histological patterns that can provide deeper insights into breast cancer. In conditions where there is no relevant specialist, machine learning can also help diagnosis based on previous data. machine learning can also be a predictive analysis tool to determine a person's risk of developing breast cancer based on historical data. [3]

## Business Understanding

As mentioned earlier, machine learning can help improve the consistency and accuracy of diagnosis by minimizing the incidence of human error and finding hidden patterns in clinical data. Therefore, this project will discuss the use of machine learning to help diagnose breast cancer based on imaging image data that has been measured and calculated by a computer into numerical data.

### Problem Statements

Based on the background, the problem will be discussed as follows:
- What are the patterns and trends in the breast cancer dataset?

- How can machine learning models be created to predict breast cancer diagnosis with high accuracy?

- Which machine learning model has the best metric measurement to predict breast cancer diagnosis?

### Goals

Referring to the problem that has been described, the objectives to be achieved in this project are as follows:

- Perform exploratory data analysis (EDA) on the dataset to see patterns and trends in the data.

- Creating a machine learning model that can predict breast cancer diagnosis.

- Comparing the Gradient Boosting, Random Forest, and Stacking Classifier methods to see which model has the best metric measurement results.

### Solution statements

- Build a machine learning model using three methods for comparison, namely Gradient Boosting, Random Forest, and Stacking Classifier. Each model was built using default parameters.

- Comparing the performance of the three models using appropriate assessment metrics. Some metrics that must be considered are accuracy, sensitivity (recall), specificity, and others.

In health models, the highest accuracy does not necessarily mean the best model. Also, consider other metrics such as sensitivity, which measures the model's ability to detect positive cases because a robust model is needed to detect the presence of a disease and prevent misdiagnosis. Additionally, specificity measures the model's ability to avoid detecting negative cases as positive cases. This is necessary because in a healthcare context, the model must minimize false positive predictions that cause patients to receive unnecessary treatment and waste hospital resources.

## Data Understanding

This dataset is breast cancer diagnosis outcome data taken from the University of Wisconsin Hospitals. It was contributed to the UCI Machine Learning Repository in 1995. The creators of this dataset are William Wolberg, Olvi Mangasarian, Nick Street, and W. Street. [4]

The dataset has 569 rows and 32 columns (30 numeric, target variable, and identity columns). There are no missing values in the data.

### Variabel-variabel pada Breast Cancer UCI dataset adalah sebagai berikut:

- radius: average distance from the center to the surrounding points

- texture: standard deviation of gray-scale values
- perimeter
- area
- smoothness: local variation in radius length
- compactness: perimeter^2 / area - 1.0
- concavity: severity of the concave part of the contour
- concave points: number of concave parts of the contour
- symmetry
- fractal dimension: “coastline approximation” - 1

The mean, standard error, and “worst” or largest (average of the three worst/largest values) of these features were calculated by the computer for each digital imaging image, resulting in 30 features. For example, field 0 is Mean Radius, field 10 is Radius SE, field 20 is Worst Radius.

Target variable (diagnosis):
* M - Malignant (212 data)
* B - Benign (357 data)

Benign (labeled with “0” in the data preparation stage) means benign tumor growth (not cancerous).
Malignant (labeled with “1” at the data preparation stage) means malignant tumor growth (breast cancer).

The amount of data between classes has a gap of 145 data. Although not too significant, the amount of data is imbalance when compared to the total amount of data which is only 569 data. This will be resolved with SMOTE oversampling to increase the amount of data that is lacking.

After loading the data and checking the data information, exploratory data analysis was conducted to look deeper into the dataset.

## Exploratory Data Analysis

Statistical information on the numeric columns in the dataset is viewed using df.describe(). 
  
**Univariate analysis**

Create bar charts and pie charts to see the distribution of diagnosis data. 

In the dataset, there are 357 patients with benign status, and 212 patients with malignant status.
![Bar Chart](https://i.ibb.co/TcYVHrd/D1.png)
[!Pie Chart](https://i.ibb.co/8sNTTfr/D2.png)

With this, 63% of patients in the dataset had benign status or no cancer. While 37% had breast cancer.

Box plot is a visualization created to illustrate the shape of the distribution and spread of data. The box plot shows the quartiles, distance between quartiles, minimum and maximum limits, and outliers in the data. A box plot was created for each numeric column. The box plot shows that the column has some outliers (outside the maximum limit).

Box Plot (https://i.ibb.co/rvNmjmT/BP.png)  

Violin plot is a variation of box plot plus kernel density plot to see the distribution of data such as skewness. A violin plot is created for each numerical column.

![Violin Plot](https://i.ibb.co/Z2M2zFT/VP.png)

**Multivariate analysis**

Histograms are graphs to view data distribution patterns. Histograms were created for each numeric column with an overlay of diagnosis status to see the difference in the distribution of numeric data for each type of diagnosis. In the histogram, it can be seen that the examination results for 'benign' (non-cancerous) diagnoses tend to have smaller numbers or sizes. This is in line with the size of the tumor, which will get bigger and more severe as the cancer cells grow. Cancer cells tend to be denser, thicker, irregular and rough compared to normal cells.

![Histogram](https://i.ibb.co/k6ZqmRN/HG.png)

Violin plot for each numerical column with diagnosis status overlay to see the difference in numerical data distribution for each diagnosis type.

![Violin Plot](https://i.ibb.co/H2CcBM5/VPM.png)

Viewed the average data value per column in each diagnosis to see the difference in data value in each diagnosis and can be a reference in determining the next diagnosis. This is done with groupby diagnosis.

The correlation between numerical features can be seen in the following figure.

![Correlation](https://i.ibb.co/CnynYSx/FC.png)

The correlation value is rounded to 2 numbers behind zero. The higher the value, the stronger the correlation. A positive correlation means that it is directly proportional, if the correlation is negative then the correlation is inversely proportional.

## Data Preparation

The data has 1 identity column (ID) and 1 'unknown' column that has no use, so both columns are removed when loading the data. The identity column must be removed to prevent overfitting of the model, because if there is a distinguishing identity, the model will learn to memorize the identity instead of seeing the pattern.

In the dataset, there is no empty data or inappropriate data type so there is no need to impute or change the data type.

The diagnosis results are categorical so they must be converted to numeric in order to use the model. Labeling the diagnosis column value with “B” = 0 and “M” = 1. Labeling is done by *mapping value* based on the dictionary that has been created.

After labeling the data, the data is divided into x and y. X contains the variable that determines the diagnosis result (all numeric columns except diagnosis) and Y contains the target variable (diagnosis).

Next, data balancing is done with the SMOTE oversampling method because the amount of data is not balanced. Data balancing is done so that the model can achieve higher accuracy.

The reason for choosing oversampling is because with less data, undersampling will further reduce the amount of data and potentially decrease accuracy. With the SMOTE oversampling method, the minority class (in this case the "malignant" class) will be added to the amount of data by creating synthesized data. The SMOTE method itself was chosen because it has advantages over the random oversampling method. In the random oversampling method, data from the minority class will only be duplicated until the data is balanced. While the SMOTE method performs data synthesis which will help overcome the overfitting problem caused by the random oversampling method[5].

The dataset is then divided into train and test with a ratio of 80:20.
The reason why this number is set is because 80/20 is generally considered good enough (unless the training data is very large, then the split data ratio may change). In this dataset, the number of columns is not too large so enough training data is needed to ensure the model is well trained[6].

By setting random_state = 42, the dataset will output the same random data for training and testing data. Through this sharing configuration, we get 80% training data (571 data) from the dataset and 20% testing data from the dataset (143 data).

## Modeling

This project compares 3 models created with different methods, namely Gradient Boosting, Random Forest, and Stacking Classifier.

Gradient Boosting and Random Forest are base models, while Stacking Classifier is a method to combine several base models to improve the performance of these models. The advantages of these models are quite similar with an emphasis on high accuracy, being able to handle unbalanced data, and being suitable for a wide range of data. The difference is that Random Forest tends to be more resistant to overfitting and Stacking Classifier can improve the accuracy of a possibly weaker base model by combining multiple base models. The drawbacks of the three models are also similar in that they can be time-consuming to train and can be too complex for the dataset (especially for the Stacking Classifier). The reason for experimenting with the stacking method is to produce a model that can learn a dataset from different angles with each model working differently.

In all models in this project, the same dataset is used and no parameters are given or set in order to see the comparison more clearly.

**Steps to build a Gradient Boosting model**
1.Import the *Gradient Boosting* algorithm from sklearn.

2.Training/*fitting* the model using the previously divided x_train and y_train.

Import the Gradient Boosting algorithm from sklearn.
Model training/fitting process using x_train and y_train that have been divided before.
3. Predict testing data with the trained model.

**Steps to build a Random Forest model**

1. Import Random Forest algorithm from sklearn.

2. Model training/fitting process using x_train and y_train that have been divided before.

3. Predict the testing data with the trained model.

**Steps to build the Stacking Classifier model**

1. Import the required base models such as Decision Tree, Logistic Regression, Random Forest, and XGBoost.

2. Import stacking method from sklearn.

3. Combine the base models and set logistic regression as the final model (meta-model).

4. Training/fitting the model using the previously divided x_train and y_train.

5. Predicting the testing data with the trained model.

In the Stacking Classifier, Decision Tree, Logistic Regression, Random Forest, and XGBoost models were selected. Logistic Regression was chosen as the final model. Reason:

- Decision Tree is able to handle non-linear data and can provide a good understanding of influential features. Decision Tree can provide variety in decision-making and help overcome complex situations.

- Random Forest can overcome overfitting, improve accuracy, and can handle datasets with many features. By combining multiple decision trees, Random Forest can increase model stability and improve overall performance.

- XGBoost is efficient in handling large data, improving accuracy, and optimizing computation time. XGBoost is a powerful boosting algorithm, which can help improve model stacking performance by giving each model optimal weights.

- Logistic Regression is suitable for binary classification, has easy interpretation, and does not require many parameters. As the final model, Logistic Regression provides stability in the final results and a clear interpretation of the relationship between input and output variables.

After the models were trained and made predictions from the testing data, the three models were evaluated with several metrics which will be further discussed in the evaluation section. The evaluation results will determine which model is the best.

## Evaluation

The three models that have been created will be evaluated with various metrics, namely precision, recall (also called "sensitivity" in the context of medical research), specificity, f1-score, accuracy, and roc-auc score.

In addition, confusion matrix is also used to see the true positive, true negative, false positive, and false negative values.

Gradient Boosting (https://i.ibb.co/T16m477/CMGB.png)
![Random Forest](https://i.ibb.co/5smb7c0/CMRF.png)
![Stacking Model](https://i.ibb.co/37j8VTr/CMSC.png)

Because this project is a project in the health sector, the model with the highest measurement results will be sought to minimize the chance of error.

The following metrics are used in this project:
- Accuracy: Accuracy is the most commonly used metric to measure the performance of machine learning models.

$$Accuracy = (True Positives + True Negatives) / (Total Number of Cases)$$


Accuracy is defined as the proportion of test data that is correctly classified by the model.
Accuracy relates to the number of correct predictions by the model, in the context of health, in this case breast cancer, accuracy indicates how reliable the model is in predicting the patient's diagnosis (cancer or non-cancer).

- Recall/Sensitivity: Sensitivity is a metric that measures the ability of the model to detect positive cases. Sensitivity is defined as the proportion of positive cases that are correctly classified by the model. The sensitivity calculation can be seen from the classification report in the recall calculation for class "1" (malignant or positive in this case).

$$Recall = True Positives / (True Positives + False Negatives)$$

Recall or often referred to as "sensitivity" in healthcare, serves to measure how many positive cases are correctly predicted from the actual number of positive cases. In this context, the sensitivity metric is used to see the sensitivity of the model to detect positive cases of breast cancer.

- Specificity: Specificity is a metric that measures the model's ability to avoid detecting negative cases as positive cases. Specificity is defined as the proportion of negative cases that are correctly classified by the model. The specificity calculation can be seen from the classification report in the recall calculation for class "0" (benign or negative in this case).

$$Specificity = True Negatives / (True Negatives + False Positives)$$

Specificity measures how many negative cases are correctly predicted out of the actual number of negative cases.In the case of breast cancer prediction, this is a metric that must be considered to see how often the model can correctly predict negative cases to prevent patient anxiety and unnecessary treatment.

Specificity and precision metrics have some similarities in that they both seek to reduce false positive cases, but the difference is that specificity focuses on finding true negatives while precision focuses on the rate of true positive cases out of all predicted positive cases to ensure swiftness in treatment.

- Precision: Precision is a metric that measures the model's ability to detect true positive cases. Precision is defined as the proportion of positive cases classified by the model as true positive cases.

$$Precision = True Positives / (True Positives + False Positives)$$

Precision is a metric that serves to measure how many positive cases are correctly predicted from the number of positive predictions made by the model. In the case of breast cancer prediction, it helps hospitals in predicting true positive breast cancer cases so that resources can be prioritized to those patients.

- AUC-ROC: AUC-ROC (Area Under the Receiver Operating Characteristic Curve) is a metric that measures the performance of machine learning models on binary classification tasks. AUC-ROC is calculated by graphing the True Positive Rate (TPR) against the False Positive Rate (FPR) at various thresholds, and then calculating the area under the resulting curve.

$$AUC-ROC = ∫_0^1 TPR(t) dt$$

A high value on this metric indicates the model's ability to distinguish breast cancer cases from non-cancer cases.

- F1-score: F1-score is a combination of precision and sensitivity/recall.

$$F1-score = 2 * (Precision * Recall) / (Precision + Recall)$$

F1-score shows the performance of the model in balancing its ability to correctly predict positive cases and minimize the occurrence of false alarms.

Based on the evaluation results of the three models, the score of each metric can be seen in the following table:

| **Model** | **Train Acc** | **Test Acc** | **Precision** | **Recall** | **Specificity** | **F1-score** | **ROC-AUC score** |
|:---------------------:|:-------------:|:------------:|:-------------:|:----------:|:---------------:|:------------:|:-----------------:|
| **Gradient Boosting** | 1.00 | 0.951 | 0.95 | 0.95 | 0.959 | 0.95 | 0.95 |
| **Random Forest** | 1.00 | 0.965| 0.96 | 0.97 | 0.959 | 0.97 | 0.965 |
| **Stacking Model** | 1.0 | 0.972 | 0.97 | 0.97 | 0.97 | 0.972 | 0.971 |

Based on the results of the evaluation metrics, the best model is **Stacking Classifier** with a slightly higher difference in metric results than Random Forest and Gradient Boosting.
In the health context, recall (sensitivity) and specificity values are taken into consideration as these affect the diagnosis of a disease. The Stacking Classifier model has the highest value on these two metrics, it is hoped that this can help the accuracy of patient diagnosis and reduce false positive results so as not to waste hospital resources and ensure patients with cancer get fast treatment.

But when considering the small amount of data and computational load, the base model is still very good to use in this case. In this project, the difference in time and computational load of each model has not been seen because the amount of data is small, so only the results of the evaluation metrics are used as a consideration of the best model, namely the **Stacking Classifier** model.

In cases with more data, additional evaluations such as computational time and load should be conducted to review whether the prediction time and memory used are still worth the results.

## Reference
[[1] International Agency for Research on Cancer, “GLOBOCAN 2022: 360 Indonesia Fact
Sheets”, 2022.](https://gco.iarc.who.int/media/globocan/factsheets/populations/360-indonesia-fact-sheet.pdf)
[[2] The International Association for Business Analytics Certification, “Machine Learning in Healthcare: Transforming Medical Diagnostics,”Accessed: Jan. 20, 2024. [Online]. Available: https://iabac.org/blog/machine-learning-in-healthcare-transforming-medical-diagnostics](https://iabac.org/blog/machine-learning-in-healthcare-transforming-medical-diagnostics)
[[3] S. Zakareya, H. Izadkhah, dan J. Karimpour, “A New Deep-Learning-Based Model for Breast Cancer Diagnosis from Medical Images,”  Diagnostics, vol. 13, no. 11, pp. 1944, 2023, doi:10.3390/diagnostics13111944.](https://www.mdpi.com/2075-4418/13/11/1944)
[[4] W. Wolberg, O. Mangasarian, N. Street, dan W. Street, “Breast Cancer Wisconsin (Diagnostic),” UCI Machine Learning Repository, 1995, doi:10.24432/C5DW2B.](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
[[5] J. Brownlee, “Random Oversampling and Undersampling for Imbalanced Classification,” Accessed: Jan. 20, 2024. [Online]. Available: https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/)
[[6] V. Roshan Joseph, “Optimal Ratio for Data Splitting,” Statistical Analysis and Data Mining, The ASA Data Science Journal, vol. 15, no. 4, pp. 531-538, 2022, doi:10.1002/sam.11583.](https://onlinelibrary.wiley.com/doi/full/10.1002/sam.11583)
