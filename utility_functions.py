import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc


#----------------------------------
#-Classification Confusion Matrix--
#----------------------------------
def my_confusion_matrix(actual_df, predicted_l):
    """
    Function that calculates the confusion matrix.

    Parameters
    ----------
    actual_df : dataframe
        Actual values in the format of a pandas dataframe.
    predicted : array-like
        Predicted values (can be a pandas Series, list, or numpy array).

    Returns
    -------
    tuple
        Returns a tuple containing True Positives (tp), False Positives (fp),
        False Negatives (fn), and True Negatives (tn).
    """
    # True Positives
    tp = sum(actual_df[actual_df == predicted_l] == 1)
    # False Positive
    fp = sum(actual_df[actual_df != predicted_l] == 0)
    # False Negative
    fn = sum(actual_df[actual_df != predicted_l] == 1)
    # True Negatives
    tn = sum(actual_df[actual_df == predicted_l] == 0)

    return tp, fp, fn, tn

# Example usage
# actual = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
# predicted = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
# tp, fp, fn, tn = my_confusion_matrix(actual, predicted)
# print(f'TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}')



#-------------------------
#--Classification Scores--
#-------------------------
def my_classification_scores(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """
    Function that calculates classification scores from the confusion matrix.

    Parameters
    ----------
    tp : int
        True Positives.
    fp : int
        False Positives.
    fn : int
        False Negatives.
    tn : int
        True Negatives.

    Returns
    -------
    dict
        A dictionary containing accuracy, sensitivity, specificity, and precision scores.
    """
    
    # Calculate the total number of predictions
    total = tp + fp + fn + tn

    try:
        # Calculate metrics
        accuracy = (tp + tn) / total
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    except ZeroDivisionError:
        # Handle division by zero by setting metrics to 0.0
        accuracy = 0.0
        sensitivity = 0.0
        specificity = 0.0
        precision = 0.0
        f1_score = 0.0

    results = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score
    }
    
    return results

# Example usage
# tp, fp, fn, tn = 50, 10, 5, 100
# scores = my_classification_scores(tp, fp, fn, tn)
# print(scores)
# Example usage


#-------------------------
#------Dataset Split------
#-------------------------

def my_dataframe_split(input_df, train_size = 0.80, validation_size=0, test_size=0.20, random_state = None):
    """Function that randomly sorts and split the dataframe between training, validation and test data"""

    """
    Parameters
    ---------
    input_df : pd.DataFrame
        DataFrame to be sorted and split.
    train_size : float, optional
        Percentage of the data used as training data. Default is 0.80.
    validation_size : float, optional
        Percentage of the data used as validation data. Default is 0.
    test_size : float, optional
        Percentage of the data used as test data. Default is 0.20.
    random_state : int, optional
        Seed for reproducibility. Default is None.

    Returns
    -------
    tuple of pd.DataFrame
        Returns three dataframes: train_data, validation_data, and test_data.
    ---------
    """
    # Validate that the sizes sum to 1
    if round(train_size + validation_size + test_size, 10) != 1.0:
        raise ValueError("The sum of train_size, validation_size, and test_size must be 1.0")
    
    # Setting the seed for reproducibility
    if random_state != None:
        seed_value = random_state
        np.random.seed(seed_value)
    
    #Randomly sort the data
    random_indices = np.random.permutation(input_df.index)
    df_random = input_df.iloc[random_indices].reset_index(drop=True)

    # Calculate the number of rows for each split
    train_amount = round(len(df_random)*train_size)
    validation_amount = round(len(df_random)*validation_size)
  
    # Define indices for each split, begining and end
    train_e = train_amount
    validation_b = train_e
    validation_e = validation_b + validation_amount
    test_b = validation_e

    train_data = df_random.iloc[:train_e]
    validation_data = df_random.iloc[validation_b:validation_e]
    test_data = df_random.iloc[test_b:]

    return(train_data,validation_data,test_data)


# Example use
# example_path = "data/train.csv"
# example_df = pd.read_csv(example_path)
# my_test_df = example_df[:10000]

# train_df, validation_df, test_df = my_dataframe_split(my_test_df,0.7,0.15,0.15)

# train_df
# validation_df
# test_df

# forecast = pd.Series([1] * len(train_df["Response"]))

# tp, fp, fn, tn = my_confusion_matrix(train_df["Response"],forecast)
# scores = my_classification_scores(tp, fp, fn, tn)

