import pandas as pd


def classify_prediction(actual_label, predicted_label, normal_label='NORMAL'):
    if actual_label is None or pd.isna(actual_label):
        return 'UNVERIFIED'

    actual = str(actual_label).strip().upper()
    predicted = str(predicted_label).strip().upper()
    normal = str(normal_label).strip().upper()
    if not actual:
        return 'UNVERIFIED'
    if actual == normal:
        return 'TRUE_NEGATIVE' if predicted == normal else 'FALSE_POSITIVE'
    if predicted == normal:
        return 'FALSE_NEGATIVE'
    if actual == predicted:
        return 'TRUE_POSITIVE'
    return 'MISCLASSIFIED_ATTACK'