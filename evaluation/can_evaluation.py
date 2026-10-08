import numpy as np
import pandas as pd
from statistics import median
from time import perf_counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ModelEvaluator:
    """
    Evaluates ML models for Automotive Cybersecurity applications.
    Calculates Accuracy, Precision, Recall, F1, False Positive Rate (FPR),
    Confusion Matrix, and Inference Time without fabricating any values.
    """
    @staticmethod
    def evaluate_model(model, X_test, y_test, model_name="ML Model", normal_label='NORMAL'):
        if len(X_test) == 0:
            raise ValueError('Cannot evaluate a model on an empty test set.')

        y_pred = model.predict(X_test)

        timing_count = min(len(X_test), 10000)
        timing_features = X_test.iloc[:timing_count] if hasattr(X_test, 'iloc') else X_test[:timing_count]
        model.predict(timing_features)
        timing_runs = []
        for _ in range(3):
            start_time = perf_counter()
            model.predict(timing_features)
            timing_runs.append(perf_counter() - start_time)
        inference_time_ms = median(timing_runs) * 1000.0 / timing_count

        # Overall multi-class / binary metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        macro_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Calculate Confusion Matrix
        labels = sorted(list(set(y_test)))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        
        # Calculate False Positive Rate (FPR) for non-normal classes vs normal class
        # FPR = FP / (FP + TN)
        if normal_label in labels:
            normal_idx = labels.index(normal_label)
            fp = np.sum(cm[normal_idx, :]) - cm[normal_idx, normal_idx]
            tn = cm[normal_idx, normal_idx]
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        else:
            fpr = 0.0

        results = {
            'model_name': model_name,
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'precision_macro': float(macro_prec),
            'recall_macro': float(macro_rec),
            'f1_macro': float(macro_f1),
            'false_positive_rate': float(fpr),
            'inference_time_ms': float(inference_time_ms),
            'inference_sample_count': int(timing_count),
            'labels': [str(l) for l in labels],
            'confusion_matrix': cm.tolist()
        }
        
        return results

    @staticmethod
    def print_evaluation_report(results):
        print(f"\n==========================================")
        print(f" MODEL EVALUATION REPORT: {results['model_name']}")
        print(f"==========================================")
        print(f" Accuracy:            {results['accuracy']*100:.2f}%")
        print(f" Precision (Weighted):{results['precision']*100:.2f}%")
        print(f" Recall (Weighted):   {results['recall']*100:.2f}%")
        print(f" F1-Score (Weighted): {results['f1_score']*100:.2f}%")
        print(f" Precision (Macro):   {results['precision_macro']*100:.2f}%")
        print(f" Recall (Macro):      {results['recall_macro']*100:.2f}%")
        print(f" F1-Score (Macro):    {results['f1_macro']*100:.2f}%")
        print(f" False Positive Rate: {results['false_positive_rate']*100:.2f}%")
        print(f" Inference Time:      {results['inference_time_ms']:.4f} ms/sample")
        print(f" Confusion Matrix:\n{np.array(results['confusion_matrix'])}\n")
