"""On-demand SHAP Kernel explanations over the active ONNX model interface."""

import numpy as np
import pandas as pd


class ExplanationUnavailable(RuntimeError):
    pass


def explain_prediction(alert, predictor, sample_path, sample_limit=20, nsamples=100):
    if predictor is None:
        raise ExplanationUnavailable('The prediction model is unavailable.')
    features = alert.get('relevant_features') or {}
    feature_names = list(getattr(predictor, 'feature_names', []))
    missing = [name for name in feature_names if name not in features]
    if missing:
        raise ExplanationUnavailable('The stored prediction does not include all model input features.')
    try:
        import shap
    except ImportError as exc:
        raise ExplanationUnavailable('SHAP is not installed in this deployment.') from exc

    try:
        samples = pd.read_csv(sample_path, nrows=max(sample_limit * 4, sample_limit))
        background = samples[feature_names].apply(pd.to_numeric, errors='coerce').dropna().head(sample_limit)
    except Exception as exc:
        raise ExplanationUnavailable('A compatible model background dataset is unavailable.') from exc
    if background.empty:
        raise ExplanationUnavailable('A compatible model background dataset is empty.')

    row = pd.DataFrame([[float(features[name]) for name in feature_names]], columns=feature_names)
    classes = [str(value) for value in predictor.label_encoder.classes_]
    predicted_class = str(alert.get('predicted_class') or alert.get('attack_type') or '')
    try:
        class_index = classes.index(predicted_class)
    except ValueError as exc:
        raise ExplanationUnavailable('Stored predicted class is incompatible with the active model.') from exc

    def target_probability(matrix):
        frame = pd.DataFrame(np.asarray(matrix), columns=feature_names)
        return np.asarray(predictor.model.predict_proba(frame))[:, class_index]

    try:
        explainer = shap.KernelExplainer(target_probability, background.to_numpy())
        values = explainer.shap_values(row.to_numpy(), nsamples=nsamples)
        if isinstance(values, list):
            values = values[0]
        contributions = np.asarray(values).reshape(-1)
        base_values = np.asarray(explainer.expected_value).reshape(-1)
        base_value = float(base_values[0])
        output_value = float(target_probability(row.to_numpy())[0])
    except Exception as exc:
        raise ExplanationUnavailable('SHAP could not explain this model output.') from exc

    ranked = sorted(
        ({'feature': name, 'value': float(row.iloc[0][name]), 'contribution': float(contributions[index])}
         for index, name in enumerate(feature_names)),
        key=lambda item: abs(item['contribution']), reverse=True,
    )
    return {
        'alert_id': alert['alert_id'],
        'prediction': predicted_class,
        'explainer': 'SHAP KernelExplainer',
        'base_value': base_value,
        'output_value': output_value,
        'contributions': ranked,
        'top_features': ranked[:10],
        'limitations': 'Contributions explain the selected class probability relative to sampled holdout background records; they are not causal claims.',
    }
