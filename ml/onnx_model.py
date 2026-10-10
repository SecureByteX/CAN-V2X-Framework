"""Small ONNX Runtime adapter matching the sklearn prediction interface."""

import numpy as np
import onnxruntime as ort


class LabelDecoder:
    def __init__(self, classes):
        self.classes_ = np.asarray(classes)

    def inverse_transform(self, indices):
        return self.classes_[np.asarray(indices, dtype=np.int64)]


class OnnxClassifier:
    """Expose predict/predict_proba for classifiers exported with ZipMap off."""

    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        outputs = self.session.get_outputs()
        if len(outputs) < 2:
            raise ValueError(f"ONNX classifier has no probability output: {model_path}")

        # Classifier converters may order the label and probability outputs
        # differently. The probabilities are the rank-two tensor.
        probability_outputs = [output for output in outputs if len(output.shape) == 2]
        label_outputs = [output for output in outputs if len(output.shape) == 1]
        if not probability_outputs or not label_outputs:
            raise ValueError(f"Unexpected ONNX classifier outputs: {model_path}")
        self.probability_output = probability_outputs[0].name
        self.label_output = label_outputs[0].name

    @staticmethod
    def _matrix(features):
        if hasattr(features, "to_numpy"):
            features = features.to_numpy()
        return np.asarray(features, dtype=np.float32, order="C")

    def _run(self, features):
        return self.session.run(
            [self.label_output, self.probability_output],
            {self.input_name: self._matrix(features)},
        )

    def predict(self, features):
        labels, _ = self._run(features)
        return np.asarray(labels, dtype=np.int64).reshape(-1)

    def predict_proba(self, features):
        _, probabilities = self._run(features)
        return np.asarray(probabilities, dtype=np.float32)
