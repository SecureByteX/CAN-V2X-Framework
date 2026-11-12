"""Deterministic and auditable SOC risk scoring."""

SEVERITY_POINTS = {'UNKNOWN': 25, 'INFO': 5, 'LOW': 20, 'MEDIUM': 45, 'HIGH': 70, 'CRITICAL': 90}
OUTCOME_ADJUSTMENTS = {
    'TRUE_POSITIVE': 5,
    'MISCLASSIFIED_ATTACK': 3,
    'FALSE_NEGATIVE': 5,
    'FALSE_POSITIVE': -25,
    'UNVERIFIED': -10,
    'TRUE_NEGATIVE': -30,
}


def risk_level(score):
    if score >= 80:
        return 'CRITICAL'
    if score >= 55:
        return 'HIGH'
    if score >= 30:
        return 'MEDIUM'
    return 'LOW'


def score_alert(alert, severity_points=None, outcome_adjustments=None):
    severity_points = severity_points or SEVERITY_POINTS
    outcome_adjustments = outcome_adjustments or OUTCOME_ADJUSTMENTS
    severity = str(alert.get('severity') or 'UNKNOWN').upper()
    outcome = str(alert.get('prediction_outcome') or alert.get('outcome') or 'UNVERIFIED').upper()
    confidence = alert.get('confidence')
    confidence_component = None if confidence is None else round(float(confidence) * 100, 2)
    severity_component = float(severity_points.get(severity, severity_points['UNKNOWN']))
    # Severity contributes 70%; model confidence contributes only 30%.
    # Outcome calibration separates verified attacks, false positives and
    # unverified predictions instead of treating confidence as proof.
    raw = severity_component * 0.70 + (confidence_component or 0.0) * 0.30
    adjustment = float(outcome_adjustments.get(outcome, -10))
    score = round(max(0.0, min(100.0, raw + adjustment)), 2)
    factors = {
        'severity': severity,
        'severity_component': severity_component,
        'confidence_component': confidence_component,
        'confidence_weight': 0.30,
        'severity_weight': 0.70,
        'prediction_outcome': outcome,
        'outcome_adjustment': adjustment,
        'formula': '0.70 * severity_points + 0.30 * (confidence * 100 when present) + outcome_adjustment',
    }
    if outcome in {'FALSE_POSITIVE', 'UNVERIFIED', 'TRUE_NEGATIVE'}:
        factors['evidence_note'] = 'Prediction is not treated as a confirmed attack.'
    return {'risk_score': score, 'risk_level': risk_level(score), 'factors': factors}


def score_incident(alert_assessments, correlation_strength=0.0):
    assessments = list(alert_assessments)
    if not assessments:
        raise ValueError('At least one alert assessment is required.')
    highest = max(float(item['risk_score']) for item in assessments)
    corroboration = min(12.0, max(0, len(assessments) - 1) * 3.0)
    correlation_bonus = 10.0 * max(0.0, min(1.0, float(correlation_strength))) if len(assessments) > 1 else 0.0
    score = round(min(100.0, highest + corroboration + correlation_bonus), 2)
    factors = {
        'highest_alert_score': highest,
        'related_alert_count': len(assessments),
        'corroboration_points': corroboration,
        'correlation_strength': round(float(correlation_strength), 3),
        'correlation_bonus': round(correlation_bonus, 2),
        'formula': 'highest alert score + up to 12 points for additional alerts + at most 10 correlation points',
    }
    return {'risk_score': score, 'risk_level': risk_level(score), 'factors': factors}
