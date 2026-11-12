"""Optional OpenAI-compatible incident investigation; never used for detection."""

import json
import os
import urllib.error
import urllib.request


REQUIRED_FIELDS = {
    'incident_summary': str,
    'suspected_attack': str,
    'evidence': list,
    'possible_impact': str,
    'mitre_techniques': list,
    'recommended_investigation': list,
    'recommended_actions': list,
    'confidence_and_limitations': str,
}


class InvestigationUnavailable(RuntimeError):
    pass


def configured():
    return bool(os.environ.get('LLM_API_KEY', '').strip())


def _validate_result(value):
    if not isinstance(value, dict):
        raise InvestigationUnavailable('Investigation provider returned a non-object response.')
    result = {}
    for field, expected in REQUIRED_FIELDS.items():
        item = value.get(field)
        if not isinstance(item, expected):
            raise InvestigationUnavailable(f'Investigation provider returned an invalid {field} field.')
        if expected is str:
            result[field] = item[:4000]
        else:
            if any(not isinstance(entry, str) for entry in item):
                raise InvestigationUnavailable(f'Investigation provider returned invalid entries in {field}.')
            result[field] = [entry[:1000] for entry in item[:20]]
    return result


def investigate(incident, timeout_seconds=20):
    api_key = os.environ.get('LLM_API_KEY', '').strip()
    if not api_key:
        raise InvestigationUnavailable('LLM investigation is not configured.')
    base_url = os.environ.get('LLM_API_URL', 'https://api.openai.com/v1/chat/completions').strip()
    model = os.environ.get('LLM_MODEL', '').strip()
    if not model:
        raise InvestigationUnavailable('LLM_MODEL must be configured when LLM_API_KEY is set.')
    safe_evidence = {
        'incident_id': incident.get('incident_id'),
        'status': incident.get('status'),
        'risk': incident.get('risk'),
        'correlation_reason': incident.get('correlation_reason'),
        'alerts': [
            {key: alert.get(key) for key in (
                'source', 'timestamp', 'attack_type', 'predicted_class', 'confidence',
                'severity', 'prediction_outcome', 'vehicle_id', 'can_id', 'sender_id',
                'receiver_id', 'relevant_features', 'model_name'
            )}
            for alert in incident.get('alerts', [])[:20]
        ],
        'shap_explanations': incident.get('shap_explanations', []),
        'mitre': incident.get('mitre', []),
    }
    prompt = (
        'Investigate the supplied vehicle-security incident evidence. Do not claim an attack is confirmed; '
        'distinguish observed facts from hypotheses. Do not issue commands or claim actions were executed. '
        'Return only a JSON object with fields: incident_summary, suspected_attack, evidence (string array), '
        'possible_impact, mitre_techniques (string array), recommended_investigation (string array), '
        'recommended_actions (string array), confidence_and_limitations.\n\n'
        + json.dumps(safe_evidence, separators=(',', ':'), default=str)[:18000]
    )
    request_body = json.dumps({
        'model': model,
        'temperature': 0.1,
        'response_format': {'type': 'json_object'},
        'messages': [
            {'role': 'system', 'content': 'You are an incident investigation assistant. Analyze only supplied evidence.'},
            {'role': 'user', 'content': prompt},
        ],
    }).encode('utf-8')
    req = urllib.request.Request(
        base_url,
        data=request_body,
        headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            envelope = json.loads(response.read().decode('utf-8'))
        content = envelope['choices'][0]['message']['content']
        return _validate_result(json.loads(content)), model, base_url
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise InvestigationUnavailable('The configured investigation provider could not be reached.') from exc
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise InvestigationUnavailable('The investigation provider returned malformed structured output.') from exc
