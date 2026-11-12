"""Conservative mappings to documented MITRE ATT&CK techniques."""

DOS_NAMES = {'DOS', 'DOSRANDOM', 'DOSDISRUPTIVE', 'DOSRANDOMSYBIL', 'DOSDISRUPTIVESYBIL'}


def map_attack(attack_type):
    label = str(attack_type or '').strip().upper().replace('-', '').replace('_', '').replace(' ', '')
    if label not in DOS_NAMES:
        return []
    return [
        {
            'technique_id': 'T1498',
            'technique_name': 'Network Denial of Service',
            'framework': 'MITRE ATT&CK Enterprise',
            'justification': 'The predicted class denotes denial of service. This is a broad behavioral mapping; analyst review is required to confirm the ATT&CK scope and tactic for the observed vehicle/network evidence.',
            'reference_url': 'https://attack.mitre.org/techniques/T1498/',
            'requires_review': True,
        },
        {
            'technique_id': 'T0814',
            'technique_name': 'Denial of Service',
            'framework': 'MITRE ATT&CK for ICS',
            'justification': 'MITRE describes DoS as disruption of expected device functionality. The project prediction alone does not prove a specific ICS target or operational impact, so analyst review is required.',
            'reference_url': 'https://attack.mitre.org/techniques/T0814/',
            'requires_review': True,
        },
    ]
