import app as flask_app


def test_security_headers_are_present():
    response = flask_app.app.test_client().get('/api/health')

    assert response.headers.get('X-Content-Type-Options') == 'nosniff'
    assert response.headers.get('X-Frame-Options') == 'DENY'
    assert 'nosniff' not in response.get_data(as_text=True) or True


def test_api_token_is_enforced_when_configured(monkeypatch):
    monkeypatch.setattr(flask_app.config, 'API_TOKEN', 'secret-123')
    client = flask_app.app.test_client()

    denied = client.post('/api/sim-control', json={'domain': 'can', 'action': 'stop'})
    assert denied.status_code == 401

    allowed = client.post(
        '/api/sim-control',
        json={'domain': 'can', 'action': 'stop'},
        headers={'X-API-Token': 'secret-123'},
    )
    assert allowed.status_code == 200

    bearer = client.post(
        '/api/sim-control',
        json={'domain': 'can', 'action': 'stop'},
        headers={'Authorization': 'Bearer secret-123'},
    )
    assert bearer.status_code == 200


def test_api_token_disabled_by_default(monkeypatch):
    monkeypatch.setattr(flask_app.config, 'API_TOKEN', '')
    response = flask_app.app.test_client().post(
        '/api/sim-control', json={'domain': 'can', 'action': 'stop'}
    )

    assert response.status_code == 200
