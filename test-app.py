import pytest
from unittest.mock import patch, MagicMock
from app import app as flask_app


@pytest.fixture
def app():
    # Set up your app with test configurations
    flask_app.config.update({
        "TESTING": True,
    })
    yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@patch('app.get_db_connection')
def test_get_languages(mock_get_db_connection, client):
    # Set up a mock cursor and connection
    mock_cursor = MagicMock()
    mock_connection = MagicMock()
    mock_get_db_connection.return_value = mock_connection
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [
        {'id': 1, 'english': 'Hello', 'taino': 'Taino', 'french': 'Bonjour', 'latin': '', 'spanish': '', 'gaelic': ''}
    ]

    response = client.get("/get_languages")
    assert response.status_code == 200
    assert isinstance(response.json, dict)
    assert "languages" in response.json
    assert response.json['languages'][0]['english'] == 'Hello'
