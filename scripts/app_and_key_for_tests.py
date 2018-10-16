import os
import sys

import requests


EMAIL = os.environ['CLARIFAI_USER_EMAIL']
PASSWORD = os.environ['CLARIFAI_USER_PASSWORD']


BASE = 'https://api.clarifai.com/v2'


def create_app(env_name):
  session_token, user_id = _login()

  url = '%s/users/%s/apps' % (BASE, user_id)
  payload = {'apps': [{'name': 'auto-created-in-%s-ci-test-run' % env_name}]}
  response = requests.post(url=url, json=payload, headers=_auth_headers(session_token))
  _raise_on_http_error(response)
  data = response.json()
  app_id = data['apps'][0]['id']

  # This print needs to be present so we can read the value in CI.
  print(app_id)


def create_key(app_id):
  session_token, user_id = _login()

  url = '%s/users/%s/keys' % (BASE, user_id)
  payload = {
    'keys': [{
      'description': 'Auto-created in a CI test run',
      'scopes': ['All'],
      'apps': [{'id': app_id, 'user_id': user_id}]
    }]
  }
  response = requests.post(url=url, json=payload, headers=_auth_headers(session_token))
  _raise_on_http_error(response)
  data = response.json()
  key_id = data['keys'][0]['id']

  # This print needs to be present so we can read the value in CI.
  print(key_id)


def delete(app_id):
  session_token, user_id = _login()

  # All the related keys will be deleted automatically when the app is deleted
  _delete_app(session_token, user_id, app_id)


def _delete_app(session_token, user_id, app_id):
  url = '%s/users/%s/apps/%s' % (BASE, user_id, app_id)
  response = requests.delete(url=url, headers=_auth_headers(session_token))
  _raise_on_http_error(response)


def _auth_headers(session_token):
  headers = {'Content-Type': 'application/json', 'X-Clarifai-Session-Token': session_token}
  return headers


def _login():
  url = '%s/login' % BASE
  payload = {'email': EMAIL, 'password': PASSWORD}
  response = requests.post(url=url, json=payload)
  _raise_on_http_error(response)
  data = response.json()
  user_id = data['v2_user_id']
  session_token = data['session_token']
  return session_token, user_id


def _raise_on_http_error(response):
  if int(response.status_code) // 100 != 2:
    raise Exception('Unexpected response %s: %s' % (response.status_code, response.text))


def run(arguments):
  command = arguments[0] if arguments else '--help'
  if command == '--create-app':
    if len(arguments) != 2:
      raise Exception('--create-app takes one argument')

    env_name = arguments[1]
    create_app(env_name)
  elif command == '--create-key':
    if len(arguments) != 2:
      raise Exception('--create-key takes one argument')

    app_id = arguments[1]
    create_key(app_id)
  elif command == '--delete-app':
    if len(arguments) != 2:
      raise Exception('--delete-app takes one argument')
    app_id = arguments[1]
    delete(app_id)
  elif command == '--help':
    print('''DESCRIPTION: Creates and delete applications and API keys
ARGUMENTS: 
--create-app [env_name]   ... Creates a new application.
--create-key [app_id]     ... Creates a new API key.
--delete-app [app_id]     ... Deletes an application (API keys that use it are deleted as well).
--help                    ... This text.''')
  else:
    print('Unknown argument. Please see --help')
    exit(1)


if __name__ == '__main__':
  run(arguments=sys.argv[1:])
