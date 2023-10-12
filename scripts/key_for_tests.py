#!/usr/bin/env python

import argparse
import json
import os
import sys

try:
  from urllib.error import HTTPError
  from urllib.request import HTTPHandler, Request, build_opener
except ImportError:
  from urllib2 import HTTPError, HTTPHandler, Request, build_opener

EMAIL = os.environ["CLARIFAI_USER_EMAIL"]
PASSWORD = os.environ["CLARIFAI_USER_PASSWORD"]


def _assert_response_success(response):
  assert "status" in response, f"Invalid response {response}"
  assert "code" in response["status"], f"Invalid response {response}"
  assert response["status"]["code"] == 10000, f"Invalid response {response}"


def _request(method, url, payload={}, headers={}):
  base_url = os.environ.get("CLARIFAI_GRPC_BASE", "api.clarifai.com")

  opener = build_opener(HTTPHandler)
  full_url = f"https://{base_url}/v2{url}"
  request = Request(full_url, data=json.dumps(payload).encode())
  for k in headers.keys():
    request.add_header(k, headers[k])
  request.get_method = lambda: method
  try:
    response = opener.open(request).read().decode()
  except HTTPError as e:
    error_body = e.read().decode()
    try:
      error_body = json.dumps(json.loads(error_body), indent=4)
    except Exception:
      pass
    raise Exception("ERROR after a HTTP request to: %s %s" % (method, full_url) +
                    ". Response: %d %s:\n%s" % (e.code, e.reason, error_body))
  return json.loads(response)


def login():
  url = "/login"
  payload = {"email": EMAIL, "password": PASSWORD}
  data = _request(method="POST", url=url, payload=payload)
  _assert_response_success(data)

  assert "v2_user_id" in data, f"Invalid response {data}"
  user_id = data["v2_user_id"]
  assert user_id, f"Invalid response {data}"

  assert "session_token" in data, f"Invalid response {data}"
  session_token = data["session_token"]
  assert session_token, f"Invalid response {data}"

  return session_token, user_id


def _auth_headers(session_token):
  headers = {"Content-Type": "application/json", "X-Clarifai-Session-Token": session_token}
  return headers


def create_pat():
  session_token, user_id = login()
  os.environ["CLARIFAI_USER_ID"] = user_id

  url = "/users/%s/keys" % user_id
  payload = {
      "keys": [{
          "description": "Auto-created in a CI test run",
          "scopes": ["All"],
          "type": "personal_access_token",
          "apps": [],
      }]
  }
  data = _request(method="POST", url=url, payload=payload, headers=_auth_headers(session_token))
  _assert_response_success(data)

  assert "keys" in data, f"Invalid response {data}"
  assert len(data["keys"]) == 1, f"Invalid response {data}"
  assert "id" in data["keys"][0], f"Invalid response {data}"
  pat_id = data["keys"][0]["id"]
  assert pat_id, f"Invalid response {data}"

  # This print needs to be present so we can read the value in CI.
  print(pat_id)


def run(arguments):
  if arguments.email:
    global EMAIL
    EMAIL = arguments.email  # override the default testing email
  if arguments.password:
    global PASSWORD
    PASSWORD = arguments.password  # override the default testing password
  # these options are mutually exclusive
  if arguments.create_pat:
    create_pat()
  elif arguments.get_userid:
    _, user_id = login()
    # This print needs to be present so we can read the value in CI.
    print(user_id)
  else:
    print(f"No relevant arguments specified. Run {sys.argv[0]} --help to see available options")
    exit(1)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Create Applications, Keys, and Workflows for testing.")
  parser.add_argument(
      "--user-email",
      dest="email",
      help=
      "The email of the account for which the command will run. (Defaults to ${CLARIFAI_USER_EMAIL})",
  )
  parser.add_argument(
      "--user-password",
      dest="password",
      help=
      "The password of the account for which the command will run. (Defaults to ${CLARIFAI_USER_PASSWORD})",
  )
  group = parser.add_mutually_exclusive_group()
  group.add_argument("--create-pat", action="store_true", help=" Creates a new PAT key.")
  group.add_argument("--get-userid", action="store_true", help=" Gets the user id.")

  args = parser.parse_args()
  run(args)
