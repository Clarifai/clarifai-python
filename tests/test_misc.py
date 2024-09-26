import datetime
import json
import logging

from rich.logging import RichHandler

from clarifai.utils.logging import JsonFormatter, _get_library_name, get_logger, set_logger_context


def test_get_logger():
  logger = get_logger("DEBUG", "test_logger")
  assert logger.level == logging.DEBUG
  assert logger.name == "test_logger"
  assert isinstance(logger.handlers[0], RichHandler)


def test_get_logger_defaults():
  logger = get_logger()
  assert logger.level == logging.NOTSET
  assert logger.name == _get_library_name()
  assert isinstance(logger.handlers[0], RichHandler)


def test_json_logger():
  filename = "testy.py"
  msg = "testy"
  lineno = 1
  sinfo = "testf2"
  r = logging.LogRecord(
      name="testy",
      level=logging.INFO,
      pathname=filename,
      lineno=lineno,
      msg=msg,
      args=(),
      exc_info=None,
      func="testf",
      sinfo=sinfo)
  jf = JsonFormatter()
  # format the record as a json line.
  json_line = jf.format(r)
  result = json.loads(json_line)
  # parse timestamp of format "2024-09-24T22:06:49.573038Z" in @timestamp field.
  assert result["@timestamp"] is not None
  ts = result["@timestamp"]
  # assert the ts was within 10 seconds of now (in UTC time)
  assert abs(datetime.datetime.utcnow() - datetime.datetime.strptime(
      ts, "%Y-%m-%dT%H:%M:%S.%fZ")) < datetime.timedelta(seconds=10)
  assert result['filename'] == filename
  assert result['msg'] == msg
  assert result['lineno'] == lineno
  assert result['stack_info'] == sinfo
  assert result['level'] == "info"
  assert 'req_id' not in result

  req_id = "test_req_id"
  set_logger_context(req_id=req_id)
  json_line = jf.format(r)
  result = json.loads(json_line)
  assert abs(datetime.datetime.utcnow() - datetime.datetime.strptime(
      ts, "%Y-%m-%dT%H:%M:%S.%fZ")) < datetime.timedelta(seconds=10)
  assert result['filename'] == filename
  assert result['msg'] == msg
  assert result['lineno'] == lineno
  assert result['stack_info'] == sinfo
  assert result['level'] == "info"
  assert result['req_id'] == req_id
