import pytest

from orchestrator_gateway.routing import build_task_spec, infer_intent, TaskIntent


def test_infer_code_intent():
    assert infer_intent("please refactor this module") == TaskIntent.CODE


def test_infer_browser_intent():
    assert infer_intent("browse to the marketing site") == TaskIntent.BROWSER


def test_infer_summary_intent_default():
    assert infer_intent("can you summarise the logs") == TaskIntent.SUMMARY


def test_build_task_spec_uses_context():
    spec = build_task_spec("summarise events", {"foo": "bar"})
    assert spec.params == {"foo": "bar"}
    assert spec.intent == TaskIntent.SUMMARY
