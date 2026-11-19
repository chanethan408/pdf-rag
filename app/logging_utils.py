"""Request-scoped logging helpers. Logs IDs, scores, latency and status only - never keys, questions or document text."""
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    return request_id_var.get()
