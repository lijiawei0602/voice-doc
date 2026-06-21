from typing import Any


def success_response(data: Any, message: str = "success") -> dict[str, Any]:
    return {"code": "SUCCESS", "message": message, "data": data}


def error_response(code: str, message: str, details: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        payload["details"] = details
    return payload
