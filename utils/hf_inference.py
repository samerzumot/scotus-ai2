from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp


class HFInferenceError(RuntimeError):
    pass


def _extract_generated_text(payload: Any) -> str:
    # Common response shapes:
    # - [{"generated_text": "..."}]
    # - [{"summary_text": "..."}] (summarization)
    # - {"generated_text": "..."} (less common)
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            for k in ("generated_text", "summary_text", "translation_text"):
                if k in first and isinstance(first[k], str):
                    return first[k]
    if isinstance(payload, dict):
        for k in ("generated_text", "summary_text", "translation_text"):
            if k in payload and isinstance(payload[k], str):
                return payload[k]
    return ""


def _is_hf_error(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, str) and err.strip():
            return err.strip()
        # Some errors appear as {"estimated_time":...,"error":...}
    return None


@dataclass(frozen=True)
class HFGenerationParams:
    max_new_tokens: int = 900
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True
    return_full_text: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": int(self.max_new_tokens),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "do_sample": bool(self.do_sample),
            "return_full_text": bool(self.return_full_text),
        }


class HFInferenceClient:
    def __init__(
        self,
        *,
        session: aiohttp.ClientSession,
        api_token: Optional[str],
        base_url: str = "https://api-inference.huggingface.co",
        timeout_s: int = 35,
    ) -> None:
        self._session = session
        self._token = api_token or ""
        self._base_url = base_url.rstrip("/")
        self._timeout_s = int(timeout_s)

    def _headers(self) -> Dict[str, str]:
        h = {"Accept": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    async def _post(self, *, model: str, payload: Dict[str, Any]) -> Any:
        url = f"{self._base_url}/models/{model}"
        timeout = aiohttp.ClientTimeout(total=self._timeout_s)
        async with self._session.post(url, json=payload, headers=self._headers(), timeout=timeout) as resp:
            text = await resp.text()
            try:
                data = json.loads(text)
            except Exception:
                raise HFInferenceError(f"HF non-JSON response ({resp.status}): {text[:300]}")
            err = _is_hf_error(data)
            if resp.status >= 400 or err:
                raise HFInferenceError(f"HF error ({resp.status}): {err or data}")
            return data

    async def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        params: Optional[HFGenerationParams] = None,
    ) -> str:
        params = params or HFGenerationParams()
        payload = {
            "inputs": prompt,
            "parameters": params.to_dict(),
            "options": {"wait_for_model": True, "use_cache": True},
        }
        data = await self._post(model=model, payload=payload)
        out = _extract_generated_text(data)
        if not out:
            raise HFInferenceError(f"HF did not return generated text: {data}")
        return out

    async def feature_extraction(self, *, model: str, text: str) -> List[float]:
        payload = {
            "inputs": text,
            "options": {"wait_for_model": True, "use_cache": True},
        }
        data = await self._post(model=model, payload=payload)
        # Common shapes: [[...]] or [...]
        if isinstance(data, list) and data and isinstance(data[0], list):
            vec = data[0]
        elif isinstance(data, list):
            vec = data
        else:
            raise HFInferenceError(f"HF returned unexpected embedding payload: {type(data)}")

        out: List[float] = []
        for x in vec:
            try:
                out.append(float(x))
            except Exception:
                continue
        if not out:
            raise HFInferenceError("HF embedding vector was empty.")
        return out


