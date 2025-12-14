from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import google.generativeai as genai


class GoogleInferenceError(RuntimeError):
    pass


class GoogleInferenceClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
    ) -> None:
        self._api_key = (api_key or os.getenv("GOOGLE_AI_KEY") or "").strip()
        if not self._api_key:
            raise GoogleInferenceError("GOOGLE_AI_KEY not set. Set it in env.local or pass api_key parameter.")
        genai.configure(api_key=self._api_key)

    async def generate_text(
        self,
        *,
        model: str = "models/gemini-2.5-pro",
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 8192,
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Generate text using Google Gemini API.
        Returns the generated text as a string.
        """
        try:
            def _sync_generate():
                model_obj = genai.GenerativeModel(
                    model_name=model,
                    system_instruction=system_instruction,
                )
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                
                # Configure safety settings to allow legal content analysis
                safety_settings = {
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
                
                response = model_obj.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
                if not response or not response.text:
                    raise GoogleInferenceError("Google API returned empty response.")
                return response.text.strip()
            
            return await asyncio.to_thread(_sync_generate)
        except Exception as e:
            raise GoogleInferenceError(f"Google API error: {e}")

    async def generate_json(
        self,
        *,
        model: str = "models/gemini-2.5-pro",
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 8192,
        system_instruction: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate JSON using Google Gemini API with JSON mode.
        Supports JSON Schema for structured outputs (Gemini 1.5+).
        Returns a parsed JSON object.
        """
        try:
            def _sync_generate():
                # Use JSON mode if available (Gemini 1.5 Pro supports it)
                system_instruction_json = (system_instruction or "") + "\n\nIMPORTANT: Return ONLY valid JSON. No markdown, no code blocks, no explanations outside the JSON."
                model_obj = genai.GenerativeModel(
                    model_name=model,
                    system_instruction=system_instruction_json,
                )
                
                # Build generation config with JSON Schema if provided
                config_kwargs = {
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "response_mime_type": "application/json",
                }
                # Add JSON Schema if provided (for structured outputs)
                if json_schema:
                    config_kwargs["response_schema"] = json_schema
                
                generation_config = genai.types.GenerationConfig(**config_kwargs)
                
                # Configure safety settings to allow legal content analysis
                # Legal questions about Supreme Court cases can trigger default safety filters
                safety_settings = {
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
                
                response = model_obj.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
                if not response or not response.text:
                    raise GoogleInferenceError("Google API returned empty response.")
                text = response.text.strip()
                # Remove markdown code blocks if present (fallback)
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                return json.loads(text)
            
            return await asyncio.to_thread(_sync_generate)
        except json.JSONDecodeError as e:
            raise GoogleInferenceError(f"Google API returned invalid JSON: {e}")
        except Exception as e:
            raise GoogleInferenceError(f"Google API error: {e}")

    async def embed_text(
        self,
        *,
        model: str = "models/text-embedding-004",
        text: str,
    ) -> List[float]:
        """
        Generate embeddings using Google's embedding model.
        Returns a list of floats (embedding vector).
        """
        try:
            def _sync_embed():
                result = genai.embed_content(
                    model=model,
                    content=text,
                )
                if not result or "embedding" not in result:
                    raise GoogleInferenceError("Google embedding API returned empty response.")
                embedding = result["embedding"]
                if not isinstance(embedding, list):
                    raise GoogleInferenceError(f"Google embedding API returned unexpected type: {type(embedding)}")
                return [float(x) for x in embedding]
            
            return await asyncio.to_thread(_sync_embed)
        except Exception as e:
            raise GoogleInferenceError(f"Google embedding API error: {e}")

