import os
import json
import math
import random
import requests
import numpy as np
import re


def _extract_first_json(text: str):
    # attempt to extract the first JSON object from text
    text = text.strip()
    # quick heuristic: find first { ... }
    start = text.find('{')
    if start == -1:
        return None
    # try to find matching closing brace
    stack = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            stack += 1
        elif text[i] == '}':
            stack -= 1
            if stack == 0:
                try:
                    return json.loads(text[start:i+1])
                except Exception:
                    return None
    return None


class LlamaAgent:
    """
    Wrapper for a Llama-based agent. If environment variables `LLAMA_API_URL` and
    `LLAMA_API_KEY` are set the wrapper will attempt to call that endpoint using
    an OpenAI-style request (model + input) and will robustly parse JSON output.

    Environment variables supported:
    - LLAMA_API_URL: URL of the LLM endpoint
    - LLAMA_API_KEY: Bearer API key
    - LLAMA_MODEL: model name to request (optional)
    - LLAMA_STEP_SIZE: max step size (meters) agents may move in one propose
    """

    def __init__(self, name, api_url=None, api_key=None, model=None):
        self.name = name
        self.api_url = api_url or os.getenv('LLAMA_API_URL')
        self.api_key = api_key or os.getenv('LLAMA_API_KEY')
        self.model = model or os.getenv('LLAMA_MODEL')
        try:
            self.max_step = float(os.getenv('LLAMA_STEP_SIZE', '10.0'))
        except Exception:
            self.max_step = 10.0

    def has_api_interface(self):
        return bool(self.api_url and self.api_key)

    def _call_api(self, prompt: str, timeout=10):
        if not (self.api_url and self.api_key):
            return None
        payload = {}
        # try to be compatible with OpenAI Responses API and generic gateways
        if self.model:
            payload['model'] = self.model
        # prefer `input` for some gateways, `prompt` for others
        payload['input'] = prompt
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        try:
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    def _parse_response_for_pos(self, resp_json):
        if resp_json is None:
            return None
        # common response shapes
        # 1) {'output': '...'}
        if isinstance(resp_json, dict):
            for key in ('output', 'text', 'result'):
                if key in resp_json and isinstance(resp_json[key], str):
                    obj = _extract_first_json(resp_json[key])
                    if obj and 'pos' in obj:
                        return obj['pos']
            # 2) OpenAI-style choices
            if 'choices' in resp_json and isinstance(resp_json['choices'], list) and len(resp_json['choices']) > 0:
                c = resp_json['choices'][0]
                # nested message.content
                content = None
                if isinstance(c, dict):
                    if 'delta' in c and isinstance(c['delta'], dict):
                        content = c['delta'].get('content')
                    if 'message' in c and isinstance(c['message'], dict):
                        content = c['message'].get('content') or content
                    content = content or c.get('text') or c.get('content')
                if isinstance(content, str):
                    obj = _extract_first_json(content)
                    if obj and 'pos' in obj:
                        return obj['pos']
        return None

    def _validate_and_clamp(self, uav, proposed_pos):
        try:
            pos = np.array(proposed_pos, dtype=float)
            if pos.shape[0] != 3:
                return None
        except Exception:
            return None
        # clamp step size
        delta = pos - uav.pos
        dist = np.linalg.norm(delta)
        if dist > self.max_step:
            pos = uav.pos + (delta / dist) * self.max_step
        # enforce a minimum altitude (e.g., 5m)
        if pos[2] < 5.0:
            pos[2] = 5.0
        return tuple(float(x) for x in pos)

    def propose(self, uav, antennas, context=None):
        """Return a candidate new 3D position for the UAV (tuple)

        If an API is configured the agent will send a short prompt asking for a
        JSON reply with a `pos` field (list of 3 numbers). If parsing fails we
        fall back to the local heuristic.
        """
        # Build prompt
        ant_list = [list(a.pos) for a in antennas]
        prompt = (
            f"You are an agent proposing the next 3D waypoint for UAV '{uav.uid}'.\n"
            f"Current position: {list(uav.pos)}\n"
            f"Antennas: {ant_list}\n"
            "Return a single JSON object with a 'pos' key containing [x,y,z].\n"
            "Do not include any extra text outside the JSON object.\n"
            "Example: {\"pos\": [123.4, 56.7, 50.0]}\n"
        )

        resp = self._call_api(prompt)
        pos = self._parse_response_for_pos(resp)
        pos = self._validate_and_clamp(uav, pos) if pos is not None else None
        if pos is not None:
            return pos

        # Fallback heuristic: move toward the nearest antenna with small random noise
        best_a = None
        best_dist = float('inf')
        for a in antennas:
            d = np.linalg.norm(uav.pos - a.pos)
            if d < best_dist:
                best_dist = d
                best_a = a
        step = min(self.max_step, 5.0)
        dir_vec = (best_a.pos - uav.pos)
        if np.linalg.norm(dir_vec) < 1e-6:
            dir_vec = np.random.randn(3)
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        noise = np.random.randn(3) * 0.5
        new_pos = uav.pos + dir_vec * step + noise
        # ensure altitude
        if new_pos[2] < 5.0:
            new_pos[2] = 5.0
        return tuple(float(x) for x in new_pos)

