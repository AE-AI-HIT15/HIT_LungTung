from __future__ import annotations

from shared.base import BaseModel


class T2IInput(BaseModel):
    prompt: str


class T2IOutput(BaseModel):
    image_base64: str
