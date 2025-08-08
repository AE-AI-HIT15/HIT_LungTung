from __future__ import annotations

import base64
import io
from functools import cached_property

import torch
from diffusers import StableDiffusionXLPipeline

from shared.base import BaseModel
from shared.logging import get_logger
from shared.settings import Settings

logger = get_logger(__name__)


class Text2ImageInput(BaseModel):
    prompt: str


class Text2ImageOutput(BaseModel):
    image: str


class Text2ImageService(BaseModel):
    settings: Settings

    @cached_property
    def model_loaded(self):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.settings.t2i.base_model_id,
            torch_dtype=torch.float16,
            variant='fp16',
        ).to('cuda')
        if self.settings.t2i.lora_weights:
            pipe.load_lora_weights(self.settings.t2i.lora_weights)
        return pipe

    def process(self, inputs: Text2ImageInput) -> Text2ImageOutput:
        """ Process the text-to-image generation request.

        Args:
            inputs (Text2ImageInput): The input data containing the prompt.

        Returns:
            Text2ImageOutput: The output containing the generated image in base64 format.
        """
        logger.info(f'Processing T2I request with prompt: {inputs.prompt}')
        pipe = self.model_loaded
        image = pipe(prompt=inputs.prompt).images[0]
        buffered = io.BytesIO()
        image.save(buffered, format='PNG')
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return Text2ImageOutput(image=image_base64)
