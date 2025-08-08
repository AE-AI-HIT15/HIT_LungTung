from __future__ import annotations

import base64
import io
from functools import cached_property

import torch
from deep_translator import GoogleTranslator
from diffusers import StableDiffusionXLPipeline
from langdetect import detect

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

    def check_and_translate_prompt(self, prompt: str) -> str:
        """ Check the language of the prompt and translate it if necessary.

        Args:
            prompt (str): The input text prompt to be checked and potentially translated.

        Returns:
            str: The original prompt if it's in English, or the translated prompt if it's in Vietnamese.
        """
        lang = detect(prompt)
        if lang == 'vi':
            translated = GoogleTranslator(
                source='vi', target='en',
            ).translate(prompt)
            return translated
        else:
            return prompt

    def process(self, inputs: Text2ImageInput) -> Text2ImageOutput:
        """ Process the text-to-image generation request.

        Args:
            inputs (Text2ImageInput): The input data containing the prompt.

        Returns:
            Text2ImageOutput: The output containing the generated image in base64 format.
        """
        logger.info(f'Processing T2I request with prompt: {inputs.prompt}')
        pipe = self.model_loaded
        prompt = self.check_and_translate_prompt(inputs.prompt)
        image = pipe(prompt=prompt).images[0]
        buffered = io.BytesIO()
        image.save(buffered, format='PNG')
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return Text2ImageOutput(image=image_base64)
