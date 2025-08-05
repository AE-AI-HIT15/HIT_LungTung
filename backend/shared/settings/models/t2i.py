from shared.base import BaseModel

class Text2ImageSettings(BaseModel):
    base_model_id: str
    lora_weights: str
