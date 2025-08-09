from __future__ import annotations

import gc
import unittest

import torch

from infra.load_model.load_model import Text2ImageInput
from infra.load_model.load_model import Text2ImageService
from shared.settings import Settings
gc.collect()
torch.cuda.empty_cache()


class TestText2Image(unittest.TestCase):

    def setUp(self):
        self.settings = Settings()
        self.t2i = Text2ImageService(settings=self.settings)

    def test_t2i_service(self):
        result = self.t2i.process(
            inputs=Text2ImageInput(
                prompt='A beautiful sunset over the mountains',
            ),
        )
        print(result)


if __name__ == '__main__':
    unittest.main()
