import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from dtrocr.config import DTrOCRConfig
from dtrocr.model import DTrOCRLMHeadModel
from dtrocr.processor import DTrOCRProcessor


class FakeTokenizer:
    bos_token = "<bos>"
    eos_token = "<eos>"
    pad_token = "<pad>"
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    model_max_length = 8
    add_bos_token = False
    add_eos_token = False

    def __call__(self, texts, padding=False, max_length=None, truncation=False, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        ids = []
        masks = []
        for text in texts:
            seq = [10 + (ord(char) % 10) for char in text][: max_length or 8]
            if self.add_bos_token:
                seq = [self.bos_token_id] + seq
            if self.add_eos_token:
                seq = seq + [self.eos_token_id]
            ids.append(seq)
            masks.append([1] * len(seq))
        max_len = max(len(seq) for seq in ids)
        ids = [seq + [self.pad_token_id] * (max_len - len(seq)) for seq in ids]
        masks = [mask + [0] * (max_len - len(mask)) for mask in masks]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["x" for _ in ids]


class FakeImageProcessor:
    def __call__(self, images, input_data_format="channels_last", return_tensors=None):
        if isinstance(images, Image.Image):
            images = [images]
        return {"pixel_values": torch.zeros(len(images), 3, 4, 4)}


class DummyPretrained:
    def __init__(self, config):
        self.h = torch.nn.ModuleList([GPT2Block(config, layer_idx=0)])
        self.wte = torch.nn.Embedding(config.vocab_size, config.hidden_size)


class DTrOCRLightweightTests(unittest.TestCase):
    def setUp(self):
        self.config = DTrOCRConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            vocab_size=32,
            max_position_embeddings=16,
            patch_size=(2, 2),
            image_size=(4, 4),
        )

    def test_config_roundtrip(self):
        payload = self.config.to_dict()
        self.assertEqual(payload, DTrOCRConfig.from_dict(payload).to_dict())

    def test_processor_helpers(self):
        processor = DTrOCRProcessor(
            self.config,
            add_bos_token=True,
            add_eos_token=True,
            vit_processor=FakeImageProcessor(),
            tokeniser=FakeTokenizer(),
        )
        encoded = processor.encode_sample(Image.new("RGB", (4, 4)), "ab", 6)
        gen = processor.build_generation_inputs(torch.zeros(2, 3, 4, 4))
        self.assertEqual(tuple(encoded.pixel_values.shape), (3, 4, 4))
        self.assertEqual(tuple(encoded.input_ids.shape), (2,))
        self.assertEqual(tuple(gen.pixel_values.shape), (2, 3, 4, 4))
        self.assertEqual(tuple(gen.input_ids.shape), (2, 1))

    def test_forward_and_generate(self):
        processor = DTrOCRProcessor(
            self.config,
            add_bos_token=True,
            add_eos_token=True,
            vit_processor=FakeImageProcessor(),
            tokeniser=FakeTokenizer(),
        )
        with patch("dtrocr.model.GPT2Model.from_pretrained", side_effect=lambda *args, **kwargs: DummyPretrained(self.config)):
            model = DTrOCRLMHeadModel(self.config)
        batch = processor.build_generation_inputs(torch.zeros(1, 3, 4, 4))
        batch.input_ids = torch.tensor([[1, 3]])
        batch.attention_mask = torch.tensor([[1, 1]])
        outputs = model(pixel_values=batch.pixel_values, input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        generated = model.generate(inputs=batch, processor=processor, num_beams=2, use_cache=False)
        self.assertEqual(tuple(outputs.logits.shape), (1, 6, 32))
        self.assertEqual(tuple(generated.shape), (1, 8))


if __name__ == "__main__":
    unittest.main()
