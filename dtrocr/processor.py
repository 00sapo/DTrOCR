from transformers import GPT2Tokenizer, AutoImageProcessor

from PIL import Image
from typing import List, Union
from config import DTrOCRConfig
from data import DTrOCRProcessorOutput


class DTrOCRProcessor:
    def __init__(self, config: DTrOCRConfig, add_bos_token: bool = False, add_eos_token: bool = False):
        self.vit_processor = AutoImageProcessor.from_pretrained(
            config.vit_hf_model,
            size={
                "height": config.image_size[0],
                'width': config.image_size[1]
            },
            use_fast=True
        )
        self.tokeniser = GPT2Tokenizer.from_pretrained(
            config.gpt2_hf_model,
            add_bos_token=add_bos_token,
            model_max_length=config.max_position_embeddings - int(
                (config.image_size[0] / config.patch_size[0]) * (config.image_size[1] / config.patch_size[1])
            )
        )
        self.tokeniser.pad_token = self.tokeniser.bos_token
        self.tokeniser.add_eos_token = add_eos_token

        # Bind a new method to gpt2_tokeniser
        self.tokeniser.build_inputs_with_special_tokens = modified_build_inputs_with_special_tokens.__get__(
            self.tokeniser
        )

    def encode_sample(
        self,
        image: Image.Image,
        text: str,
        max_target_length: int,
        input_data_format: str = 'channels_last',
    ) -> DTrOCRProcessorOutput:
        text_inputs = self.tokeniser(
            text,
            padding="max_length",
            max_length=max_target_length,
            truncation=True,
            return_tensors="pt",
        )
        image_inputs = self.vit_processor(
            image,
            input_data_format=input_data_format,
            return_tensors="pt",
        )
        return DTrOCRProcessorOutput(
            pixel_values=image_inputs["pixel_values"].squeeze(0),
            input_ids=text_inputs["input_ids"].squeeze(0),
            attention_mask=text_inputs["attention_mask"].squeeze(0),
            labels=text_inputs["input_ids"].squeeze(0),
        )

    def build_generation_inputs(
        self,
        pixel_values,
        input_ids=None,
        attention_mask=None,
    ) -> DTrOCRProcessorOutput:
        import torch

        bos_token_id = self.tokeniser.bos_token_id
        if bos_token_id is None:
            bos_token_id = self.tokeniser.eos_token_id
        if bos_token_id is None:
            bos_token_id = self.tokeniser.pad_token_id

        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        dtype = input_ids.dtype if input_ids is not None else torch.long
        if input_ids is None:
            input_ids = torch.full((batch_size, 1), bos_token_id, dtype=dtype, device=device)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, 1), dtype=torch.long, device=device)

        return DTrOCRProcessorOutput(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
        )

    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]] = None,
        texts: Union[str, List[str]] = None,
        return_labels: bool = False,
        input_data_format: str = 'channels_last',
        padding: Union[bool, str] = False,
        *args,
        **kwargs
    ) -> DTrOCRProcessorOutput:
        text_inputs = self.tokeniser(
            texts, padding=padding, *args, **kwargs
        ) if texts is not None else None

        image_inputs = self.vit_processor(
            images, input_data_format=input_data_format, *args, **kwargs
        ) if images is not None else None

        return DTrOCRProcessorOutput(
            pixel_values=image_inputs["pixel_values"] if images is not None else None,
            input_ids=text_inputs['input_ids'] if texts is not None else None,
            attention_mask=text_inputs['attention_mask'] if texts is not None else None,
            labels=text_inputs['input_ids'] if texts is not None and return_labels else None
        )


def modified_build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    if self.add_bos_token:
        bos_token_ids = [self.bos_token_id]
    else:
        bos_token_ids = []

    if self.add_eos_token:
        eos_token_ids = [self.eos_token_id]
    else:
        eos_token_ids = []

    output = bos_token_ids + token_ids_0 + eos_token_ids

    if token_ids_1 is None:
        return output

    return output + bos_token_ids + token_ids_1
