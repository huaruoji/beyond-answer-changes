"""Transformers wrapper for NVIDIA Audio Flamingo 3 text generation."""

from __future__ import annotations

from pathlib import Path

from .base import AudioLanguageModel


class AudioFlamingo3(AudioLanguageModel):
    def __init__(
        self,
        model_id: str = "nvidia/audio-flamingo-3-hf",
        revision: str | None = "7d4bae64ee29878af6504ae6f6bb3e40492838ad",
        *,
        max_new_tokens: int = 128,
        device_map: str = "auto",
    ) -> None:
        import torch
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        load_kwargs = {"revision": revision} if revision else {}
        self._processor = AutoProcessor.from_pretrained(model_id, **load_kwargs)
        self._model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id,
            **load_kwargs,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self._model.eval()

    def generate(self, audio: str | Path, question: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "audio", "path": str(Path(audio).resolve())},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self._model.device)
        # AF3's processor emits float32 log-mel features while the BF16 model
        # has BF16 convolution weights. Cast only floating-point features;
        # attention masks and token IDs must retain their integer dtypes.
        for key, value in inputs.items():
            if self._torch.is_floating_point(value):
                inputs[key] = value.to(dtype=self._model.dtype)
        with self._torch.inference_mode():
            outputs = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated = outputs[:, inputs.input_ids.shape[1] :]
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
