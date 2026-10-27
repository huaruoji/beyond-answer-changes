"""Qwen2.5-Omni text-only generation wrapper for audio QA."""

from __future__ import annotations

from pathlib import Path

from .base import AudioLanguageModel


class Qwen25Omni(AudioLanguageModel):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Omni-3B",
        revision: str | None = "f75b40e3da2003cdd6e1829b1f420ca70797c34e",
        *,
        max_new_tokens: int = 128,
        device_map: str = "auto",
    ) -> None:
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        load_kwargs = {"revision": revision} if revision else {}
        self._processor = Qwen2_5OmniProcessor.from_pretrained(model_id, **load_kwargs)
        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id,
            **load_kwargs,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        # Text output is the experimental target and keeping the talker wastes VRAM.
        self._model.disable_talker()
        self._model.eval()

    def generate(self, audio: str | Path, question: str) -> str:
        from qwen_omni_utils import process_mm_info

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(Path(audio).resolve())},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self._processor(
            text=prompt,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        # Do not cast the complete BatchFeature: input_ids and masks must stay
        # integer/bool tensors.  The model handles floating-point audio
        # features internally after device placement.
        inputs = inputs.to(self._model.device)
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                use_audio_in_video=False,
                return_audio=False,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        # Qwen returns prompt + completion token IDs.  Decode only newly
        # generated tokens; otherwise the evaluator can parse "Answer A or B"
        # from the echoed user prompt instead of the model's actual answer.
        generated = generated[:, inputs["input_ids"].shape[1] :]
        decoded = self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0].strip()
