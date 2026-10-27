"""Fun-Audio-Chat text-only generation wrapper for audio QA.

Fun-Audio-Chat ships its modelling code in its own repository (``funaudiochat``
package) rather than depending on a transformers release, and registers the
classes with the Auto* APIs at import time. That repository is vendored under
``third_party/Fun-Audio-Chat`` (or pointed at by ``FUN_AUDIO_CHAT_ROOT``).

The reference speech-to-text path is ``examples/infer_s2t.py``: a system prompt
asking for text, the audio template concatenated with the user instruction, and
``text_greedy``/``disable_speech`` set so generation is a deterministic text-only
greedy decode.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .base import AudioLanguageModel

REPO_ROOT = Path(__file__).resolve().parents[2]


def _fun_audio_chat_root() -> Path:
    override = os.environ.get("FUN_AUDIO_CHAT_ROOT")
    if override:
        return Path(override)
    return REPO_ROOT / "third_party" / "Fun-Audio-Chat"


class FunAudioChat(AudioLanguageModel):
    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        *,
        max_new_tokens: int = 128,
        device: str = "cuda:0",
    ) -> None:
        root = _fun_audio_chat_root()
        if not (root / "funaudiochat").is_dir():
            raise FileNotFoundError(
                f"Fun-Audio-Chat code not found under {root}; set FUN_AUDIO_CHAT_ROOT"
            )
        # The repo provides both the ``funaudiochat`` package and the ``utils``
        # constants module, so its root goes on the path.
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        import librosa  # noqa: F401  (imported for side effect of availability check)
        import torch
        from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor

        from funaudiochat.register import register_funaudiochat
        from utils.constant import AUDIO_TEMPLATE, DEFAULT_S2T_PROMPT

        register_funaudiochat()

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        self._librosa = librosa
        self._audio_template = AUDIO_TEMPLATE
        self._system_prompt = DEFAULT_S2T_PROMPT

        self._config = AutoConfig.from_pretrained(model_id)
        self.revision = revision or getattr(self._config, "_commit_hash", None) or "unknown"
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            config=self._config,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        # Text-only deterministic decode: greedy text tokens, no speech tokens.
        self._model.sp_gen_kwargs.update({"text_greedy": True, "disable_speech": True})
        self._model.eval()

    def generate(self, audio: str | Path, question: str) -> str:
        waveform = self._librosa.load(str(Path(audio).resolve()), sr=16000)[0]
        conversation = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": self._audio_template + "\n" + question},
        ]
        text = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self._processor(
            text=text,
            audio=[waveform],
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self._model.device)
        with self._torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        # The reference path returns (generate_ids, speech_output); accept either
        # a bare tensor or the tuple so a wrapper change cannot silently break.
        generate_ids = output[0] if isinstance(output, (tuple, list)) else output
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        decoded = self._processor.decode(generate_ids[0], skip_special_tokens=True)
        return decoded.strip()
