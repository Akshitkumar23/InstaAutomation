#!/usr/bin/env python3
"""
Model runtime adapters for Instagram AI Agent.

Provides pluggable providers for text, embeddings, and image generation.
All providers are optional and fall back to safe template behavior.
"""

from __future__ import annotations

import json
import hashlib
import logging
import math
import random
import re
import shutil
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


@dataclass
class TextGenerationResult:
    caption: str
    hashtags: List[str]


class TextProvider:
    """Interface for text generation providers."""

    def generate(self, topic: str, constraints: Dict) -> Optional[TextGenerationResult]:
        raise NotImplementedError


class TemplateTextProvider(TextProvider):
    """Fallback text provider using deterministic templates."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def generate(self, topic: str, constraints: Dict) -> Optional[TextGenerationResult]:
        prefs = constraints.get("content_preferences", {})
        preferred_emojis = prefs.get("preferred_emojis", [])
        emoji = self._rng.choice(preferred_emojis) if preferred_emojis else "*"

        templates = [
            (
                f"{emoji} {topic}! Here are 3 practical ideas you can try today:\n\n"
                "1. Start with one small step\n"
                "2. Keep it simple and repeatable\n"
                "3. Track progress for 7 days\n\n"
                "Save this post if it helps."
            ),
            (
                f"{emoji} Quick breakdown: {topic}\n\n"
                "Why it matters:\n"
                "- Clarity\n"
                "- Consistency\n"
                "- Compounding results\n\n"
                "Which point will you try first?"
            ),
            (
                f"{emoji} {topic} made simple:\n\n"
                "1) Identify the bottleneck\n"
                "2) Build a tiny routine\n"
                "3) Review weekly\n\n"
                "Share this with someone who needs it."
            ),
        ]

        caption = self._rng.choice(templates)
        caption = _trim_words(caption, prefs.get("max_caption_length", 150))

        hashtags_seed = _generate_hashtags(topic=topic)
        hashtags = _finalize_hashtags(
            hashtags_seed,
            min_hashtags=prefs.get("min_hashtags", 15),
            max_hashtags=prefs.get("max_hashtags", 25),
            banned=prefs.get("banned_hashtags", []),
        )

        return TextGenerationResult(caption=caption, hashtags=hashtags)


class TransformersTextProvider(TextProvider):
    """Text provider backed by a local HuggingFace Transformers model."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_new_tokens: int = 220,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("Transformers provider requires torch and transformers") from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model_kwargs = {"trust_remote_code": True}
        if self.device == "auto":
            model_kwargs["device_map"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        if self.device not in ("auto", None):
            self._model.to(self.device)
        if hasattr(self._model, "eval"):
            self._model.eval()

    def generate(self, topic: str, constraints: Dict) -> Optional[TextGenerationResult]:
        self._ensure_loaded()
        prefs = constraints.get("content_preferences", {})
        min_hashtags = prefs.get("min_hashtags", 15)
        max_hashtags = prefs.get("max_hashtags", 25)
        banned = prefs.get("banned_hashtags", [])
        max_words = prefs.get("max_caption_length", 150)

        prompt = (
            "You are an Instagram copywriter. Output exactly two lines:\n"
            "CAPTION: <caption text>\n"
            "HASHTAGS: <space-separated hashtags>\n"
            f"Constraints: caption max {max_words} words; hashtags {min_hashtags}-{max_hashtags}; "
            f"avoid banned hashtags {banned}.\n"
            f"Topic: {topic}\n"
        )

        input_ids = self._tokenizer(prompt, return_tensors="pt")
        if hasattr(self._model, "device"):
            input_ids = {k: v.to(self._model.device) for k, v in input_ids.items()}

        output = self._model.generate(
            **input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        decoded = self._tokenizer.decode(output[0], skip_special_tokens=True)

        caption, hashtags = _parse_caption_and_hashtags(decoded)
        if not caption:
            return None

        caption = _trim_words(caption, max_words)
        hashtags = _finalize_hashtags(hashtags, min_hashtags, max_hashtags, banned)
        return TextGenerationResult(caption=caption, hashtags=hashtags)


class EmbeddingProvider:
    """Interface for embedding providers."""

    def is_too_similar(self, topic: str, recent_topics: List[str], threshold: float) -> bool:
        raise NotImplementedError


class SimpleSimilarityProvider(EmbeddingProvider):
    """Basic string similarity fallback."""

    def is_too_similar(self, topic: str, recent_topics: List[str], threshold: float) -> bool:
        topic_lower = topic.lower()
        for recent in recent_topics:
            recent_lower = recent.lower()
            if topic_lower in recent_lower or recent_lower in topic_lower:
                return True
        return False


class SentenceTransformersProvider(EmbeddingProvider):
    """Embedding provider backed by sentence-transformers."""

    def __init__(self, model_path: str, device: str = "auto") -> None:
        self.model_path = model_path
        self.device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError("sentence-transformers is required for embeddings") from exc
        self._model = SentenceTransformer(self.model_path, device=None if self.device == "auto" else self.device)

    def is_too_similar(self, topic: str, recent_topics: List[str], threshold: float) -> bool:
        if not recent_topics:
            return False
        self._ensure_loaded()
        texts = [topic] + recent_topics
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        topic_vec = embeddings[0]
        for vec in embeddings[1:]:
            similarity = float(_dot(topic_vec, vec))
            if similarity >= threshold:
                return True
        return False


class ImageProvider:
    """Interface for image generation providers."""

    def generate(self, prompt: str, output_dir: str, size: Tuple[int, int]) -> Optional[str]:
        raise NotImplementedError


class PlaceholderImageProvider(ImageProvider):
    """Generate a placeholder image using Pillow if available."""

    def generate(self, prompt: str, output_dir: str, size: Tuple[int, int]) -> Optional[str]:
        output_path = _next_image_path(output_dir)
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception as exc:
            logger.warning("Pillow not installed. Unable to generate placeholder image: %s", exc)
            _write_prompt_file(output_path.with_suffix(".txt"), prompt)
            return None

        width, height = size if size and size[0] and size[1] else (1080, 1080)
        topic = _extract_topic_from_prompt(prompt)
        if not topic:
            topic = "Daily Insight"

        rng = random.Random(_stable_hash(prompt))
        palette = _pick_palette(rng)
        image = _make_gradient((width, height), palette[0], palette[1])

        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        accent = palette[2]
        for _ in range(4):
            radius = rng.randint(int(width * 0.08), int(width * 0.18))
            cx = rng.randint(0, width)
            cy = rng.randint(0, height)
            color = (*accent, rng.randint(30, 70))
            overlay_draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)

        for _ in range(3):
            x1 = rng.randint(0, width)
            y1 = rng.randint(0, height)
            x2 = rng.randint(0, width)
            y2 = rng.randint(0, height)
            color = (*accent, rng.randint(40, 80))
            overlay_draw.line((x1, y1, x2, y2), fill=color, width=rng.randint(3, 8))

        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(image)

        margin = int(min(width, height) * 0.08)
        max_text_width = width - (margin * 2)
        title = _title_case(topic)

        title_font = _fit_font(draw, title, max_text_width, int(height * 0.09), int(height * 0.045), bold=True)
        lines = _wrap_text_to_width(draw, title, title_font, max_text_width)

        while len(lines) > 3:
            new_size = max(int(_font_size(title_font) * 0.9), int(height * 0.045))
            title_font = _load_font(new_size, bold=True)
            lines = _wrap_text_to_width(draw, title, title_font, max_text_width)
            if new_size <= int(height * 0.045):
                break

        line_height = int(_font_size(title_font) * 1.25)
        text_height = line_height * len(lines)
        text_width = max(_text_width(draw, line, title_font) for line in lines) if lines else 0

        block_top = int((height - text_height) * 0.52)
        block_top = max(margin, min(block_top, height - text_height - margin))
        block_left = margin

        is_dark = _is_dark_color(palette[1])
        text_color = (245, 245, 245) if is_dark else (20, 20, 20)
        box_color = (0, 0, 0, 120) if is_dark else (255, 255, 255, 140)

        box_padding_x = int(margin * 0.6)
        box_padding_y = int(_font_size(title_font) * 0.6)
        box_left = block_left - box_padding_x
        box_top = block_top - box_padding_y
        box_right = block_left + text_width + box_padding_x
        box_bottom = block_top + text_height + box_padding_y

        box_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        box_draw = ImageDraw.Draw(box_overlay)
        if hasattr(box_draw, "rounded_rectangle"):
            box_draw.rounded_rectangle(
                (box_left, box_top, box_right, box_bottom),
                radius=int(box_padding_y * 0.6),
                fill=box_color,
            )
        else:
            box_draw.rectangle((box_left, box_top, box_right, box_bottom), fill=box_color)
        image = Image.alpha_composite(image.convert("RGBA"), box_overlay).convert("RGB")
        draw = ImageDraw.Draw(image)

        y = block_top
        for line in lines:
            draw.text((block_left, y), line, fill=text_color, font=title_font)
            y += line_height

        image.save(output_path)
        return str(output_path)


class ComfyUIImageProvider(ImageProvider):
    """Generate images via a running ComfyUI server."""

    def __init__(
        self,
        server_url: str,
        workflow_path: str,
        checkpoint_name: Optional[str] = None,
        prompt_node_id: Optional[str] = None,
        prompt_field: str = "text",
        negative_prompt: str = "",
        input_dir: Optional[str] = None,
        reference_image: Optional[str] = None,
        pose_dir: Optional[str] = None,
        pose_history_path: Optional[str] = None,
        reference_node_id: Optional[str] = None,
        pose_node_id: Optional[str] = None,
        pose_size: int = 512,
        timeout_sec: int = 300,
        poll_interval_sec: int = 2,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.workflow_path = workflow_path
        self.checkpoint_name = checkpoint_name
        self.prompt_node_id = prompt_node_id
        self.prompt_field = prompt_field
        self.negative_prompt = negative_prompt
        self.input_dir = input_dir
        self.reference_image = reference_image
        self.pose_dir = pose_dir
        self.pose_history_path = pose_history_path
        self.reference_node_id = reference_node_id
        self.pose_node_id = pose_node_id
        self.pose_size = pose_size
        self.timeout_sec = timeout_sec
        self.poll_interval_sec = poll_interval_sec

    def generate(self, prompt: str, output_dir: str, size: Tuple[int, int]) -> Optional[str]:
        workflow = _load_workflow(self.workflow_path)
        if workflow is None:
            return None

        if not _prepare_comfyui_inputs(
            workflow=workflow,
            input_dir=self.input_dir,
            reference_image=self.reference_image,
            pose_dir=self.pose_dir,
            pose_history_path=self.pose_history_path,
            reference_node_id=self.reference_node_id,
            pose_node_id=self.pose_node_id,
            pose_size=self.pose_size,
        ):
            logger.error("ComfyUI input preparation failed")
            return None

        if self.checkpoint_name:
            if not _inject_checkpoint_name(workflow, self.checkpoint_name):
                logger.warning("ComfyUI workflow missing CheckpointLoaderSimple; checkpoint_name not applied")

        seed = random.randint(1, 2**31 - 1)
        if not _inject_seed(workflow, seed):
            logger.warning("ComfyUI workflow missing KSampler; seed not applied")

        if self.prompt_node_id:
            node = workflow.get(self.prompt_node_id)
            if not node:
                logger.error("ComfyUI prompt node id %s not found", self.prompt_node_id)
                return None
            node.setdefault("inputs", {})[self.prompt_field] = prompt
        else:
            if not _inject_prompt_into_workflow(workflow, prompt, self.prompt_field):
                logger.error("Unable to inject prompt into ComfyUI workflow")
                return None

        if self.negative_prompt:
            _inject_negative_prompt(workflow, self.negative_prompt)

        try:
            response = requests.post(
                f"{self.server_url}/prompt",
                json={"prompt": workflow},
                timeout=30,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.error("ComfyUI prompt request failed: %s", exc)
            return None

        data = response.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            logger.error("ComfyUI did not return a prompt_id")
            return None

        deadline = time.time() + self.timeout_sec
        while time.time() < deadline:
            try:
                history = requests.get(
                    f"{self.server_url}/history/{prompt_id}",
                    timeout=30,
                )
                history.raise_for_status()
            except Exception:
                time.sleep(self.poll_interval_sec)
                continue

            payload = history.json().get(prompt_id)
            if not payload:
                time.sleep(self.poll_interval_sec)
                continue

            outputs = payload.get("outputs", {})
            for node_output in outputs.values():
                images = node_output.get("images") or []
                if not images:
                    continue
                image_info = images[0]
                filename = image_info.get("filename")
                if not filename:
                    continue

                subfolder = image_info.get("subfolder", "")
                image_type = image_info.get("type", "output")
                image_bytes = _download_comfyui_image(self.server_url, filename, subfolder, image_type)
                if image_bytes is None:
                    return None

                output_path = _next_image_path(output_dir, suffix=Path(filename).suffix or ".png")
                try:
                    from PIL import Image
                    image = Image.open(BytesIO(image_bytes))
                    if image.mode not in ("RGB", "RGBA"):
                        image = image.convert("RGB")
                    if size and (image.width, image.height) != size:
                        image = image.resize(size, resample=Image.LANCZOS)
                    image.save(output_path)
                except Exception as exc:
                    logger.warning("Failed to resize ComfyUI image, saving original bytes: %s", exc)
                    output_path.write_bytes(image_bytes)
                return str(output_path)

            time.sleep(self.poll_interval_sec)

        logger.error("Timed out waiting for ComfyUI image generation")
        return None


class ModelRuntime:
    """Factory and helper for model providers."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.template_text_provider = TemplateTextProvider()
        self.text_provider = _build_text_provider(config)
        self.embedding_provider = _build_embedding_provider(config)
        self.image_provider = _build_image_provider(config)

    def generate_caption_and_hashtags(self, topic: str) -> TextGenerationResult:
        constraints = self.config
        result = None
        if self.text_provider:
            try:
                result = self.text_provider.generate(topic, constraints)
            except Exception as exc:
                logger.warning("Text provider failed, falling back to templates: %s", exc)
                result = None

        if not result:
            result = self.template_text_provider.generate(topic, constraints)

        # Ensure constraints are met even if provider returned partial data
        prefs = constraints.get("content_preferences", {})
        min_h = prefs.get("min_hashtags", 15)
        max_h = prefs.get("max_hashtags", 25)
        banned = prefs.get("banned_hashtags", [])
        hashtags = _finalize_hashtags(result.hashtags, min_h, max_h, banned)
        caption = _trim_words(result.caption, prefs.get("max_caption_length", 150))

        return TextGenerationResult(caption=caption, hashtags=hashtags)

    def is_too_similar(self, topic: str, recent_topics: List[str]) -> bool:
        threshold = self.config.get("model_providers", {}).get("embeddings", {}).get("similarity_threshold", 0.85)
        if not recent_topics:
            return False

        provider = self.embedding_provider
        if provider is None:
            provider = SimpleSimilarityProvider()

        try:
            return provider.is_too_similar(topic, recent_topics, float(threshold))
        except Exception as exc:
            logger.warning("Embedding provider failed, using simple similarity: %s", exc)
            return SimpleSimilarityProvider().is_too_similar(topic, recent_topics, float(threshold))

    def should_generate_images(self) -> bool:
        image_config = self.config.get("image_generation", {})
        publisher_config = self.config.get("publisher", {})
        return bool(image_config.get("enabled", False) or publisher_config.get("enabled", False))

    def generate_image(self, prompt: str) -> Optional[str]:
        if not self.image_provider:
            return None
        image_config = self.config.get("image_generation", {})
        output_dir = image_config.get("output_dir", "generated_images")
        output_dir = _resolve_relative_path(output_dir, self.config.get("__base_dir"))
        dimensions = image_config.get("dimensions", "1080x1080")
        size = _parse_dimensions(dimensions)
        result = self.image_provider.generate(prompt, output_dir, size)
        if result is None and not isinstance(self.image_provider, PlaceholderImageProvider):
            logger.warning("Image provider failed; using placeholder image instead.")
            return PlaceholderImageProvider().generate(prompt, output_dir, size)
        return result


# -----------------------------
# Helper functions
# -----------------------------

def _build_text_provider(config: Dict) -> Optional[TextProvider]:
    provider_cfg = config.get("model_providers", {}).get("text", {})
    provider_name = provider_cfg.get("provider", "template")

    if provider_name == "template":
        return None

    if provider_name == "transformers":
        model_path = provider_cfg.get("model_path")
        if not model_path:
            logger.warning("Transformers provider configured without model_path")
            return None
        return TransformersTextProvider(
            model_path=model_path,
            device=provider_cfg.get("device", "auto"),
            max_new_tokens=int(provider_cfg.get("max_new_tokens", 220)),
            temperature=float(provider_cfg.get("temperature", 0.7)),
            top_p=float(provider_cfg.get("top_p", 0.9)),
        )

    logger.warning("Unknown text provider '%s', falling back to templates", provider_name)
    return None


def _build_embedding_provider(config: Dict) -> Optional[EmbeddingProvider]:
    provider_cfg = config.get("model_providers", {}).get("embeddings", {})
    provider_name = provider_cfg.get("provider", "simple")

    if provider_name == "simple":
        return SimpleSimilarityProvider()

    if provider_name == "sentence_transformers":
        model_path = provider_cfg.get("model_path")
        if not model_path:
            logger.warning("Sentence-transformers provider configured without model_path")
            return None
        return SentenceTransformersProvider(model_path=model_path, device=provider_cfg.get("device", "auto"))

    logger.warning("Unknown embedding provider '%s', using simple similarity", provider_name)
    return SimpleSimilarityProvider()


def _build_image_provider(config: Dict) -> Optional[ImageProvider]:
    provider_cfg = config.get("model_providers", {}).get("image", {})
    provider_name = provider_cfg.get("provider", "placeholder")

    if provider_name == "placeholder":
        return PlaceholderImageProvider()

    if provider_name == "comfyui":
        comfy_cfg = provider_cfg.get("comfyui", {})
        workflow_path = comfy_cfg.get("workflow_path")
        if not workflow_path:
            logger.warning("ComfyUI provider configured without workflow_path")
            return None
        workflow_path = _resolve_relative_path(workflow_path, config.get("__base_dir"))
        input_dir = _resolve_relative_path(comfy_cfg.get("input_dir", ""), config.get("__base_dir"))
        reference_image = _resolve_relative_path(comfy_cfg.get("reference_image", ""), config.get("__base_dir"))
        pose_dir = _resolve_relative_path(comfy_cfg.get("pose_dir", ""), config.get("__base_dir"))
        pose_history_path = _resolve_relative_path(comfy_cfg.get("pose_history_path", ""), config.get("__base_dir"))
        return ComfyUIImageProvider(
            server_url=comfy_cfg.get("server_url", "http://127.0.0.1:8188"),
            workflow_path=workflow_path,
            checkpoint_name=comfy_cfg.get("checkpoint_name") or None,
            prompt_node_id=comfy_cfg.get("prompt_node_id") or None,
            prompt_field=comfy_cfg.get("prompt_field", "text"),
            negative_prompt=comfy_cfg.get("negative_prompt", ""),
            input_dir=input_dir or None,
            reference_image=reference_image or None,
            pose_dir=pose_dir or None,
            pose_history_path=pose_history_path or None,
            reference_node_id=comfy_cfg.get("reference_node_id"),
            pose_node_id=comfy_cfg.get("pose_node_id"),
            pose_size=int(comfy_cfg.get("pose_size", 512)),
            timeout_sec=int(comfy_cfg.get("timeout_sec", 300)),
            poll_interval_sec=int(comfy_cfg.get("poll_interval_sec", 2)),
        )

    logger.warning("Unknown image provider '%s', using placeholder", provider_name)
    return PlaceholderImageProvider()


def _parse_caption_and_hashtags(text: str) -> Tuple[Optional[str], List[str]]:
    caption = None
    hashtags_line = None
    for line in text.splitlines():
        line_stripped = line.strip()
        if line_stripped.lower().startswith("caption:"):
            caption = line_stripped.split(":", 1)[1].strip()
        elif line_stripped.lower().startswith("hashtags:"):
            hashtags_line = line_stripped.split(":", 1)[1].strip()

    hashtags = _extract_hashtags(hashtags_line or "")
    return caption, hashtags


def _extract_hashtags(text: str) -> List[str]:
    if not text:
        return []
    tags = []
    for token in text.replace(",", " ").split():
        if token.startswith("#"):
            tags.append(token)
    return tags


def _trim_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


DEFAULT_BASE_HASHTAGS = [
    "productivity",
    "motivation",
    "lifestyle",
    "success",
    "growth",
    "mindset",
    "habits",
    "goals",
    "inspiration",
    "selfimprovement",
    "focus",
    "learning",
    "discipline",
    "consistency",
    "positivity",
    "mindfulness",
    "worksmart",
    "personaldevelopment",
    "goalsetting",
    "timemanagement",
]


def _generate_hashtags(topic: str) -> List[str]:
    base_hashtags = DEFAULT_BASE_HASHTAGS
    topic_words = [word.strip("#") for word in topic.lower().split() if len(word) > 3]
    hashtags = [f"#{tag}" for tag in base_hashtags]
    hashtags.extend(f"#{word}" for word in topic_words)

    return hashtags


def _finalize_hashtags(
    hashtags: List[str],
    min_hashtags: int,
    max_hashtags: int,
    banned: List[str],
) -> List[str]:
    banned_set = {tag.lower().lstrip("#") for tag in banned}
    cleaned = []
    seen = set()
    for tag in hashtags:
        normalized = tag.lower().lstrip("#")
        if normalized in banned_set or not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(f"#{normalized}")

    if len(cleaned) < min_hashtags:
        filler_pool = [f"#{tag}" for tag in DEFAULT_BASE_HASHTAGS]
        for tag in filler_pool:
            normalized = tag.lower().lstrip("#")
            if normalized in banned_set or normalized in seen:
                continue
            cleaned.append(tag)
            seen.add(normalized)
            if len(cleaned) >= min_hashtags:
                break

    if len(cleaned) > max_hashtags:
        cleaned = cleaned[:max_hashtags]

    return cleaned


def _dot(vec_a, vec_b) -> float:
    return sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))


def _parse_dimensions(dimensions: str) -> Tuple[int, int]:
    try:
        width, height = dimensions.lower().split("x")
        return int(width), int(height)
    except Exception:
        return 1080, 1080


def _next_image_path(output_dir: str, suffix: str = ".png") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return out_dir / f"instagram_post_{timestamp}{suffix}"


def _resolve_relative_path(path: str, base_dir: Optional[str]) -> str:
    if not path or not base_dir:
        return path
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str(Path(base_dir) / path_obj)


def _wrap_text(text: str, width: int) -> str:
    words = text.split()
    lines = []
    current = []
    for word in words:
        if len(" ".join(current + [word])) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def _write_prompt_file(path: Path, prompt: str) -> None:
    try:
        path.write_text(prompt, encoding="utf-8")
    except Exception:
        pass


def _stable_hash(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _extract_topic_from_prompt(prompt: str) -> str:
    match = re.search(r"about ['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    words = [word.strip(" .,!?:;\"'") for word in prompt.split()]
    return " ".join(words[:5]).strip()


def _title_case(text: str) -> str:
    return " ".join(word.capitalize() for word in text.split())


def _pick_palette(rng: random.Random) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    palettes = [
        ((16, 24, 38), (42, 78, 110), (255, 208, 126)),
        ((8, 42, 54), (24, 93, 112), (246, 238, 220)),
        ((25, 32, 47), (86, 108, 140), (255, 130, 118)),
        ((14, 44, 38), (62, 118, 98), (253, 240, 200)),
        ((27, 29, 35), (76, 83, 92), (255, 199, 86)),
    ]
    return rng.choice(palettes)


def _make_gradient(size: Tuple[int, int], color_top: Tuple[int, int, int], color_bottom: Tuple[int, int, int]):
    from PIL import Image
    width, height = size
    base = Image.new("RGB", (width, height), color_top)
    top = Image.new("RGB", (width, height), color_bottom)
    mask = Image.new("L", (1, height))
    for y in range(height):
        value = int(255 * (y / max(height - 1, 1)))
        mask.putpixel((0, y), value)
    mask = mask.resize((width, height))
    return Image.composite(top, base, mask)


def _load_font(size: int, bold: bool = False):
    from PIL import ImageFont
    candidates = [
        "C:\\Windows\\Fonts\\segoeuib.ttf" if bold else "C:\\Windows\\Fonts\\segoeui.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf" if bold else "C:\\Windows\\Fonts\\arial.ttf",
    ]
    for path in candidates:
        try:
            if Path(path).exists():
                return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _font_size(font) -> int:
    return int(getattr(font, "size", 20))


def _text_width(draw, text: str, font) -> int:
    try:
        return int(draw.textlength(text, font=font))
    except Exception:
        bbox = draw.textbbox((0, 0), text, font=font)
        return int(bbox[2] - bbox[0])


def _wrap_text_to_width(draw, text: str, font, max_width: int) -> List[str]:
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        if _text_width(draw, test, font) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _fit_font(draw, text: str, max_width: int, max_size: int, min_size: int, bold: bool = False):
    size = max_size
    while size >= min_size:
        font = _load_font(size, bold=bold)
        if _text_width(draw, text, font) <= max_width:
            return font
        size -= 4
    return _load_font(min_size, bold=bold)


def _is_dark_color(color: Tuple[int, int, int]) -> bool:
    r, g, b = color
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance < 120


def _load_workflow(path: str) -> Optional[Dict]:
    if not path:
        logger.error("ComfyUI workflow path is empty")
        return None
    workflow_path = Path(path)
    if not workflow_path.exists():
        logger.error("ComfyUI workflow file not found: %s", workflow_path)
        return None
    try:
        return json.loads(workflow_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to read ComfyUI workflow: %s", exc)
        return None


def _inject_prompt_into_workflow(workflow: Dict, prompt: str, prompt_field: str) -> bool:
    for node in workflow.values():
        inputs = node.get("inputs") or {}
        if prompt_field in inputs and isinstance(inputs[prompt_field], str):
            inputs[prompt_field] = prompt
            node["inputs"] = inputs
            return True
    return False


def _inject_negative_prompt(workflow: Dict, negative_prompt: str) -> None:
    for node in workflow.values():
        inputs = node.get("inputs") or {}
        if "negative_prompt" in inputs and isinstance(inputs["negative_prompt"], str):
            inputs["negative_prompt"] = negative_prompt


def _inject_checkpoint_name(workflow: Dict, checkpoint_name: str) -> bool:
    updated = False
    for node in workflow.values():
        if node.get("class_type") == "CheckpointLoaderSimple":
            inputs = node.get("inputs") or {}
            inputs["ckpt_name"] = checkpoint_name
            node["inputs"] = inputs
            updated = True
    return updated


def _inject_seed(workflow: Dict, seed: int) -> bool:
    updated = False
    for node in workflow.values():
        if node.get("class_type") == "KSampler":
            inputs = node.get("inputs") or {}
            inputs["seed"] = int(seed)
            node["inputs"] = inputs
            updated = True
    return updated


def _download_comfyui_image(server_url: str, filename: str, subfolder: str, image_type: str) -> Optional[bytes]:
    try:
        response = requests.get(
            f"{server_url}/view",
            params={"filename": filename, "subfolder": subfolder, "type": image_type},
            timeout=30,
        )
        response.raise_for_status()
        return response.content
    except Exception as exc:
        logger.error("Failed to download image from ComfyUI: %s", exc)
        return None


def _prepare_comfyui_inputs(
    workflow: Dict,
    input_dir: Optional[str],
    reference_image: Optional[str],
    pose_dir: Optional[str],
    pose_history_path: Optional[str],
    reference_node_id: Optional[str],
    pose_node_id: Optional[str],
    pose_size: int,
) -> bool:
    if not input_dir:
        logger.warning("ComfyUI input_dir not configured; skipping input image injection")
        return True

    input_path = Path(input_dir)
    input_path.mkdir(parents=True, exist_ok=True)

    if reference_image and reference_node_id:
        ref_path = Path(reference_image)
        if not ref_path.exists():
            logger.error("Reference image not found: %s", ref_path)
            return False
        ref_filename = _stage_image_to_comfyui_input(ref_path, input_path, "reference", pose_size)
        if not _inject_load_image(workflow, reference_node_id, ref_filename):
            logger.error("Failed to inject reference image into workflow")
            return False

    if pose_dir and pose_node_id:
        pose_path = _select_pose_image(Path(pose_dir), Path(pose_history_path) if pose_history_path else None, pose_size)
        if pose_path is None:
            logger.error("No pose image available")
            return False
        pose_filename = _stage_image_to_comfyui_input(pose_path, input_path, "pose", pose_size)
        if not _inject_load_image(workflow, pose_node_id, pose_filename):
            logger.error("Failed to inject pose image into workflow")
            return False

    return True


def _inject_load_image(workflow: Dict, node_id: Optional[str], filename: str) -> bool:
    if not node_id:
        return False
    node = workflow.get(str(node_id))
    if not node:
        logger.error("ComfyUI load image node id %s not found", node_id)
        return False
    inputs = node.get("inputs") or {}
    inputs["image"] = filename
    node["inputs"] = inputs
    return True


def _stage_image_to_comfyui_input(src_path: Path, input_dir: Path, prefix: str, size: int) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    dest_path = input_dir / filename
    try:
        from PIL import Image

        image = Image.open(src_path)
        image = image.convert("RGB")
        if size:
            image = image.resize((size, size), resample=Image.LANCZOS)
        image.save(dest_path)
    except Exception as exc:
        logger.warning("Failed to resize image %s (%s); copying original", src_path, exc)
        shutil.copy2(src_path, dest_path)
    return filename


def _select_pose_image(poses_dir: Path, history_path: Optional[Path], size: int) -> Optional[Path]:
    poses_dir.mkdir(parents=True, exist_ok=True)
    pose_files = sorted(poses_dir.glob("*.png"))
    if not pose_files:
        _generate_pose_library(poses_dir, count=24, size=size)
        pose_files = sorted(poses_dir.glob("*.png"))
    if not pose_files:
        return None

    used = set()
    if history_path and history_path.exists():
        try:
            used = set(json.loads(history_path.read_text(encoding="utf-8")))
        except Exception:
            used = set()

    available = [p for p in pose_files if p.name not in used]
    if not available:
        used = set()
        available = pose_files

    chosen = random.choice(available)
    used.add(chosen.name)
    if history_path:
        try:
            history_path.write_text(json.dumps(sorted(used), indent=2), encoding="utf-8")
        except Exception:
            pass
    return chosen


def _generate_pose_library(poses_dir: Path, count: int, size: int) -> None:
    for idx in range(count):
        seed = idx + 1
        out_path = poses_dir / f"pose_{seed:03d}.png"
        if out_path.exists():
            continue
        _generate_pose_map(out_path, size, random.Random(seed))


def _generate_pose_map(path: Path, size: int, rng: random.Random) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return

    width = height = size or 512
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    cx = width * (0.5 + rng.uniform(-0.08, 0.08))
    cy = height * (0.5 + rng.uniform(-0.08, 0.08))
    torso = height * rng.uniform(0.22, 0.28)
    shoulder = width * rng.uniform(0.12, 0.16)
    hip = width * rng.uniform(0.08, 0.12)
    arm = height * rng.uniform(0.16, 0.20)
    forearm = height * rng.uniform(0.16, 0.20)
    leg = height * rng.uniform(0.20, 0.26)
    calf = height * rng.uniform(0.18, 0.24)

    neck = (cx, cy - torso * 0.5)
    hip_center = (cx, cy + torso * 0.5)
    left_shoulder = (cx - shoulder, neck[1])
    right_shoulder = (cx + shoulder, neck[1])
    left_hip = (cx - hip, hip_center[1])
    right_hip = (cx + hip, hip_center[1])
    head = (cx, neck[1] - torso * 0.35)

    def limb(start, length, angle):
        return (start[0] + math.cos(angle) * length, start[1] + math.sin(angle) * length)

    left_elbow = limb(left_shoulder, arm, rng.uniform(-2.4, -0.3))
    right_elbow = limb(right_shoulder, arm, rng.uniform(-2.8, -0.6))
    left_wrist = limb(left_elbow, forearm, rng.uniform(-2.6, 0.2))
    right_wrist = limb(right_elbow, forearm, rng.uniform(-2.9, -0.2))

    left_knee = limb(left_hip, leg, rng.uniform(0.8, 1.9))
    right_knee = limb(right_hip, leg, rng.uniform(0.8, 1.9))
    left_ankle = limb(left_knee, calf, rng.uniform(1.1, 2.2))
    right_ankle = limb(right_knee, calf, rng.uniform(1.1, 2.2))

    colors = [
        (255, 0, 0),
        (255, 85, 0),
        (255, 170, 0),
        (255, 255, 0),
        (170, 255, 0),
        (85, 255, 0),
        (0, 255, 0),
        (0, 255, 85),
        (0, 255, 170),
        (0, 255, 255),
        (0, 170, 255),
        (0, 85, 255),
        (0, 0, 255),
    ]

    def draw_line(p1, p2, color):
        draw.line((p1[0], p1[1], p2[0], p2[1]), fill=color, width=int(width * 0.015))

    def draw_point(p, color):
        r = int(width * 0.02)
        draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=color)

    draw_line(head, neck, colors[0])
    draw_line(neck, left_shoulder, colors[1])
    draw_line(neck, right_shoulder, colors[2])
    draw_line(neck, hip_center, colors[3])
    draw_line(hip_center, left_hip, colors[4])
    draw_line(hip_center, right_hip, colors[5])
    draw_line(left_shoulder, left_elbow, colors[6])
    draw_line(left_elbow, left_wrist, colors[7])
    draw_line(right_shoulder, right_elbow, colors[8])
    draw_line(right_elbow, right_wrist, colors[9])
    draw_line(left_hip, left_knee, colors[10])
    draw_line(left_knee, left_ankle, colors[11])
    draw_line(right_hip, right_knee, colors[12])
    draw_line(right_knee, right_ankle, colors[0])

    for point, color in [
        (head, colors[0]),
        (neck, colors[1]),
        (left_shoulder, colors[2]),
        (right_shoulder, colors[3]),
        (left_elbow, colors[4]),
        (right_elbow, colors[5]),
        (left_wrist, colors[6]),
        (right_wrist, colors[7]),
        (left_hip, colors[8]),
        (right_hip, colors[9]),
        (left_knee, colors[10]),
        (right_knee, colors[11]),
        (left_ankle, colors[12]),
        (right_ankle, colors[0]),
    ]:
        draw_point(point, color)

    image.save(path)
