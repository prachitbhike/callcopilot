"""Model-aware request kwargs. Sonnet 5 / Opus 5-era models reject `temperature` and think by default;
forced tool use needs thinking disabled there. Older models (Haiku 4.5) take temperature."""


def sampling_kwargs(model: str, temperature: float) -> dict:
    legacy = any(t in model for t in ("haiku-4", "sonnet-4", "opus-4", "3-"))
    return {"temperature": temperature} if legacy else {"thinking": {"type": "disabled"}}
