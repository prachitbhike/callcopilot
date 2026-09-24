"""Step 0: pick current Haiku / Sonnet via models.list() and write exact IDs into .env."""
import re
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()
load_dotenv(".env.local")


def main():
    client = anthropic.Anthropic()
    ids = [m.id for m in client.models.list(limit=100)]  # newest first
    haiku = next(i for i in ids if "haiku" in i)
    sonnet = next(i for i in ids if "sonnet" in i)
    env = Path(".env")
    text = env.read_text() if env.exists() else ""
    for k, v in (("GEN_MODEL", haiku), ("JUDGE_MODEL", sonnet)):
        text = re.sub(rf"^{k}=.*$", f"{k}={v}", text, flags=re.M) if re.search(rf"^{k}=", text, re.M) else text.rstrip("\n") + f"\n{k}={v}\n"
    env.write_text(text)
    r = client.messages.create(model=haiku, max_tokens=1, messages=[{"role": "user", "content": "hi"}])
    print(f"models available: {len(ids)}")
    print(f"GEN_MODEL={haiku}\nJUDGE_MODEL={sonnet}")
    print(f"1-token call ok: stop_reason={r.stop_reason}")


if __name__ == "__main__":
    main()
