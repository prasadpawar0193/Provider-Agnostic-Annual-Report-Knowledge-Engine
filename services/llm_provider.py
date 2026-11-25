
import os

from dotenv import load_dotenv

load_dotenv()


def generate_answer(prompt, provider=None):
    """
    Generate answers using Gemini, OpenAI or Claude.
    """

    provider = (
        provider or os.getenv("LLM_PROVIDER", "gemini")
    ).lower()

    # ---------------- GEMINI ----------------
    if provider == "gemini":

        from google import genai

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing in .env")

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            contents=prompt
        )

        return response.text or ""

    # ---------------- OPENAI ----------------
    elif provider == "openai":

        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing in .env")

        client = OpenAI(api_key=api_key)

        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            input=prompt
        )

        return response.output_text or ""

    # ---------------- CLAUDE ----------------
    elif provider == "claude":

        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing in .env")

        client = Anthropic(api_key=api_key)

        response = client.messages.create(
            model=os.getenv(
                "CLAUDE_MODEL",
                "claude-sonnet-4-5"
            ),
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return "\n".join(
            block.text
            for block in response.content
            if block.type == "text"
        )

    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            "Choose gemini, openai, or claude."
        )
