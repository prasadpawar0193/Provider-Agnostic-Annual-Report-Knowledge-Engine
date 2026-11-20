
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from anthropic import Anthropic

load_dotenv()


def generate_answer(prompt, provider=None):
    """
    Generate an answer using the selected LLM provider.
    Supported providers: Gemini, OpenAI, Claude.
    """

    provider = (
        provider or os.getenv("LLM_PROVIDER", "gemini")
    ).lower()

    if provider == "gemini":

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing in .env")

        client = genai.Client(api_key=api_key)

        # Retry temporary Gemini server errors up to 3 times
        for attempt in range(3):

            try:
                response = client.models.generate_content(
                    model=os.getenv(
                        "GEMINI_MODEL",
                        "gemini-3.8-flash"
                    ),
                    contents=prompt
                )

                return response.text or ""

            except Exception as error:

                error_message = str(error).upper()

                if (
                    "503" in error_message
                    or "UNAVAILABLE" in error_message
                ):

                    if attempt < 2:

                        wait_time = 2 ** (attempt + 1)

                        print(
                            f"Gemini temporarily unavailable. "
                            f"Retrying in {wait_time} seconds..."
                        )

                        time.sleep(wait_time)

                    else:
                        raise

                else:
                    raise

    elif provider == "openai":

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing in .env")

        client = OpenAI(api_key=api_key)

        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            input=prompt
        )

        return response.output_text or ""

    elif provider == "claude":

        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing in .env")

        client = Anthropic(api_key=api_key)

        response = client.messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5"),
            max_tokens=1500,
            messages=[
                {"role": "user", "content": prompt}
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
