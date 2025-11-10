"""LLM provider implementations."""

import os
from typing import Optional
import openai
import anthropic
from ..base import BaseLLM
from ...utils.logger import get_logger

logger = get_logger()


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        timeout: int = 30
    ):
        """Initialize OpenAI LLM.

        Args:
            model_name: Name of the OpenAI model
            api_key: OpenAI API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Set API key
        if api_key:
            openai.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key must be provided or set in environment")

        logger.info(f"Initialized OpenAI LLM with model: {model_name}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (overrides default)
            temperature: Sampling temperature (overrides default)
            **kwargs: Additional OpenAI-specific arguments

        Returns:
            Generated text
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        logger.debug(f"Generating with OpenAI (model={self.model_name}, max_tokens={max_tokens})")

        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=self.timeout,
                **kwargs
            )

            generated_text = response.choices[0].message.content
            logger.debug(f"Generated {len(generated_text)} characters")

            return generated_text

        except Exception as e:
            logger.error(f"Error generating with OpenAI: {e}")
            raise


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM provider."""

    def __init__(
        self,
        model_name: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        timeout: int = 30
    ):
        """Initialize Anthropic LLM.

        Args:
            model_name: Name of the Claude model
            api_key: Anthropic API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Set API key
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        elif os.getenv("ANTHROPIC_API_KEY"):
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError("Anthropic API key must be provided or set in environment")

        logger.info(f"Initialized Anthropic LLM with model: {model_name}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (overrides default)
            temperature: Sampling temperature (overrides default)
            **kwargs: Additional Anthropic-specific arguments

        Returns:
            Generated text
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        logger.debug(f"Generating with Anthropic (model={self.model_name}, max_tokens={max_tokens})")

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                timeout=self.timeout,
                **kwargs
            )

            generated_text = response.content[0].text
            logger.debug(f"Generated {len(generated_text)} characters")

            return generated_text

        except Exception as e:
            logger.error(f"Error generating with Anthropic: {e}")
            raise
