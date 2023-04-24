"""Helper functions and classes used to call the OpenAPI asynchronously, and parse the results."""

import asyncio
from source.library.openai_pricing import EmbeddingModels, InstructModels, num_tokens, cost  # noqa
from source.library.openai_utilities import ExceededMaxTokensError, InvalidModelTypeError, \
    MissingApiKeyError, OpenAIInstructResult, OpenAIEmbeddingResult, OpenAIResponse, \
    OpenAIResponses, gather_payloads


class OpenAI:
    """
    Wrapper around OpenAI API that implement asynchronous requests and return objects that wrap
    the responses and make it easy to extract data.
    """

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise MissingApiKeyError("Must pass a non-empty API key")
        self.api_key = api_key

    def text_completion(
            self,
            model: InstructModels,
            prompts: list[str],
            max_tokens: int | list[int],
            temperature: float = 0,
            top_p: float = 1,
            n: int = 1,
            stream: bool = False,
            logprobs: int | None = None,
            stop: str | None = None,
            ) -> OpenAIResponses:
        """
        Generate text completions for a list of prompts using OpenAI API.

        Args:
            model (InstructModels):
                Model to use for generating completions.
            prompts (list[str]):
                List of input prompts to generate completions for.
            max_tokens (int | list[int]):
                From OpenAI:
                ```
                The maximum number of tokens to generate in the completion.
                The token count of your prompt plus max_tokens cannot exceed the model's context
                length. Most models have a context length of 2048 tokens (except for the newest
                models, which support 4096).
                ```

                If an integer is provided, it will be used for all prompts.
                If a list is provided, it should have the same length as prompts.
            temperature (float, optional):
                From OpenAI:
                ```
                What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make
                the output more random, while lower values like 0.2 will make it more focused and
                deterministic.
                We generally recommend altering this or top_p but not both.
                ```
            top_p (float, optional):
                From OpenAI:
                ```
                An alternative to sampling with temperature, called nucleus sampling, where the
                model considers the results of the tokens with top_p probability mass. So 0.1 means
                only the tokens comprising the top 10% probability mass are considered.
                We generally recommend altering this or temperature but not both.
                ```
            n (int, optional):
                Number of completions to generate for each prompt. Defaults to 1.
            stream (bool, optional):
                Whether to stream the API response. Defaults to False.
            logprobs (int | None, optional):
                Include the log probabilities on the logprobs most likely tokens, as well the
                chosen tokens.
            stop (str | None, optional):
                Up to 4 sequences where the API will stop generating further tokens. The returned
                text will not contain the stop sequence.
        """
        if not isinstance(model, InstructModels):
            raise InvalidModelTypeError("model parameter must be an instance of InstructModels")

        if isinstance(max_tokens, int):
            max_tokens = [max_tokens] * len(prompts)

        def _create_payload(_prompt: str, _max_tokens: int) -> dict:
            return {
                "model": model.value,
                "prompt": _prompt,
                "max_tokens": _max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n": n,
                "stream": stream,
                "logprobs": logprobs,
                "stop": stop,
            }
        payloads = [
            _create_payload(_prompt=p, _max_tokens=m)
            for p, m in zip(prompts, max_tokens, strict=True)
        ]
        url = 'https://api.openai.com/v1/completions'
        responses = asyncio.run(gather_payloads(url=url, api_key=self.api_key, payloads=payloads))

        def convert_response(status: int, reason: str, oai_result: dict) -> OpenAIResponse:
            return OpenAIResponse(
                response_status=status,
                response_reason=reason,
                result=OpenAIInstructResult(result=oai_result),
            )

        return OpenAIResponses(
            responses=[convert_response(x[0].status, x[0].reason, x[1]) for x in responses],
        )

    def text_embeddings(
            self,
            model: EmbeddingModels,
            inputs: list[str],
            max_tokens: int = 8191) -> OpenAIResponses:
        """
        Generate text embeddings for a list of inputs using OpenAI API.

        Args:
            model: Model to use for generating embeddings.
            inputs: List of input texts to generate embeddings for.
            max_tokens: Maximum number of tokens allowed for each input.

        Raises:
            TypeError: If the model is not an instance of EmbeddingModels.
            ValueError: If any of the inputs have more tokens than the maximum allowed.
        """
        if not isinstance(model, EmbeddingModels):
            raise InvalidModelTypeError("model must be an instance of EmbeddingModels")
        if not all(num_tokens(value=x, model=model) <= max_tokens for x in inputs):
            raise ExceededMaxTokensError(
                f"All inputs must have fewer tokens than the maximum allowed ({max_tokens})",
            )

        def _create_payload(_input: str) -> dict:
            return {
                "model": model.value,
                "input": _input.replace('\n', ' '),
            }

        payloads = [_create_payload(_input=i) for i in inputs]
        url = 'https://api.openai.com/v1/embeddings'
        responses = asyncio.run(gather_payloads(url=url, api_key=self.api_key, payloads=payloads))

        def convert_response(status: int, reason: str, oai_result: dict) -> OpenAIResponse:
            return OpenAIResponse(
                response_status=status,
                response_reason=reason,
                result=OpenAIEmbeddingResult(result=oai_result),
            )

        return OpenAIResponses(
            responses=[convert_response(x[0].status, x[0].reason, x[1]) for x in responses],
        )
