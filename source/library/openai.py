"""Helper functions and classes used to call the OpenAPI asynchronously, and parse the results."""
from abc import abstractmethod
from datetime import datetime
import asyncio
import aiohttp
import json
from pydantic import BaseModel, validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, \
    RetryError
from source.library.openai_pricing import EmbeddingModels, InstructModels, cost, num_tokens


API_KEY = None
RETRY_ATTEMPTS = 3
RETRY_MULTIPLIER = 1
RETRY_MAX = 10


class OpenAIResultBase(BaseModel):
    """Class that parses the dictionary result returned by OpenAI"""
    result: dict | None = {}

    @validator('result', pre=True, always=True)
    def set_x_to_empty_dict(cls, value):
        """Ensure that if the value pass to result is None then set it to an empty dict."""
        return value or {}

    @property
    @abstractmethod
    def has_data():
        """
        Indicates whether or not OpenAI provides a response with valid data (e.g. text, embeddings,
        etc.)
        """

    @property
    def timestamp(self) -> int:
        """Returns the timestamp of the response provided by OpenAI."""
        return self.result.get('created')

    @property
    def timestamp_utc(self) -> str:
        """Return the timestamp in the format of YYYY-MM-DD HH:MM:SS"""
        if self.timestamp:
            dt = datetime.utcfromtimestamp(self.timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return None

    @property
    def model(self) -> str:
        """Returns the name of the model used (e.g. `text-davinci-003`)."""
        return self.result.get('model')

    @property
    def type(self) -> str:
        """Returns the type of model used (e.g. `text_completion`)."""
        return self.result.get('object')

    @property
    def usage_total_tokens(self) -> int:
        """Returns the total number of tokens used (and charged for)."""
        return self.result.get('usage', {}).get('total_tokens')

    @property
    def cost_total(self) -> float:
        """Returns the total cost of the API call."""
        if self.model and self.usage_total_tokens:
            return cost(self.usage_total_tokens, model=self.model)
        else:
            return None

    @property
    def error_code(self) -> str:
        """Returns the error code, if there was an error from OpenAI"""
        if not self.result:
            return 'Uknown Error (error probably occurred outside OpenAI API call)'
        return self.result.get('error', {}).get('code')

    @property
    def error_type(self) -> str:
        """Returns the error type, if there was an error from OpenAI"""
        if not self.result:
            return 'Uknown Error (error probably occurred outside OpenAI API call)'
        return self.result.get('error', {}).get('type')

    @property
    def error_message(self) -> str:
        """Returns the error message, if there was an error from OpenAI"""
        if not self.result:
            return 'Uknown Error (error probably occurred outside OpenAI API call)'
        return self.result.get('error', {}).get('message')


class OpenAIInstructResult(OpenAIResultBase):
    """
    Class that parses the dictionary result returned by OpenAI, specific to an Instruct Model.
    """
    @property
    def choices(self) -> list[dict]:
        """If multiple prompts were given, this property gives access to all of the responses."""
        return self.result.get('choices', [dict(text='')])

    @property
    def reply(self) -> str:
        """This property gives the text/reply of the first (or only) response to the prompt(s)."""
        text = self.choices[0].get('text', '')
        if text:
            return text.strip()
        return ''

    @property
    def usage_prompt_tokens(self) -> int:
        """Returns the number of tokens used in the prompt."""
        return self.result.get('usage', {}).get('prompt_tokens')

    @property
    def usage_completion_tokens(self) -> int:
        """Returns the number of tokens used in the completion of the prompt."""
        return self.result.get('usage', {}).get('completion_tokens')

    @property
    def has_data(self) -> bool:
        """
        Returns the whether or not the result from OpenAI has data (e.g. not an empty string).
        """
        return bool(self.reply)


class OpenAIEmbeddingResult(OpenAIResultBase):
    """
    Class that parses the dictionary result returned by OpenAI, specific to an Embeddings Model.
    """
    @property
    def data(self) -> list[dict]:
        """
        If multiple inputs were given, this property gives access to all of the embeddings for each
        input.
        """
        return self.result.get('data', [dict(embedding=[])])

    @property
    def embedding(self) -> list[dict]:
        """This property gives the embedding of the first (or only) input(s)."""
        return self.data[0].get('embedding', [])

    @property
    def has_data(self) -> bool:
        """Indicates whether or not a non-empty array was returned."""
        return bool(self.embedding)

    @property
    def model(self) -> str:
        """Returns the name of the model used (e.g. `text-embedding-ada-002`)."""
        return super().model.replace('-v2', '')


class OpenAIResponse(BaseModel):
    """Class that wraps the http status/response and the result given by OpenAI."""
    response_status: int
    response_reason: str
    result: OpenAIResultBase

    @validator('response_reason')
    def clean_reason(cls, value: str):
        return value.strip()

    @validator('response_status')
    def clean_status(cls, value: int):
        """The http status of the API call (e.g. 200 for success)"""
        if value < 0:
            raise ValueError(f"Invalid HTTP status code: {value}")
        return value

    @property
    def has_error(self) -> bool:
        """Indicates whether or not the api call or the OpenAI result has an error."""
        return self.response_status != 200 or self.result.error_code is not None

    @property
    def has_data(self) -> bool:
        """Indicates whether or not the OpenAI result has any data (e.g. not an empty string)."""
        return self.result.has_data


class OpenAIResponses(BaseModel):
    """Class that wraps multiple http responses and OpenAI results"""
    responses: list[OpenAIResponse]

    @property
    def any_errors(self) -> bool:
        """Indicates if any of the api calls or the OpenAI results have an error."""
        return any(r.has_error for r in self.responses)

    @property
    def any_missing_data(self) -> bool:
        """Indicates if any of the OpenAI results are missing data (e.g. an empty string)."""
        return any(not r.has_data for r in self.responses)

    @property
    def total_tokens(self) -> int:
        """Returns the total number of tokens used across all requests (and charged for)."""
        return sum(r.result.usage_total_tokens for r in self.responses)

    @property
    def total_cost(self) -> float:
        """Returns the total cost across all of the API calls."""
        return sum(r.result.cost_total for r in self.responses)

    def __len__(self):
        return len(self.responses)

    def __iter__(self):
        for response in self.responses:
            yield response

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.responses[index]
        elif isinstance(index, slice):
            return self.responses[index.start:index.stop:index.step]
        else:
            raise TypeError("Invalid index type")


class RateLimitError(Exception):
    """Exception to indicate that there was a OpenAI gave an error related to its Rate-Limit."""


class MissingApiKeyError(Exception):
    """Exception to indicate that the OpenAI api key was not set."""


class CustomResponse(BaseModel):
    """Custom Reponse object used when there is an error and we need to return status/reason."""
    status: int
    reason: str


@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX),
    retry=retry_if_exception_type(RateLimitError)
)
async def _post_async_with_retry(
        session: aiohttp.ClientSession,
        url: str,
        payload: dict) -> tuple[aiohttp.ClientResponse, dict]:
    """
    Helper function that allows us to retry in case of rate-limit errors, and recover
    gracefully if we exceed the maximum attempts.
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    async with session.post(url, headers=headers, data=json.dumps(payload)) as response:
        result = await response.json()

    if response.status == 429:
        # 429 is `Too Many Requests`
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
        raise RateLimitError("Rate limit exceeded")

    return response, result


async def post_async(
        session: aiohttp.ClientSession,
        url: str,
        payload: dict) -> tuple[aiohttp.ClientResponse, dict]:
    """
    The post_async function is an asynchronous function that sends an HTTP POST request to a
    specified URL with a payload and returns a ClientResponse object along with the json/dict
    returned by the OpenAI response.

    If the response status is 429 (Too Many Requests), the function raises a RateLimitError,
    which enables automatic retrying of the function via tenacity. The retry configuration
    parameters are set using constants defined by RETRY_ATTEMPTS, RETRY_MULTIPLIER, and RETRY_MAX).

    Parameters:
        session: The HTTP client session to use for the request.
        url: The URL to which the request is to be sent.
        payload: The payload data to send with the request.
    """
    try:
        return await _post_async_with_retry(session=session, url=url, payload=payload)
    except RetryError:
        return CustomResponse(status=429, reason='Too Many Requests'), None


async def gather_payloads(
        url: str,
        payloads: list[dict]) -> list[tuple[aiohttp.ClientResponse, dict]]:
    """
    This function makes calls to the url asynchronously via post_async for all of the payloads
    provided. and returns an OpenAIResponse object.

    If the response status is 429 (Too Many Requests), the function raises a RateLimitError,
    which enables automatic retrying of the function via tenacity. The retry configuration
    parameters are set using constants defined by RETRY_ATTEMPTS, RETRY_MULTIPLIER, and RETRY_MAX).

    Parameters:
        urls: The URL to which the request is to be sent.
        payloads: A list of payload data to send with each request.
    """
    if not API_KEY:
        raise MissingApiKeyError()

    async with aiohttp.ClientSession() as session:
        tasks = [post_async(session=session, url=url, payload=payload) for payload in payloads]
        results = await asyncio.gather(*tasks)
        return results


def text_completion(
        model: InstructModels,
        prompts: list[str],
        max_tokens: int | list[int],
        temperature: float = 0,
        top_p: float = 1,
        n: int = 1,
        stream: bool = False,
        logprobs: int | None = None,
        stop: str | None = None
        ) -> OpenAIResponses:
    """
    Generate text completions for a list of prompts using OpenAI API.

    Args:
        model (InstructModels):
            Model to use for generating completions.
        prompts (list[str]):
            List of input prompts to generate completions for.
        max_tokens (int | list[int]):
            Maximum number of tokens to generate for each prompt.
            If an integer is provided, it will be used for all prompts.
            If a list is provided, it should have the same length as prompts.
        temperature (float, optional):
            Sampling temperature. Higher values make output more random.
                                       Defaults to 0.
        top_p (float, optional):
            Probability mass cutoff for nucleus sampling. Defaults to 1.
        n (int, optional):
            Number of completions to generate for each prompt. Defaults to 1.
        stream (bool, optional):
            Whether to stream the API response. Defaults to False.
        logprobs (int | None, optional):
            Number of log probabilities to return. None for not returning.
                                         Defaults to None.
        stop (str | None, optional):
            Token at which to stop generating tokens. Defaults to None.
    """

    if not isinstance(model, InstructModels):
        raise TypeError("model parameter must be an instance of InstructModels")

    if isinstance(max_tokens, int):
        max_tokens = [max_tokens] * len(prompts)

    def _create_payload(_prompt: str, _max_tokens: int):
        return dict(
            model=model.value,
            prompt=_prompt,
            max_tokens=_max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            logprobs=logprobs,
            stop=stop,
        )
    payloads = [
        _create_payload(_prompt=p, _max_tokens=m) for p, m in zip(prompts, max_tokens, strict=True)
    ]
    url = 'https://api.openai.com/v1/completions'
    responses = asyncio.run(gather_payloads(url=url, payloads=payloads))

    def convert_response(status: int, reason: str, oai_result: dict) -> OpenAIResponse:
        return OpenAIResponse(
            response_status=status,
            response_reason=reason,
            result=OpenAIInstructResult(result=oai_result),
        )

    return OpenAIResponses(
        responses=[convert_response(x[0].status, x[0].reason, x[1]) for x in responses]
    )


def text_embeddings(
        model: EmbeddingModels,
        inputs: list[str],
        max_tokens: int = 8191) -> OpenAIResponse:
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
        raise TypeError("model must be an instance of EmbeddingModels")
    if not all(num_tokens(value=x, model=model) <= max_tokens for x in inputs):
        raise ValueError(f"All inputs must have fewer tokens than the maximum allowed ({max_tokens})")  # noqa

    def _create_payload(_input: str):
        return dict(
            model=model.value,
            input=_input,
        )

    payloads = [_create_payload(_input=i) for i in inputs]
    url = 'https://api.openai.com/v1/embeddings'

    responses = asyncio.run(gather_payloads(url=url, payloads=payloads))

    def convert_response(status: int, reason: str, oai_result: dict) -> OpenAIResponse:
        return OpenAIResponse(
            response_status=status,
            response_reason=reason,
            result=OpenAIEmbeddingResult(result=oai_result),
        )

    return OpenAIResponses(
        responses=[convert_response(x[0].status, x[0].reason, x[1]) for x in responses]
    )
