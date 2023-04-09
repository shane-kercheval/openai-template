from abc import abstractmethod
from datetime import datetime
import asyncio
import aiohttp  # for running API calls concurrently
import json
from pydantic import BaseModel, validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, \
    RetryError
from source.library.openai_pricing import EmbeddingModels, cost, InstructModels


API_KEY = None
RETRY_ATTEMPTS = 3
RETRY_MULTIPLIER = 1
RETRY_MAX = 10


class OpenAIResultBase(BaseModel):
    result: dict | None = {}

    @validator('result', pre=True, always=True)
    def set_x_to_empty_dict(cls, value):
        return value or {}

    @property
    @abstractmethod
    def has_data():
        """
        Indicates whether or not OpenAI give a response with valid data (e.g. text, embeddings,
        etc.)
        """

    @property
    def timestamp(self) -> int:
        return self.result.get('created')

    @property
    def timestamp_utc(self) -> str:
        if self.timestamp:
            dt = datetime.utcfromtimestamp(self.timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return None

    @property
    def model(self) -> str:
        return self.result.get('model')

    @property
    def type(self) -> str:
        return self.result.get('object')

    @property
    def usage_total_tokens(self) -> int:
        return self.result.get('usage', {}).get('total_tokens')

    @property
    def cost_total(self) -> float:
        if self.model and self.usage_total_tokens:
            return cost(self.usage_total_tokens, model=self.model)
        else:
            return None

    @property
    def error_code(self) -> str:
        if not self.result:
            return 'Uknown Error (error probably occurred outside OpenAI API call)'
        return self.result.get('error', {}).get('code')

    @property
    def error_type(self) -> str:
        if not self.result:
            return 'Uknown Error (error probably occurred outside OpenAI API call)'
        return self.result.get('error', {}).get('type')

    @property
    def error_message(self) -> str:
        if not self.result:
            return 'Uknown Error (error probably occurred outside OpenAI API call)'
        return self.result.get('error', {}).get('message')


class OpenAIInstructResult(OpenAIResultBase):
    @property
    def choices(self) -> list[dict]:
        return self.result.get('choices', [dict(text='')])

    @property
    def reply(self) -> str:
        text = self.choices[0].get('text', '')
        if text:
            return text.strip()
        return ''

    @property
    def usage_prompt_tokens(self) -> int:
        return self.result.get('usage', {}).get('prompt_tokens')

    @property
    def usage_completion_tokens(self) -> int:
        return self.result.get('usage', {}).get('completion_tokens')

    @property
    def has_data(self) -> bool:
        return bool(self.reply)


class OpenAIEmbeddingResult(OpenAIResultBase):
    @property
    def data(self) -> list[dict]:
        return self.result.get('data', [dict(embedding=[])])

    @property
    def embedding(self) -> list[dict]:
        return self.data[0].get('embedding', [])

    @property
    def has_data(self) -> bool:
        return bool(self.embedding)

    @property
    def model(self) -> str:
        return super().model.replace('-v2', '')


class OpenAIResponse(BaseModel):
    response_status: int
    response_reason: str
    result: OpenAIResultBase

    @validator('response_reason')
    def clean_reason(cls, value: str):
        return value.strip()

    @validator('response_status')
    def clean_status(cls, value: int):
        if value < 0:
            raise ValueError(f"Invalid HTTP status code: {value}")
        return value

    @property
    def has_error(self) -> bool:
        return self.response_status != 200 or self.result.error_code is not None

    @property
    def has_data(self) -> bool:
        return self.result.has_data


class OpenAIResponses(BaseModel):
    responses: list[OpenAIResponse]

    @property
    def any_errors(self) -> bool:
        return any(r.has_error for r in self.responses)

    @property
    def any_missing_data(self) -> bool:
        return any(not r.has_data for r in self.responses)

    @property
    def total_tokens(self) -> int:
        return sum(r.result.usage_total_tokens for r in self.responses)

    @property
    def total_cost(self) -> float:
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
    pass


class MissingApiKeyError(Exception):
    pass


class CustomResponse(BaseModel):
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
    specified URL with a payload and returns an OpenAIResponse object.

    If the response status is 429 (Too Many Requests), the function raises a RateLimitError,
    which enables automatic retrying of the function via tenacity. The retry configuration
    parameters are set using constants defined by RETRY_ATTEMPTS, RETRY_MULTIPLIER, and RETRY_MAX).

    Parameters:
        session : The HTTP client session to use for the request.
        url : The URL to which the request is to be sent.
        payload : The payload data to send with the request.
    """
    try:
        return await _post_async_with_retry(session=session, url=url, payload=payload)
    except RetryError:
        return CustomResponse(status=429, reason='Too Many Requests'), None


async def gather_payloads(
        url: str,
        payloads: list[dict]) -> list[tuple[aiohttp.ClientResponse, dict]]:
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

    assert isinstance(model, InstructModels)
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
    payloads = [_create_payload(_prompt=p, _max_tokens=m) for p, m in zip(prompts, max_tokens)]
    url = 'https://api.openai.com/v1/completions'
    responses = asyncio.run(gather_payloads(url=url, payloads=payloads))

    def convert_response(status, reason, oai_result) -> OpenAIResponse:
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
        inputs: list[str]) -> OpenAIResponse:
    assert isinstance(model, EmbeddingModels)

    def _create_payload(_input: str):
        return dict(
            model=model.value,
            input=_input,
        )
    payloads = [_create_payload(_input=i) for i in inputs]
    url = 'https://api.openai.com/v1/embeddings'

    responses = asyncio.run(gather_payloads(url=url, payloads=payloads))

    def convert_response(status, reason, oai_result) -> OpenAIResponse:
        return OpenAIResponse(
            response_status=status,
            response_reason=reason,
            result=OpenAIEmbeddingResult(result=oai_result),
        )

    return OpenAIResponses(
        responses=[convert_response(x[0].status, x[0].reason, x[1]) for x in responses]
    )
