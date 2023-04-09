from datetime import datetime
import asyncio
import aiohttp  # for running API calls concurrently
import json
from pydantic import BaseModel, validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, \
    RetryError
from source.library.openai_pricing import cost


API_KEY = None
RETRY_ATTEMPTS = 3
RETRY_MULTIPLIER = 1
RETRY_MAX = 10


class OpenAIResult(BaseModel):
    result: dict | None = {}

    @validator('result', pre=True, always=True)
    def set_x_to_empty_dict(cls, value):
        return value or {}

    @property
    def choices(self) -> list:
        return self.result.get('choices', [dict(text='')])

    @property
    def text(self) -> str:
        return self.choices[0]['text'].strip()

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
    def usage_total_tokens(self) -> int:
        return self.result.get('usage', {}).get('total_tokens')

    @property
    def usage_prompt_tokens(self) -> int:
        return self.result.get('usage', {}).get('prompt_tokens')

    @property
    def usage_completion_tokens(self) -> int:
        return self.result.get('usage', {}).get('completion_tokens')

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


class OpenAIResponse(BaseModel):
    response_status: int
    response_reason: str
    openai_result: OpenAIResult

    @validator('response_reason')
    def clean_reason(cls, value: str):
        return value.strip()

    @validator('response_status')
    def clean_status(cls, value: int):
        if value < 0:
            raise ValueError(f"Invalid HTTP status code: {value}")
        return value


class RateLimitError(Exception):
    pass


@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX),
    retry=retry_if_exception_type(RateLimitError)
)
async def _post_async_with_retry(
        session: aiohttp.ClientSession,
        url: str,
        payload: dict) -> OpenAIResponse:
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

    return OpenAIResponse(
        response_status=response.status,
        response_reason=response.reason,
        openai_result=OpenAIResult(result=result),
    )


async def post_async(
        session: aiohttp.ClientSession,
        url: str,
        payload: dict) -> OpenAIResponse:
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
        return OpenAIResponse(
            response_status=429,
            response_reason='Too Many Requests',
            openai_result=OpenAIResult(result=None),
        )


async def gather_payloads(url: str, payloads: list[dict]) -> list[OpenAIResponse]:
    async with aiohttp.ClientSession() as session:
        tasks = [post_async(session=session, url=url, payload=payload) for payload in payloads]
        results = await asyncio.gather(*tasks)
        return results


def text_completion(
        model: str,
        prompts: list[str],
        max_tokens: int | list[int],
        temperature: float = 0,
        top_p: float = 1,
        n: int = 1,
        stream: bool = False,
        logprobs: int | None = None,
        stop: str | None = None
        ) -> list[OpenAIResponse]:

    if isinstance(max_tokens, int):
        max_tokens = [max_tokens] * len(prompts)

    def _create_payload(_prompt: str, _max_tokens: int):
        return dict(
            model=model,
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
    return responses
