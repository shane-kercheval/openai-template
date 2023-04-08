import asyncio  # for making API calls concurrently
import aiohttp  # for running API calls concurrently
from dataclasses import dataclass
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, \
    RetryError


# openai.api_key = os.getenv("OPENAI_API_KEY")
with open('/.openai_api_key', 'r') as handle:
    openai_api_key = handle.read().strip()

URL = 'https://api.openai.com/v1/completions'

HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {openai_api_key}'
}

# Example prompts
prompts = [
    "What are the benefits of exercise?",
    "How to bake a chocolate cake?",
    "What is the capital of France?",
]


class RateLimitError(Exception):
    pass


@dataclass
class OpenAIResponse:
    response_status: int
    response_reason: str
    prompt: str
    openai_result: dict


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type(RateLimitError)
)
async def call_openai_api_with_retry(session, prompt) -> OpenAIResponse:
    payload = {
        'model': 'text-babbage-001',
        'prompt': prompt,
        'max_tokens': 50
    }
    async with session.post(URL, headers=HEADERS, data=json.dumps(payload)) as response:
        print(prompt)  # TODO: delete
        result = await response.json()
        if response.status == 429:
            # 429 is `Too Many Requests`
            # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
            raise RateLimitError("Rate limit exceeded")

        return OpenAIResponse(
            response_status=response.status,
            response_reason=response.reason,
            prompt=prompt,
            openai_result=result
        )


async def call_openai_api(session, prompt) -> OpenAIResponse:
    try:
        return await call_openai_api_with_retry(session, prompt)
    except RetryError:
        return OpenAIResponse(
            response_status=429,
            response_reason='Too Many Requests',
            prompt=prompt,
            openai_result=None
        )


async def process_prompts() -> list[OpenAIResponse]:
    async with aiohttp.ClientSession() as session:
        tasks = [call_openai_api(session, prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        return results


def main():
    responses = asyncio.run(process_prompts())
    print("\n----\nResults\n----\n")
    for result in responses:
        if result.response_status == 200:
            print(f"Prompt: {result.prompt}\nResponse: {result.openai_result['choices'][0]['text'].strip()}\n")
        else:
            print(
                f"Prompt: {result.prompt}\n"
                f"Error {result.response_status}: {result.response_reason}\n"
            )


if __name__ == "__main__":
    main()
