import pytest
import aiohttp
from tenacity import wait_none
import source.library.openai as openail
from tests.conftest import CustomAsyncMock, MockResponse, is_valid_datetime_format


@pytest.mark.asyncio
async def test__api_post(
        OPENAI_API_KEY,
        OPENAI_URL_COMPLETION,
        OPENAI_MODEL,
        babbage_model_payload__italy_capital):
    assert OPENAI_API_KEY
    openail.API_KEY = OPENAI_API_KEY

    async with aiohttp.ClientSession() as session:
        result = await openail.post_async(
            session=session,
            url=OPENAI_URL_COMPLETION,
            payload=babbage_model_payload__italy_capital,
        )
    assert 'Rome' in result.openai_result.text

    assert result.response_status == 200
    assert result.response_reason == 'OK'
    assert len(result.openai_result.choices) == 1
    assert result.openai_result.timestamp > 1680995788
    assert is_valid_datetime_format(result.openai_result.timestamp_utc)
    assert result.openai_result.model == OPENAI_MODEL
    assert result.openai_result.usage_total_tokens > 0
    assert 0 < result.openai_result.usage_prompt_tokens < result.openai_result.usage_total_tokens
    assert 0 < result.openai_result.usage_completion_tokens < result.openai_result.usage_total_tokens  # noqa


@pytest.mark.asyncio
async def test__api_post__invalid_api_key(
        OPENAI_URL_COMPLETION,
        babbage_model_payload__italy_capital):
    openail.API_KEY = 'invalid'
    async with aiohttp.ClientSession() as session:
        result = await openail.post_async(
            session=session,
            url=OPENAI_URL_COMPLETION,
            payload=babbage_model_payload__italy_capital,
        )
    assert result.response_status == 401
    assert result.response_reason == 'Unauthorized'
    # assert result.openai_result['error'] is not None


@pytest.mark.asyncio
async def test__post_async__429_error():
    url = 'https://example.com/api'
    payload = {'key': 'value'}

    mock_response = MockResponse(429)
    post_mock = CustomAsyncMock(return_value=mock_response)

    async with aiohttp.ClientSession() as session:
        session.post = post_mock
        openail._post_async_with_retry.retry.wait = wait_none()
        response = await openail.post_async(session, url, payload)
        assert response.response_status == 429
        assert response.response_reason == 'Too Many Requests'
        assert response.openai_result is None
        assert session.post.call_count == 3


def test__complete(OPENAI_MODEL, OPENAI_API_KEY):
    prompts = [
        "Question: What is the capital of Italy? ",
        "What is the capital of France?",
        "What is the capital of the United Kingdom?",
    ]
    openail.API_KEY = OPENAI_API_KEY
    results = openail.complete(model=OPENAI_MODEL, prompts=prompts, max_tokens=10)
    assert len(results) == len(prompts)
    assert all([r.response_status == 200 for r in results])
    assert all([r.response_reason == 'OK' for r in results])
    assert all([r.openai_result is not None for r in results])

    # results[0].openai_result['choices'][0]['text']
    # results[1].openai_result['choices'][0]['text']
    # results[2].openai_result['choices'][0]['text']
