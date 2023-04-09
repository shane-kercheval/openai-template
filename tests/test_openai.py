import pytest
import aiohttp
from tenacity import wait_none
import source.library.openai as oai
from source.library.openai_pricing import cost

from tests.conftest import CustomAsyncMock, MockResponse, verify_openai_response, \
    verify_openai_response_on_error


@pytest.mark.asyncio
async def test__api_post(
        OPENAI_API_KEY,
        OPENAI_URL_COMPLETION,
        OPENAI_MODEL,
        babbage_model_payload__italy_capital):
    assert OPENAI_API_KEY
    oai.API_KEY = OPENAI_API_KEY

    async with aiohttp.ClientSession() as session:
        response = await oai.post_async(
            session=session,
            url=OPENAI_URL_COMPLETION,
            payload=babbage_model_payload__italy_capital,
        )
    assert 'Rome' in response.openai_result.text
    verify_openai_response(response=response, expected_model=OPENAI_MODEL)


@pytest.mark.asyncio
async def test__api_post__invalid_api_key(
        OPENAI_URL_COMPLETION,
        babbage_model_payload__italy_capital):
    oai.API_KEY = 'invalid'
    async with aiohttp.ClientSession() as session:
        response = await oai.post_async(
            session=session,
            url=OPENAI_URL_COMPLETION,
            payload=babbage_model_payload__italy_capital,
        )
    assert response.response_status == 401
    assert response.response_reason == 'Unauthorized'
    verify_openai_response_on_error(response)


@pytest.mark.asyncio
async def test__post_async__429_error():
    url = 'https://example.com/api'
    payload = {'key': 'value'}

    mock_response = MockResponse(429)
    post_mock = CustomAsyncMock(return_value=mock_response)

    async with aiohttp.ClientSession() as session:
        session.post = post_mock
        oai._post_async_with_retry.retry.wait = wait_none()
        response = await oai.post_async(session, url, payload)
        assert response.response_status == 429
        assert response.response_reason == 'Too Many Requests'
        assert response.openai_result is not None
        assert session.post.call_count == 3
        verify_openai_response_on_error(response)


def test__complete(OPENAI_MODEL, OPENAI_API_KEY):
    prompts = [
        "Question: What is the capital of Italy? ",
        "What is the capital of France?",
        "What is the capital of the United Kingdom?",
    ]
    oai.API_KEY = OPENAI_API_KEY
    responses = oai.text_completion(model=OPENAI_MODEL, prompts=prompts, max_tokens=10)
    assert len(responses) == len(prompts)
    assert all([r.response_status == 200 for r in responses])
    assert all([r.response_reason == 'OK' for r in responses])
    assert all([r.openai_result is not None for r in responses])
    verify_openai_response(responses[0], expected_model=OPENAI_MODEL)
    verify_openai_response(responses[1], expected_model=OPENAI_MODEL)
    verify_openai_response(responses[2], expected_model=OPENAI_MODEL)
    assert 'Rome' in responses[0].openai_result.text
    assert 'Paris' in responses[1].openai_result.text
    assert 'London' in responses[2].openai_result.text
    total_tokens = sum([x.openai_result.usage_total_tokens for x in responses])
    total_cost = sum([x.openai_result.cost_total for x in responses])
    assert total_cost == cost(total_tokens, model=OPENAI_MODEL)


def test__complete__missing_api_key(OPENAI_MODEL):
    prompts = [
        "Question: What is the capital of Italy? ",
        "What is the capital of France?",
        "What is the capital of the United Kingdom?",
    ]
    oai.API_KEY = None
    with pytest.raises(oai.MissingApiKeyError):
        oai.text_completion(model=OPENAI_MODEL, prompts=prompts, max_tokens=10)
