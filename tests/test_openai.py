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
    assert 'Rome' in response.openai_result.reply
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
    assert len(responses[0:2]) == 2
    verify_openai_response(responses[0], expected_model=OPENAI_MODEL)
    verify_openai_response(responses[1], expected_model=OPENAI_MODEL)
    verify_openai_response(responses[2], expected_model=OPENAI_MODEL)
    assert 'Rome' in responses[0].openai_result.reply
    assert 'Paris' in responses[1].openai_result.reply
    assert 'London' in responses[2].openai_result.reply
    total_tokens = sum([x.openai_result.usage_total_tokens for x in responses])
    total_cost = sum([x.openai_result.cost_total for x in responses])
    assert total_cost == cost(total_tokens, model=OPENAI_MODEL)
    assert responses.total_cost == total_cost
    assert responses.total_tokens == sum(r.openai_result.usage_total_tokens for r in responses)
    assert not responses.any_errors

    # set response_status to non-200 to mock an error
    responses[0].response_status = 1
    assert responses[0].has_error
    assert not responses[1].has_error
    assert not responses[2].has_error
    assert responses.any_errors
    # set back to ensure there are no longer errors
    responses[0].response_status = 200
    assert not responses.any_errors
    # set error within original openai response dict to mock an error
    responses[0].openai_result.result['error'] = {'code': 'error'}
    assert responses[0].has_error
    assert not responses[1].has_error
    assert not responses[2].has_error
    assert responses.any_errors

    # test any_missing_replies() functionality
    assert not responses.any_missing_replies
    assert responses[0].has_reply
    assert responses[1].has_reply
    assert responses[2].has_reply
    # set error within original openai response dict to mock no response
    responses[1].openai_result.result['choices'][0]['text'] = '\n\n'
    assert responses.any_missing_replies
    assert responses[0].has_reply
    assert not responses[1].has_reply
    assert responses[2].has_reply
    # set back to original
    responses[1].openai_result.result['choices'][0]['text'] = '\n\nParis'
    assert not responses.any_missing_replies
    assert responses[0].has_reply
    assert responses[1].has_reply
    assert responses[2].has_reply
    # set error within original openai response dict to mock no response
    responses[1].openai_result.result['choices'][0]['text'] = None
    assert responses.any_missing_replies
    assert responses[0].has_reply
    assert not responses[1].has_reply
    assert responses[2].has_reply


def test__complete__missing_api_key(OPENAI_MODEL):
    prompts = [
        "Question: What is the capital of Italy? ",
        "What is the capital of France?",
        "What is the capital of the United Kingdom?",
    ]
    oai.API_KEY = None
    with pytest.raises(oai.MissingApiKeyError):
        oai.text_completion(model=OPENAI_MODEL, prompts=prompts, max_tokens=10)
