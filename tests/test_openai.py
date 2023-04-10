import pytest
import aiohttp
from tenacity import wait_none
import source.library.openai as oai
from source.library.openai_pricing import EmbeddingModels, cost, _encode

from tests.conftest import CustomAsyncMock, MockResponse, verify_openai_embedding_response, \
    verify_openai_instruct_response, verify_openai_instruct_response_on_error


@pytest.mark.asyncio
async def test__api_post(
        OPENAI_API_KEY,
        OPENAI_URL_COMPLETION,
        OPENAI_MODEL,
        babbage_model_payload__italy_capital):
    assert OPENAI_API_KEY
    oai.API_KEY = OPENAI_API_KEY

    async with aiohttp.ClientSession() as session:
        response, result = await oai.post_async(
            session=session,
            url=OPENAI_URL_COMPLETION,
            payload=babbage_model_payload__italy_capital,
        )
    # Create OpenAIResponse to more easily extract/test values from OpenAI
    # This object will be created for the end-user for higher-level functions
    oai_response = oai.OpenAIResponse(
        response_status=response.status,
        response_reason=response.reason,
        result=oai.OpenAIInstructResult(result=result),
    )
    assert 'Rome' in oai_response.result.reply
    verify_openai_instruct_response(response=oai_response, expected_model=OPENAI_MODEL)


@pytest.mark.asyncio
async def test__api_post__invalid_api_key(
        OPENAI_URL_COMPLETION,
        babbage_model_payload__italy_capital):
    oai.API_KEY = 'invalid'
    async with aiohttp.ClientSession() as session:
        response, result = await oai.post_async(
            session=session,
            url=OPENAI_URL_COMPLETION,
            payload=babbage_model_payload__italy_capital,
        )
    # Create OpenAIResponse to more easily extract/test values from OpenAI
    # This object will be created for the end-user for higher-level functions
    oai_response = oai.OpenAIResponse(
        response_status=response.status,
        response_reason=response.reason,
        result=oai.OpenAIInstructResult(result=result),
    )
    assert oai_response.response_status == 401
    assert oai_response.response_reason == 'Unauthorized'
    verify_openai_instruct_response_on_error(oai_response)


@pytest.mark.asyncio
async def test__post_async__429_error():
    url = 'https://example.com/api'
    payload = {'key': 'value'}

    mock_response = MockResponse(429)
    post_mock = CustomAsyncMock(return_value=mock_response)

    async with aiohttp.ClientSession() as session:
        session.post = post_mock
        oai._post_async_with_retry.retry.wait = wait_none()
        response, result = await oai.post_async(session, url, payload)
        # Create OpenAIResponse to more easily extract/test values from OpenAI
        # This object will be created for the end-user for higher-level functions
        oai_response = oai.OpenAIResponse(
            response_status=response.status,
            response_reason=response.reason,
            result=oai.OpenAIInstructResult(result=result),
        )
        assert oai_response.response_status == 429
        assert oai_response.response_reason == 'Too Many Requests'
        assert oai_response.result is not None
        assert session.post.call_count == 3
        verify_openai_instruct_response_on_error(oai_response)


def test__text_completion(OPENAI_MODEL, OPENAI_API_KEY):
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
    assert all([r.result is not None for r in responses])
    assert len(responses[0:2]) == 2
    with pytest.raises(TypeError):
        responses['test']
    verify_openai_instruct_response(responses[0], expected_model=OPENAI_MODEL)
    verify_openai_instruct_response(responses[1], expected_model=OPENAI_MODEL)
    verify_openai_instruct_response(responses[2], expected_model=OPENAI_MODEL)
    assert 'Rome' in responses[0].result.reply
    assert 'Paris' in responses[1].result.reply
    assert 'London' in responses[2].result.reply
    total_tokens = sum([x.result.usage_total_tokens for x in responses])
    total_cost = sum([x.result.cost_total for x in responses])
    assert total_cost == cost(total_tokens, model=OPENAI_MODEL)
    assert responses.total_cost == total_cost
    assert responses.total_tokens == sum(r.result.usage_total_tokens for r in responses)
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
    responses[0].result.result['error'] = {'code': 'error'}
    assert responses[0].has_error
    assert not responses[1].has_error
    assert not responses[2].has_error
    assert responses.any_errors

    # test any_missing_data() functionality
    assert not responses.any_missing_data
    assert responses[0].has_data
    assert responses[1].has_data
    assert responses[2].has_data
    # set error within original openai response dict to mock no response
    responses[1].result.result['choices'][0]['text'] = '\n\n'
    assert responses.any_missing_data
    assert responses[0].has_data
    assert not responses[1].has_data
    assert responses[2].has_data
    # set back to original
    responses[1].result.result['choices'][0]['text'] = '\n\nParis'
    assert not responses.any_missing_data
    assert responses[0].has_data
    assert responses[1].has_data
    assert responses[2].has_data
    # set error within original openai response dict to mock no response
    responses[1].result.result['choices'][0]['text'] = None
    assert responses.any_missing_data
    assert responses[0].has_data
    assert not responses[1].has_data
    assert responses[2].has_data


def test__text_completion__invalid_model_type():
    prompts = ["Question: What is the capital of Italy? "]
    oai.API_KEY = 'key'
    with pytest.raises(TypeError):
        oai.text_completion(model='invalid_model', prompts=prompts, max_tokens=10)


def test__complete__missing_api_key(OPENAI_MODEL):
    prompts = [
        "Question: What is the capital of Italy? ",
        "What is the capital of France?",
        "What is the capital of the United Kingdom?",
    ]
    oai.API_KEY = None
    with pytest.raises(oai.MissingApiKeyError):
        oai.text_completion(model=OPENAI_MODEL, prompts=prompts, max_tokens=10)


def test__embeddings(OPENAI_API_KEY):
    inputs = [
        "Question: What is the capital of Italy? ",
        "What is the capital of France?",
        "What is the capital of the United Kingdom?",
    ]
    oai.API_KEY = OPENAI_API_KEY
    responses = oai.text_embeddings(model=EmbeddingModels.ADA, inputs=inputs)
    assert len(responses) == len(inputs)
    assert all([r.response_status == 200 for r in responses])
    assert all([r.response_reason == 'OK' for r in responses])
    assert all([r.result is not None for r in responses])
    assert len(responses[0:2]) == 2

    verify_openai_embedding_response(responses[0], expected_model=EmbeddingModels.ADA)
    verify_openai_embedding_response(responses[1], expected_model=EmbeddingModels.ADA)
    verify_openai_embedding_response(responses[2], expected_model=EmbeddingModels.ADA)

    expected_num_tokens = [len(_encode(x, model=EmbeddingModels.ADA)) for x in inputs]
    actual_num_tokens = [x.result.usage_total_tokens for x in responses]
    assert expected_num_tokens == actual_num_tokens
    total_tokens = sum(actual_num_tokens)
    total_cost = sum([x.result.cost_total for x in responses])
    assert total_cost == cost(total_tokens, model=EmbeddingModels.ADA)
    assert responses.total_cost == total_cost
    assert responses.total_tokens == total_tokens
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
    responses[0].result.result['error'] = {'code': 'error'}
    assert responses[0].has_error
    assert not responses[1].has_error
    assert not responses[2].has_error
    assert responses.any_errors

    # test any_missing_data() functionality
    assert not responses.any_missing_data
    assert responses[0].has_data
    assert responses[1].has_data
    assert responses[2].has_data
    # set error within original openai response dict to mock no response
    responses[1].result.result['data'][0]['embedding'] = []
    assert responses.any_missing_data
    assert responses[0].has_data
    assert not responses[1].has_data
    assert responses[2].has_data
    # set back to original
    responses[1].result.result['data'][0]['embedding'] = [1, 2, 3]
    assert not responses.any_missing_data
    assert responses[0].has_data
    assert responses[1].has_data
    assert responses[2].has_data
    # set error within original openai response dict to mock no response
    responses[1].result.result['data'][0]['embedding'] = None
    assert responses.any_missing_data
    assert responses[0].has_data
    assert not responses[1].has_data
    assert responses[2].has_data


def test__text_embeddings__invalid_model_type():
    inputs = ["Question: What is the capital of Italy? "]
    oai.API_KEY = 'key'
    with pytest.raises(TypeError):
        oai.text_embeddings(model='invalid_model', inputs=inputs, max_tokens=10)


def test__text_embeddings__max_tokens_exceeded():
    inputs = [
        "Question: What is the capital of Italy? ",
        ' '.join((['hello '] * 20_000)),
        "What is the capital of the United Kingdom?",
    ]
    oai.API_KEY = 'key'
    with pytest.raises(ValueError):
        oai.text_embeddings(model=EmbeddingModels.ADA, inputs=inputs, max_tokens=10)


def test__OpenAIResponse():
    response = oai.OpenAIResponse(
        response_status=200,
        response_reason=' OK ',
        result=oai.OpenAIInstructResult(result=None)
    )
    assert response.response_status == 200
    assert response.response_reason == 'OK'
    assert response.result.result == {}

    with pytest.raises(ValueError):
        oai.OpenAIResponse(
            response_status=-1,
            response_reason=' OK ',
            result=oai.OpenAIInstructResult(result=None)
        )
