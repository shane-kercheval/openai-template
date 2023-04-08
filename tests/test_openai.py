import pytest
import aiohttp
import source.library.openai as openail


@pytest.fixture(scope='session')
def OPENAI_API_KEY() -> str:
    with open('/.openai_api_key', 'r') as handle:
        openai_api_key = handle.read().strip()
    return openai_api_key


@pytest.fixture(scope='session')
def OPENAI_URL_COMPLETION() -> str:
    return 'https://api.openai.com/v1/completions'


@pytest.fixture(scope='session')
def babbage_model_payload__italy_capital() -> dict:
    return {
        'model': 'text-babbage-001',
        'prompt': "What is the capital of Italy?",
        'max_tokens': 50
    }


@pytest.mark.asyncio
async def test__api_post(
        OPENAI_API_KEY,
        OPENAI_URL_COMPLETION,
        babbage_model_payload__italy_capital):
    assert OPENAI_API_KEY
    openail.API_KEY = OPENAI_API_KEY

    async with aiohttp.ClientSession() as session:
        result = await openail.post_async(
            session=session,
            url=OPENAI_URL_COMPLETION,
            payload=babbage_model_payload__italy_capital,
        )
    assert result.response_status == 200
    assert result.response_reason == 'OK'
    assert 'Rome' in result.openai_result['choices'][0]['text']


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
    assert result.openai_result['error'] is not None
