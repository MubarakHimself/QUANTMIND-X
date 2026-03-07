"""
Test script to verify GLM and Minimax API connections.
Run with: python -m tests.test_provider_connections
"""
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

from src.agents.llm_provider import (
    ProviderType,
    PROVIDER_BASE_URLS,
    API_KEY_ENV_VARS,
    has_api_key,
    get_api_key,
)


def test_provider_config():
    """Test that GLM and Minimax providers are properly configured."""
    print("=" * 60)
    print("Provider Configuration Test")
    print("=" * 60)

    # Test GLM (ZHIPU) configuration
    print("\n[GLM/ZHIPU Provider]")
    print(f"  Provider Type: {ProviderType.ZHIPU.value}")
    print(f"  Base URL: {PROVIDER_BASE_URLS[ProviderType.ZHIPU]}")
    print(f"  API Key Env: {API_KEY_ENV_VARS[ProviderType.ZHIPU]}")
    print(f"  Has API Key: {has_api_key(ProviderType.ZHIPU)}")

    # Test Minimax configuration
    print("\n[Minimax Provider]")
    print(f"  Provider Type: {ProviderType.MINIMAX.value}")
    print(f"  Base URL: {PROVIDER_BASE_URLS[ProviderType.MINIMAX]}")
    print(f"  API Key Env: {API_KEY_ENV_VARS[ProviderType.MINIMAX]}")
    print(f"  Has API Key: {has_api_key(ProviderType.MINIMAX)}")

    # Test all available providers
    print("\n[All Available Providers]")
    for provider in ProviderType:
        has_key = has_api_key(provider)
        print(f"  {provider.value}: {'✓' if has_key else '✗'}")

    return True


async def test_api_connection(provider: ProviderType) -> bool:
    """Test actual API connection for a provider."""
    api_key = get_api_key(provider)
    if not api_key:
        print(f"  No API key for {provider.value}")
        return False

    import httpx

    base_url = PROVIDER_BASE_URLS[provider]

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if provider == ProviderType.ZHIPU:
                # Test GLM API - use correct model name format
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "glm-4-plus",
                        "messages": [{"role": "user", "content": "Say 'test successful'"}],
                        "max_tokens": 50,
                    },
                )
            elif provider == ProviderType.MINIMAX:
                # Test Minimax API
                response = await client.post(
                    f"{base_url}/text/chatcompletion_v2",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "MiniMax-M2.1",
                        "messages": [{"role": "user", "content": "Say 'test successful'"}],
                        "max_tokens": 50,
                    },
                )

            if response.status_code == 200:
                print(f"  ✓ {provider.value} API connection successful!")
                return True
            else:
                print(f"  ✗ {provider.value} API error: {response.status_code}")
                print(f"    {response.text[:200]}")
                return False

    except Exception as e:
        print(f"  ✗ {provider.value} connection failed: {str(e)}")
        return False


async def main():
    """Main test function."""
    # Test configuration
    test_provider_config()

    print("\n" + "=" * 60)
    print("API Connection Tests")
    print("=" * 60)

    # Test GLM API
    print("\n[Testing GLM API Connection]")
    glm_result = await test_api_connection(ProviderType.ZHIPU)

    # Test Minimax API
    print("\n[Testing Minimax API Connection]")
    minimax_result = await test_api_connection(ProviderType.MINIMAX)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"GLM (ZHIPU): {'✓ PASSED' if glm_result else '✗ FAILED'}")
    print(f"Minimax: {'✓ PASSED' if minimax_result else '✗ FAILED'}")

    return glm_result and minimax_result


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
