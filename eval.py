import asyncio
import httpx
from langsmith import Client, aevaluate  # Import aevaluate
from datetime import datetime

import requests
import json


def get_salesforce_access_token():
    """
    Authenticate with Salesforce and retrieve the access token.

    Returns:
        str: Access token if successful, None otherwise
    """
    # Salesforce OAuth endpoint
    url = 'https://test.salesforce.com/services/oauth2/token'

    # Headers
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    # Request data
    data = {
        'grant_type': 'password',
        'client_id': '3MVG9dPGzpc3kWycYmOSXd.i6Qcz506lRlPLEdDJBz5f9Ll.NFDHfsNKKP82DUo3KEbv57qyMHdm7Zt0VvXMb',
        'client_secret': '7463971904206904573',
        'username': 'qa@llm.com',
        'password': 'Service1'
    }

    try:
        # Make POST request
        response = requests.post(url, headers=headers, data=data)

        # Check if request was successful
        response.raise_for_status()

        # Parse JSON response
        response_data = response.json()

        # Extract access token
        access_token = response_data.get('access_token')

        if access_token:
            # print("Authentication successful!")
            # print(f"Access Token: {access_token}")
            # print(f"\nFull Response:")
            # print(json.dumps(response_data, indent=2))
            return access_token
        else:
            print("No access token found in response")
            return None

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response: {response.text}")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
        return None
    except json.JSONDecodeError as json_err:
        print(f"JSON decode error: {json_err}")
        print(f"Response text: {response.text}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None


async def evaluate_chat(inputs: dict) -> dict:
    """
    Takes dataset inputs and calls your actual API endpoint
    inputs: {"input": "who was the last tech"}
    returns: {"answer": "response from your API"}
    """

    # Build the full payload from your dataset input
    payload = {
        "origin": "Go",
        "runtype": "Chat Response",
        "user_message": {
            "talker_id": "68374495",
            "role": "user",
            "message": inputs["input"],  # From your dataset
            "timestamp": datetime.utcnow().isoformat()
        },
        "conversation": [],
        "context": {
            "conversation_id": 1,
            "entity": "a1gO4000009gcF3IAI",
            "entity_resource": "SVMXC__Service_Order__c",
            "entity_name": "WO-00038665",
            "user_locale": "en_US",
            "user_time_zones": {
                "68374495": "Asia/Shanghai"
            }
        }
    }

    access_token = "Bearer " + get_salesforce_access_token()

    headers = {
        "Authorization": "Bearer your-token-here",
        "Content-Type": "application/json",
        "X-Auth-Type": "Salesforce",
        "X-Auth-Origin": "Sandbox",
        "from": "SvmxPtc@4450",
        "Authorization": access_token
    }

    # Call your actual FastAPI endpoint
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://aig-int.servicemax-api.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        result = response.json()

        # Extract the answer from your API response
        # Adjust this based on what your endpoint actually returns
        return {"answer": result.get("message") or result.get("response") or str(result)}


async def run_evaluation():
    # Use aevaluate for async functions
    results = await aevaluate(
        evaluate_chat,
        data="qallm-dataset",
        experiment_prefix="qallm-endpoint-eval",
    )

    print(f"✅ Evaluation complete!")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
