import signal
import time
from openai import OpenAI


# Define the OpenAI client call within the make_request function
def make_request(
    message: str,
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int = 512,
    temperature: float = 1,
    n: int = 1,
    **kwargs,
) -> str:
    """
    Makes a request to the OpenAI API for chat completions.

    Parameters:
    - message (str): The input message or question to be processed.
    - model (str): The model to be used for the request.
    - max_tokens (int): The maximum number of tokens to generate in the response.
    - temperature (float): The temperature to use for the request, controlling randomness.
    - n (int): The number of responses to generate.
    - **kwargs: Additional keyword arguments to pass to the API call.

    Returns:
    - dict: The response from the OpenAI API.
    """

    def call_api(message, max_tokens, temperature, api_key, base_url):
        # Initialize the OpenAI client (Note: api_key should be kept secure and not hard-coded)
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        while True:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an AI assistant that helps people find information.",
                        },
                        {"role": "user", "content": message},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response
            except Exception as e:
                print(e)
                time.sleep(2)

    response = call_api(
        message,
        max_tokens,
        temperature,
        api_key,
        base_url,
    )
    return response.choices[0].message.content


def handler(signum, frame):
    """
    Signal handler to raise an exception when an alarm is received.

    Parameters:
    - signum (int): The signal number.
    - frame (frame): The current stack frame.
    """
    raise Exception("end of time")


def make_auto_request(*args, **kwargs) -> dict:
    """
    Makes an auto request with a timeout to the OpenAI API.

    Parameters:
    - *args: Variable arguments to be passed to make_request.
    - **kwargs: Keyword arguments to be passed to make_request.

    Returns:
    - dict: The response from the OpenAI API.
    """

    ret = None
    while ret is None:
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(100)  # Set an alarm for 100 seconds
            ret = make_request(*args, **kwargs)
            signal.alarm(0)  # Disable the alarm
        except signal.AlarmClockError:  # Alarm signal received
            print("Request timed out")
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(1)  # Wait before retrying
    return ret
