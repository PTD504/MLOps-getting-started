# traffic_generator.py
import requests
import time
import random

API_URL = "http://localhost:8000/predict"
ERROR_TEST_URL = "http://localhost:8000/error_test"

sample_texts = [
    "This is a wonderful movie, I loved every minute of it!",
    "I'm so happy with this product, it's amazing.",
    "The weather today is just perfect for a walk.",
    "What a terrible experience, I would not recommend this.",
    "I am very disappointed with the service I received.",
    "This is the worst food I have ever tasted.",
    "This is neutral.",
    "I don't know what to think about this."
]

def generate_traffic(duration_seconds=600, error_rate_percent=5):
    print(f"Generating traffic for {duration_seconds} seconds with ~{error_rate_percent}% error rate...")
    start_time = time.time()
    request_count = 0
    error_count = 0

    while time.time() - start_time < duration_seconds:
        try:
            # Decide if this request should be an error
            if random.randint(1, 100) <= error_rate_percent:
                print("Simulating an error request...")
                response = requests.get(ERROR_TEST_URL, timeout=5)
                if response.status_code >= 400: # Could be 500 for our test
                    print(f"Error test returned: {response.status_code}")
                    error_count +=1
                else:
                    print(f"Error test unexpected success: {response.status_code}")
            else:
                text_to_send = random.choice(sample_texts)
                payload = {"text": text_to_send}
                print(f"Sending: {payload}")
                response = requests.post(API_URL, json=payload, timeout=5)
                response.raise_for_status() # Raise an exception for bad status codes
                print(f"Response: {response.json()}")
            
            request_count += 1
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            error_count += 1 # Count connection errors or HTTP errors not caught by error_test
        
        time.sleep(random.uniform(0.1, 1.5)) # Random delay between requests

    print(f"\nTraffic generation finished. Total requests: {request_count}, Errors: {error_count}")

if __name__ == "__main__":
    print(">>> GENERATING ERRORS FOR DEMO <<<")
    generate_traffic(duration_seconds=120, error_rate_percent=70) # Run for 1 minute, 50% error rate

    # generate_traffic(duration_seconds=1200, error_rate_percent=30) # Run for 5 minutes, 10% error rate
    
    # For higher error rate to trigger alert:
    # In traffic_generator.py, if __name__ == "__main__":
    # print(">>> SIMULATING HIGH ERROR RATE FOR ALERT DEMO <<<")
    # generate_traffic(duration_seconds=180, error_rate_percent=70) # 3 mins, 70% error rate
    