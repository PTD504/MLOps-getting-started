import requests
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import json

API_URL = "http://localhost:8000"

# Sample texts for testing
SAMPLE_TEXTS = [
    "This movie was absolutely amazing! I loved every minute of it.",
    "Terrible film, waste of time and money.",
    "Great acting and storyline. Highly recommended!",
    "Boring and predictable. Not worth watching.",
    "One of the best movies I've ever seen!",
    "Poor script and bad acting. Very disappointing.",
    "Excellent cinematography and direction.",
    "This movie is okay, nothing special.",
    "Fantastic performances by all actors!",
    "I fell asleep watching this movie."
]

def make_prediction(text, model_type="traditional"):
    """Make a prediction request"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text, "model_type": model_type},
            timeout=30
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Request failed: {e}")
        return False

def generate_traffic(duration_seconds=300, requests_per_second=2):
    """Generate traffic for testing"""
    print(f"Generating traffic for {duration_seconds} seconds at {requests_per_second} RPS")
    
    end_time = time.time() + duration_seconds
    request_count = 0
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        while time.time() < end_time:
            # Submit requests
            futures = []
            for _ in range(requests_per_second):
                text = random.choice(SAMPLE_TEXTS)
                model_type = random.choice(["traditional", "lstm"])
                future = executor.submit(make_prediction, text, model_type)
                futures.append(future)
                request_count += 1
            
            # Wait for completion
            for future in futures:
                try:
                    if future.result(timeout=5):
                        success_count += 1
                except Exception:
                    pass
            
            # Wait before next batch
            time.sleep(1)
            
            if request_count % 10 == 0:
                print(f"Sent {request_count} requests, {success_count} successful")
    
    print(f"Traffic generation completed. Total: {request_count}, Successful: {success_count}")

def generate_errors(duration_seconds=60):
    """Generate some errors for testing alerting"""
    print(f"Generating errors for {duration_seconds} seconds")
    
    end_time = time.time() + duration_seconds
    while time.time() < end_time:
        # Send invalid requests
        try:
            requests.post(f"{API_URL}/predict", json={"text": "", "model_type": "invalid"})
            requests.post(f"{API_URL}/predict", json={"invalid": "data"})
        except:
            pass
        time.sleep(2)
    
    print("Error generation completed")

if __name__ == "__main__":
    print("Starting traffic generation...")
    
    # Generate normal traffic
    print("Phase 1: Normal traffic")
    generate_traffic(duration_seconds=120, requests_per_second=3)
    
    # Generate some errors
    print("Phase 2: Error generation")
    generate_errors(duration_seconds=30)
    
    # More normal traffic
    print("Phase 3: More normal traffic")
    generate_traffic(duration_seconds=60, requests_per_second=5)
    
    print("All tests completed!")