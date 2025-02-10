import requests
import json
from typing import List, Dict
import time

def test_inference(
    url: str = "http://localhost:8000",
    metrics: List[Dict] = None,
    logs: List[Dict] = None,
    kpi_entity_id: str = "entity_0"
):
    """Test the inference endpoint"""
    
    # Sample data if none provided
    if metrics is None:
        metrics = [
            {
                "timestamp": int(time.time()),
                "entity_id": f"entity_{i}",
                "values": {
                    "cpu_usage": 0.8,
                    "memory_usage": 0.7,
                    "disk_usage": 0.6
                }
            }
            for i in range(5)
        ]
    
    if logs is None:
        logs = [
            {
                "timestamp": int(time.time()),
                "entity_id": f"entity_{i}",
                "message": "High CPU usage detected",
                "level": "WARNING"
            }
            for i in range(5)
        ]
    
    # Prepare request data
    request_data = {
        "metrics": metrics,
        "logs": logs,
        "kpi_entity_id": kpi_entity_id
    }
    
    try:
        # Check health
        health_response = requests.get(f"{url}/health")
        print(f"Health check response: {health_response.json()}")
        
        # Make prediction
        response = requests.post(
            f"{url}/predict",
            json=request_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\nPrediction Results:")
            print("Root Causes:")
            for rc in result["root_causes"]:
                print(f"  Entity: {rc['entity_id']}, Probability: {rc['probability']:.4f}")
            print(f"Should Stop: {result['should_stop']}")
            print(f"Confidence: {result['confidence']:.4f}")
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_inference() 