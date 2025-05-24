import sys
from deepmost import sales

def test_system_info():
    """Test system information"""
    info = sales.get_system_info()
    print(f"Python version: {info['python_version']}")
    print(f"CUDA available: {info['cuda_available']}")
    print(f"Supported backends: {info['supported_backends']}")

def test_basic_prediction():
    """Test basic prediction functionality"""
    try:
        conversation = ["Hi, I need a CRM", "Hello, how can I help?"]
        
        # Test the quick predict function
        probability = sales.predict(conversation, auto_download=False)
        print(f"Import successful! Probability: {probability}")
        
        # Test the Agent API
        agent = sales.Agent(auto_download=False)
        result = agent.predict(conversation)
        print(f"Agent prediction: {result}")
        
    except Exception as e:
        print(f"Error during prediction test: {e}")
        print("This is expected if no model is downloaded yet.")

if __name__ == "__main__":
    print("=== DeepMost Test ===")
    test_system_info()
    print("\n=== Basic Prediction Test ===")
    test_basic_prediction()