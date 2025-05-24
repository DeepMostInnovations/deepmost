from deepmost import sales

# Predict conversion probability
conversation = ["Hi, I need a CRM for my business", "I'd be happy to help! What's your team size?"]
probability = sales.predict(conversation)
print(f"Conversion probability: {probability:.1%}")  # Output: "Conversion probability: 73.5%"