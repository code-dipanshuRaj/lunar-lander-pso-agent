import numpy as np

def policy_action(params, observation):
    # Reconstruct first layer: shape (8, 64)
    W1 = params[:8*64].reshape(8, 64)
    b1 = params[8*64:8*64+64]
    
    # Reconstruct second layer: shape (64, 4)
    W2 = params[8*64+64:8*64+64+64*4].reshape(64, 4)
    b2 = params[8*64+64+64*4:]
    
    # Forward pass
    hidden = np.maximum(0, np.dot(observation, W1) + b1)  # ReLU activation
    logits = np.dot(hidden, W2) + b2
    return np.argmax(logits)
