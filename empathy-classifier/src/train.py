import torch
import torch.nn as nn
import sys
import os
# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))
from model import BaselineModel
from data import get_dataloaders

def soft_cross_entropy(pred, target):
    return -torch.sum(target * torch.log(pred + 1e-8), dim=1).mean()

def train_model(max_epochs=50, patience=5, batch_size=32, learning_rate=1e-3, 
                model_save_path=None, verbose=True):
    """Train the model with specified hyperparameters.
    
    Returns:
        best_eval_loss: Best evaluation loss achieved
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    train_loader, eval_loader, _ = get_dataloaders(batch_size=batch_size)
    
    # Model
    model = BaselineModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    best_eval_loss = float('inf')
    patience_counter = 0
    
    if verbose:
        print("Starting training...")
        print(f"Max epochs: {max_epochs}, Early stopping patience: {patience}, "
              f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            probs = model(input_ids, attention_mask)
            loss = soft_cross_entropy(probs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluation
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                probs = model(input_ids, attention_mask)
                loss = soft_cross_entropy(probs, labels)
                eval_loss += loss.item()
        
        avg_eval_loss = eval_loss / len(eval_loader)
        
        if verbose:
            print(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")
        
        # Early stopping
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            patience_counter = 0
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    if verbose:
        print("Training completed.")
    
    return best_eval_loss

# Default hyperparameters
MAX_EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# Run training if executed directly
if __name__ == '__main__':
    # Try to find models directory - works both locally and in Colab
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, '..', 'models', 'baseline_v1.pt')
    
    # If parent doesn't exist, try from current working directory
    if not os.path.exists(os.path.dirname(model_path)):
        model_path = os.path.join(os.getcwd(), 'models', 'baseline_v1.pt')
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    train_model(
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        model_save_path=model_path
    )

