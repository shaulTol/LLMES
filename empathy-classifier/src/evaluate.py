import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))
from model import BaselineModel
from data import get_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
import os
# Try to find model file - works both locally and in Colab
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, '..', 'models', 'baseline_v1.pt')

# If not found, try absolute path from current working directory
if not os.path.exists(model_path):
    model_path = os.path.join(os.getcwd(), 'models', 'baseline_v1.pt')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Could not find model file. Tried: {model_path}")

model = BaselineModel().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Get test loader
_, _, test_loader = get_dataloaders(batch_size=32)

# Evaluation
all_pred_hard = []
all_true_hard = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        probs = model(input_ids, attention_mask)
        pred_hard = torch.argmax(probs, dim=1).cpu().numpy()
        true_hard = torch.argmax(labels, dim=1).cpu().numpy()
        
        all_pred_hard.extend(pred_hard)
        all_true_hard.extend(true_hard)

all_pred_hard = np.array(all_pred_hard)
all_true_hard = np.array(all_true_hard)

# Metrics
accuracy = accuracy_score(all_true_hard, all_pred_hard)
print(f"Test Accuracy: {accuracy:.4f}")

# Per-class accuracy
class_names = ['Cognitive', 'Affective', 'Motivational']
for i, name in enumerate(class_names):
    class_mask = all_true_hard == i
    if class_mask.sum() > 0:
        class_acc = accuracy_score(all_true_hard[class_mask], all_pred_hard[class_mask])
        print(f"{name} Accuracy: {class_acc:.4f} ({class_mask.sum()} samples)")

# Confusion matrix
cm = confusion_matrix(all_true_hard, all_pred_hard)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
# Save confusion matrix - works both locally and in Colab
script_dir = os.path.dirname(__file__)
output_path = os.path.join(script_dir, '..', 'outputs', 'baseline_v1_confusion.png')

# If parent doesn't exist, try from current working directory
if not os.path.exists(os.path.dirname(output_path)):
    output_path = os.path.join(os.getcwd(), 'outputs', 'baseline_v1_confusion.png')

os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"\nConfusion matrix saved to {output_path}")

