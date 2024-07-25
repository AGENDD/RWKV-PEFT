
# Load the state dict from the pth file
import sys
input_weight = sys.argv[1]
output_weight = sys.argv[2]

import torch
state_dict = torch.load(input_weight)

# Create a new state dict to store the filtered weights
filtered_state_dict = {}

# Iterate over the keys in the state dict
for key in state_dict.keys():
    # Check if the key matches any of the commented weights
    if key.startswith('language_model.blocks.') and "att.time_state" in key:
        # Add the key and value to the filtered state dict
        filtered_state_dict[key] = state_dict[key]
    elif key.startswith('speech_encoder.adapter.'):
        filtered_state_dict[key] = state_dict[key]

# Save the filtered state dict to a new file
torch.save(filtered_state_dict, output_weight)
