# Retreive the last hidden state of a single string
# - Encode the string & convert tokens to tensor (return tensors = pt)
# - Get the output of the model 

import warnings; warnings.filterwarnings('ignore')
from transformers import AutoModel
from transformers import AutoTokenizer
import torch

# Load parameters of the tokeniser
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

# print(model)
# (output_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)

text = "this is a test"
inputs = tokenizer(text, return_tensors="pt")
print(f"Input tensor shape: {inputs['input_ids'].size()}") # (batch size,n_tokens)


print('\ninput items')
print(inputs.items())

inputs = {k:v.to(device) for k,v in inputs.items()}

# prediction/inference 

with torch.no_grad():
    outputs = model(**inputs)
    
# Depending on model, output can contain different info
print('\noutput\n')
print(outputs,'\n')

# hidden state tensors (batch,n_tokens,hidden_dim)
# 768 dim vector returned for each of the 6 input tokens
print(outputs.last_hidden_state.size())
print(outputs.last_hidden_state[:,0].size())

dict_data = {"hidden_state": outputs.last_hidden_state[:,0].cpu().numpy()}
dict_data
