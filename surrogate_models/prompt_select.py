import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.init as init
class PromptSelector(nn.Module):
    def __init__(self, args,device,feature_dim):
        super(PromptSelector, self).__init__()
        self.feature_dim = feature_dim
        self.device = device
        self.args = args
        self.num_prompts = self.args.num_prompts
        # self.gcn = GCNConv(feature_dim, feature_dim)  # GCN layer
        self.weight_transform = nn.Linear(feature_dim, args.prompt_size * feature_dim).to(device)  # Transform query vector to 3*feature
        self.classifier = nn.Linear(feature_dim, 1).to(device)  # Classifier to reduce dimensionality to 1
    
    def forward(self,prompt_pool, query_vector):
       
        prompt_pool = prompt_pool.to(self.device)
        prompt_pool = prompt_pool.view(self.num_prompts, self.args.prompt_size, -1)
        query_vector = query_vector.to(self.device)
        
        # Step 2: Transform query vector
        transformed_query = self.weight_transform(query_vector)  # 1*feature to 3*feature
        transformed_query = transformed_query.view(self.args.prompt_size, -1)  # Reshape to 3*feature for multiplication
        
        # Step 3: Dot product between transformed query and each prompt
        scores = torch.einsum('nij,ij->nj', prompt_pool, transformed_query)  # num*feature
        
        # Step 4: Classify each prompt score to one scalar
        scores = self.classifier(scores.view(-1, self.feature_dim))  # Flatten and apply linear layer
        
        scores = scores.view(self.num_prompts, -1).sum(dim=1)  # Sum scores for each prompt
        
        # Step 5: Softmax to select the highest score
        prompt_probabilities = F.softmax(scores, dim=0)
        _, max_index = torch.max(prompt_probabilities, dim=0)

        # Return the softmax probabilities and the index of the highest probability
        return  max_index.item() 
        

# # Example usage
# num_prompts = 10
# feature_dim = 100
# model = PromptSelector(feature_dim, num_prompts)
# prompt_pool = torch.rand(num_prompts, 3, feature_dim)
# query_vector = torch.rand(1, feature_dim)
# edge_index = torch.tensor([...], dtype=torch.long)  # Define your edge connections here

# prompt_probabilities = model(prompt_pool, query_vector, edge_index)
# selected_prompt_idx = prompt_probabilities.argmax()
