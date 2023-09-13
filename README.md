# MoLora

*WIP*

https://twitter.com/aicrumb/status/1681846805959528448

### Example usage

```python
from model import MoLoraForCausalLM

model = MoLoraForCausalLM(
    base_model="gpt2-xl",
    adapters_repo="crumb/gpt2-molora-1.5",  # any repo with folders for each adapter + a centers.pt file which contains a torch tensor shaped (N_adapters, D_embedding_model)
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    use_snapshot_download=True,  # speeds up model downloading
    device="cuda",
    load_in_4bit=False,  # will load model in nf4 precision w/ double quant
)


# finding the top-1 adapter to use throughout the entire sampling process, based on the given context
output = model.sample_best_adapter("Once upon a time", return_expert_logits=True)
expert_logits = output.expert_logits
print(output.text)


# we can also sample for a specific newly computed 'best adapter' every n tokens
output = model.sample_every_n(
    "Once upon a time",
    n=4,
    molora_temp=1,  # high temp values will introduce higher randomness into the expert choices
    molora_do_sample=True,
    return_adapter_history=True,
    # now the normal huggingface params (any! none of these are required besides max_new_tokens, use what you use)
    max_new_tokens=256,
    temperature=0.7,
    top_k=40,
    do_sample=True,
)
output_history = output.history
print(output.text)

```

### On centers.pt

instead of using e.g. a learned linear layer, this implementation of MoLora utilizes the distances to clusters from a KMeans model of the dataset or estimated centers that are the mean of each dataset. [example_cluster.py](https://github.com/aicrumb/MoLora/blob/main/example_cluster.py) includes an example for clustering a dataset for training experts and extracting a centers.pt, but if you have datasets that are completely different that you want to train experts on and compose later, you need to compute a mean embedding for that dataset. Here's some pseudocode (may not work, but general gist of what you need to do)

```python
# this is super inefficient
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

dataset = load_dataset("your_dataset_name")
# to combat efficiency problems here, either batch process these (i have code somewhere, give me a few days) or use dataset=dataset.select(range(n)) to restrict the size of your dataset to like 4000 or something
mean_vector = torch.zeros((1, 384))
for example in dataset:
    your_list = example["text"]
    mean_vector += torch.tensor(your_list) / len(dataset) # idk if dataset has a __len__ method, if you restrict your dataset you should use "n" instead of len(dataset)

torch.save(mean_vector, "center_dset0.pt")

# ----------
# once you have a center.pt for each dataset, load them and concat them like this
centers = torch.cat([vector_0, vector_1, vector_2, ...], 0)
torch.save(centers, "centers.pt")
```
