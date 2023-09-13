# this is adapted hastily from a scratch-pad notebook so apologies for things that could be optimized -crumb
# make sure you're logged in! you can do that with
"""
from huggingface_hub import login
login(token="hf-...")
"""
# or you can do it on the cmd window with
# huggingface-cli login

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tqdm.auto import trange

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
d_model = 384
output_dataset = "username/my_clustered_dataset"

dataset = load_dataset(
    "conceptofmind/cot_submix_original", streaming=False, split="train"
)

column_to_embed = "question"
# this could be 'text' for example, but for instruct datasets the user input column is suggested

# ----------------------------
# create the embeddings that will be clustered, you can set this to dataset.num   rows or whatever
# but i prefer to estimate it with a smaller subset because it's faster

n_samples = 16_384
batch_size = 8
iter_dset = iter(dataset)
embeddings = torch.ones(1, d_model)


for i in trange(n_samples // batch_size):
    batch = [next(iter_dset)[column_to_embed] for _ in range(batch_size)]
    embeddings = torch.cat([embeddings, torch.tensor(model.encode(batch))])

# ---------------------------
# fit KMeans on the embeddings we've computed, then save the centers as centers.pt
kmeans = KMeans(n_clusters=16).fit(embeddings[1:])
centers = torch.tensor(kmeans.cluster_centers_)
torch.save(centers, "centers.pt")

# ----------------------------
# now we can use that kmeans model to cluster the entire dataset
def cluster(batch):
    embedded = torch.tensor(model.encode(batch["question"]))
    cluster = [
        (i.unsqueeze(0) - centers).pow(2).mean(1).argmin().item() for i in embedded
    ]
    return {
        "system_prompt": batch["system_prompt"],
        "question": batch["question"],
        "response": batch["response"],
        "cluster": cluster,
    }


new_data = dataset.map(lambda batch: cluster(batch), batched=True)
new_data.push_to_hub(output_dataset)

# we also should push centers.pt to the dataset
from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="centers.pt",
    path_in_repo="centers.pt",
    repo_id=output_dataset,
    repo_type="dataset",
)
