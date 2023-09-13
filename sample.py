from molora import MoLoraForCausalLM

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
