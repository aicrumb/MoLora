import datetime

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from sentence_transformers import SentenceTransformer
from torch import nn
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)


class Container:
    # i use this all the time i forgot where i stole it from
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class MoLoraForCausalLM(nn.Module):
    def __init__(
        self,
        base_model,
        adapters_repo,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        use_snapshot_download=True,
        device="cuda",
        load_in_4bit=True,
        # these arent really needed for now until we get the adapter merging code in here as well
        rank=32,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        ** kwargs,
    ):
        super().__init__()
        self.device = device
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_model = embedding_model.to(device)
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None

        self.model = AutoModelForCausalLM.from_pretrained(
            (snapshot_download(base_model) if use_snapshot_download else base_model),
            quantization_config=bnb_config,
            **kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.adapters_folder = snapshot_download(adapters_repo)
        self.centers = torch.load(adapters_folder + "centers.pt")
        self.centers = self.centers.to(device)

        # make our model a PEFT Model
        config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        text_model = get_peft_model(text_model, config)

        adapter_names = [i for i in os.listdir(self.adapters_folder) if "." not in i]
        for i in adapter_names:
            text_model.load_adapter(f"{self.adapters_folder}/{i}", adapter_name=f"{i}")

    # todo: def batch_sample_best_adapter
    def sample_best_adapter(prompt, return_expert_logits=False, **kwargs):
        prompt_embedding = torch.tensor(
            self.embedding_model.encode([prompt]), device=self.device
        )
        expert_logits = torch.softmax(
            torch.nn.functional.cosine_similarity(prompt_embedding, self.centers) * 1,
            -1,
        ).tolist()

        for i in zip(adapter_names, expert_logits):
            print(f"{i[0]}: {i[1]}")

        topk = torch.topk(torch.tensor(expert_logits), 1)
        top_adapter = adapter_names[topk.indices.item()]
        text_model.set_adapter(top_adapter)

        # sample
        inputs = {
            k: v.cuda() for k, v in tokenizer(prompt, return_tensors="pt").items()
        }
        outputs = text_model.generate(
            **inputs,
            **kwargs,  # transformers .generate kwargs, e.g. max_new_tokens, temperature, do_sample, etc
        )
        if not return_expert_logits:
            return Container(text=tokenizer.decode(outputs[0]), input_ids=outputs[0])
        return Container(
            text=tokenizer.decode(outputs[0]),
            input_ids=outputs[0],
            expert_logits=expert_logits,
        )

    # ---
    def choose_adapter(context, temperature=1.0, do_sample=False):
        prompt_embedding = torch.tensor(
            self.embedding_model.encode([context]), device=self.device
        )
        expert_logits = torch.softmax(
            torch.nn.functional.cosine_similarity(prompt_embedding, self.centers)
            / temperature,
            -1,
        )
        if do_sample:
            idx = expert_logits.multinomial(1).item()
        else:
            idx = torch.topk(torch.tensor(expert_logits), 1).item()
        chosen_adapter = adapter_names[idx]
        text_model.set_adapter(chosen_adapter)
        return chosen_adapter

    # todo: batch_sample_every_n
    def sample_every_n(
        prompt,
        n=4,
        molora_temp=1.0,
        molora_do_sample=True,
        return_adapter_history=True,
        max_new_tokens=16,
        **kwargs,
    ):
        assert (
            max_new_tokens // n == 0
        ), "`max_new_tokens` argument must be cleanly divisible by argument `n`."

        inputs = {
            k: v.cuda() for k, v in tokenizer(prompt, return_tensors="pt").items()
        }
        input_ids = inputs["input_ids"]
        adapter_history = []

        for i in trange(max_new_tokens // n):
            adapter = choose_adapter(
                tokenizer.decode(input_ids[0]), molora_temp, molora_do_sample
            )
            adapter_history.append(adapter)
            input_ids = text_model.generate(
                input_ids=input_ids, max_new_tokens=n, **kwargs
            )

        if return_adapter_history:
            return Container(
                text=tokenizer.decode(input_ids[0]),
                input_ids=input_ids[0],
                adapter_history=adapter_history,
            )
        return Container(text=tokenizer.decode(input_ids[0]), input_ids=input_ids[0])

    # todo: the merge-sample code could fit here too but i want to accomodate systems in which you have just ANY lora, and that means some have different ranks
