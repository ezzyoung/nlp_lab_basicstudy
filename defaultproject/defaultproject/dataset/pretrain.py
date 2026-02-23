import os
import datasets
import itertools

def prepare_pretrain_dataset(tokenizer,
                             dataset_name_or_path: str = "chengjunyan1/smollm-12.5-corpus",
                             dataset_subset: str= "cosmopedia-v2",
                             max_length: int = 1024,
                             text_column_name: str = "text",
                             train_sample_size: int =-1,
                             eval_sample_size: int =-1,
                             cache_path:str="cache"):
    if cache_path is not None and os.path.exists(os.path.join(cache_path,"pretrain")):
        print(f"Using pre-downloaded dataset from {cache_path}.")
        train = datasets.load_from_disk(os.path.join(cache_path,"pretrain","train"))
        validation = datasets.load_from_disk(os.path.join(cache_path,"pretrain","eval"))
        return train, validation
    else:
        print(f"Download and prerpocessing dataset from {dataset_name_or_path} on subset {dataset_subset}...")
        dataset = datasets.load_dataset(dataset_name_or_path, dataset_subset)
        if "eval" in dataset:
            validation_split = "eval"
        elif "test" in dataset:
            validation_split = "test"
        else:
            print("Using train split for validation.")
            dataset = datasets.train_test_split(test_size=5000)
            validation_split = "test"
        
        train = dataset["train"] if train_sample_size == -1 else dataset["train"].select(range(train_sample_size))
        validation = dataset[validation_split] if eval_sample_size == -1 else dataset[validation_split].select(range(eval_sample_size))

        train = _packing_dataset(train, text_column_name, tokenizer, max_length)
        validation = _packing_dataset(validation, text_column_name, tokenizer, max_length)

        if cache_path is not None:
            os.makedirs(os.path.join(cache_path,"pretrain"), exist_ok=True)
            print(f"Saving dataset to {cache_path}...")
            train.save_to_disk(os.path.join(cache_path,"pretrain","train"), max_shard_size="500MB")
            validation.save_to_disk(os.path.join(cache_path,"pretrain","eval"), max_shard_size="500MB")
        return train, validation

def _packing_dataset(dataset, text_column_name, tokenizer, block_length):
    origin_model_length = tokenizer.model_max_length
    tokenizer.model_max_length = 987654321
    def tokenizing_fn(examples):
        texts = [tokenizer.bos_token + text + tokenizer.eos_token for text in examples[text_column_name]]
        return tokenizer(texts, add_special_tokens=False)
    
    tokenized_dataset = dataset.map(
        tokenizing_fn,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=32,
        desc="Tokenizing",
    )

    def grouping_fn(examples):
        concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_length) * block_length
        result = {
            k: [t[i : i + block_length] for i in range(0, total_length, block_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    grouped_dataset = tokenized_dataset.map(
        grouping_fn,
        batched=True,
        batch_size=1000,
        num_proc=32,
        desc="Grouping",
    )
    tokenizer.model_max_length = origin_model_length
    return grouped_dataset

def collate_fn_for_pretrain(batch, tokenizer, block_length):
    input_ids = [example["input_ids"] for example in batch]
    attention_mask = [example["attention_mask"] for example in batch]
    model_inputs = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        padding="max_length",
        max_length=block_length,
        return_tensors="pt",
    )
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    model_inputs['labels'][model_inputs['attention_mask'] == 0] = -100
    return model_inputs