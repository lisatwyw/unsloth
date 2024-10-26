
# 1. Terminal session (not interactive)

<details>
    

Launch ipython:
```
ipython
```

## 1b. Download dataset
```
from datasets import load_dataset  
dataset = load_dataset("yahma/alpaca-cleaned")
dataset.save_to_disk( '/scratch/st-username-1/huggingface/yahma-alpaca-cleaned')
```

## 1c. Download model

```
kaggle datasets download -d lisatwyw/alpaca-trained-llms/ 
```

</details>


# 2. Interactive session with GPU

<details>

Launch ipython:
```
ipython
```

## 2b. Load model from disk 

```
model,tokenizer = FastLanguageModel.from_pretrained( 'not_trained/' )
```

## 2c. Load dataset from disk 
```
from datasets import load_from_disk
dataset = load_from_disk('/scratch/st-username-1/huggingface/yahma-alpaca-cleaned/')
dataset = dataset.map(formatting_prompts_func, batched = True,)
```

```
os.environ['TRITON_CACHE_DIR'] = '/scratch/st-username-1/huggingface/' # important !

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 201, # Set num_train_epochs = 1 for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
trainer_stats = trainer.train()
```


Few moments later...


## Save the fined-tuned model as single file, merger of tokenizer and model weights
```
model.save_pretrained_merged( 'mistral7b-16bit', tokenizer, save_method = "merged_16bit",)
```

</details>



## 3. Inference 

```
q = alpaca_prompt.format(
    instruction, # instruction
    query, # input
    "", # output - leave this blank for generation!
    )

inputs = tokenizer( q, return_tensors = "pt").to("cuda")
outputs = tokenizer.batch_decode( model.generate(**inputs, max_new_tokens = 64, use_cache = True) )

```
