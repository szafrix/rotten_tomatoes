from vllm import LLM, SamplingParams
from datasets import load_from_disk
import pandas as pd
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
import torch
import gc

with open("classification/models/llms/prompt.txt", "r") as f:
    prompt = f.read()


sampling_params = SamplingParams(
    temperature=0, top_p=1, top_k=-1, max_tokens=256, stop="</verdict>"
)

models = [
    "BAAI/AquilaChat-7B",
    "baichuan-inc/Baichuan-7B",
    "tiiuae/falcon-7b",
    "stabilityai/stablelm-tuned-alpha-7b",
    "internlm/internlm-chat-7b",
    "mistralai/Mistral-7B-v0.1",
    "01-ai/Yi-6B",
    "mosaicml/mpt-7b",
    "Qwen/Qwen-7B-Chat",
]

data_folder_val = "data/dataset_tokenized/validation"
val = load_from_disk(data_folder_val).to_pandas()

results = []

for model in models:

    try:
        llm = LLM(
            model=model,
            trust_remote_code=True,
            gpu_memory_utilization=0.97,
        )
    except Exception as exc:
        print("ERROR:", model, exc)

    else:
        try:
            prompts = [prompt.format(sentence=text) for text in val["text"].values]

            outputs = llm.generate(prompts, sampling_params)
            outputs = [o.outputs[0].text + "</verdict>" for o in outputs]
            for p, o in zip(val["text"].values, outputs):
                print("OUTPUT: ", o)
                print("\n")
                print("---" * 30)

            results.append(
                pd.DataFrame(
                    {
                        "texts": val["text"].values,
                        "labels": val["label"].values,
                        "llm_completion": outputs,
                        "llm_name": model,
                    }
                )
            )
        except Exception as exc:
            print("ERROR:", model, exc)

        finally:
            destroy_model_parallel()
            del llm
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.destroy_process_group()

pd.concat(results).to_pickle("classification/models/llms/results")
