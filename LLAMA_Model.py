import math
import os
import sys

# Suppress tokenizer parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pandas as pd
import numpy
import torch
from peft import (
    prepare_model_for_kbit_training,
    TaskType, 
    LoraConfig, 
    AdaLoraConfig, 
    PrefixTuningConfig, 
    PromptEncoderConfig, 
    PromptTuningConfig, 
    get_peft_model, 
    PeftModel
)
from torch import nn
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    LlamaTokenizer, 
    LlamaForCausalLM, 
    get_linear_schedule_with_warmup, 
    BitsAndBytesConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer
)
from custom_datasets import GPTDataset, cot_prompt_pre
import bitsandbytes as bnb
import evaluate
from torch.nn.utils.rnn import pad_sequence

def find_all_linear_names(model):
    """
    Find all fully connected layers and add adapters to all fully connected layers.
    """
    # cls = bnb.nn.Linear4bit
    cls = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

class LLAMASeq2Seq():

    def __init__(self, base_model_path, add_eos_token=False, adapter="lora", load_adapter_path="None", source_len=300, cutoff_len=512):
        print("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()
        self.base_model = base_model_path
        self.add_eos_token = add_eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == 'cpu':
            raise RuntimeError("CUDA GPU not available. This script is configured to run on GPU only.")
        print(f"***** Running on device: {self.device} *****")

        self.adapter = adapter
        self.load_adapter_path = load_adapter_path
        self.cutoff_len = cutoff_len
        self.source_len = source_len
        # Initialize LLM model
        self.model, self.tokenizer = self.get_model_tokenizer()

        # Prepare model for k-bit training if quantization is used
        self.model = prepare_model_for_kbit_training(self.model)
        # Explicitly set use_reentrant for gradient checkpointing
        if hasattr(self.model, "gradient_checkpointing_enable") and hasattr(self.model, "is_gradient_checkpointing"):
            if self.model.is_gradient_checkpointing: # Only if it was actually enabled by prepare_model_for_kbit_training
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # Initialize adapter
        if self.load_adapter_path == "None":
            self.model = self.load_adapter_config(self.model)

        # Load the trained adapter
        if self.load_adapter_path != "None":
            self.model = PeftModel.from_pretrained(
                self.model,
                self.load_adapter_path
            )

        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        self.model.to(self.device)
        # Lazily load metrics to avoid loading them if not used
        self.sacrebleu_metric = None
        self.rouge_metric = None

    def _create_collate_fn(self):
        def collate_fn(batch):
            input_ids = [item[0] for item in batch]
            labels = [item[1] for item in batch]

            # Pad input_ids
            # self.tokenizer.padding_side = "left" # For decoder-only models, left padding is often preferred.
            padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

            # Pad labels
            padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

            return padded_input_ids, padded_labels
        return collate_fn

    def get_model_tokenizer(self):

        # Setup quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            add_eos_token=self.add_eos_token
        )  # default add_eos_token=False
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def load_adapter_config(self, model):
        t_type = TaskType.CAUSAL_LM

        if self.adapter == "lora":
            target_modules = find_all_linear_names(model)
            print(target_modules)
            config = LoraConfig(
                task_type=t_type,
                inference_mode=False,
                lora_dropout=0.05,
                r=8,
                lora_alpha=16,
                target_modules=target_modules
            )
        elif self.adapter == 'adalora':
            config = AdaLoraConfig(
                task_type=t_type,
                inference_mode=False,
            )
        elif self.adapter == "prefix":
            config = PrefixTuningConfig(
                task_type=t_type,
                prefix_projection=True
            )
        elif self.adapter == "p_tuning":
            config = PromptEncoderConfig(
                task_type=t_type
            )
        elif self.adapter == "prompt":
            config = PromptTuningConfig(
                task_type=t_type
            )
        else:
            raise KeyError("Unknow adapter: {}".format(self.adapter))

        model = get_peft_model(model, config)
        model.print_trainable_parameters()

        return model

    def train(self, train_filename, train_batch_size, learning_rate, num_train_epochs, early_stop,
              do_eval, eval_filename, eval_batch_size, output_dir, do_eval_bleu,
              train_nrows=None, eval_nrows=None):

        train_data = GPTDataset(train_filename, tokenizer=self.tokenizer, source_len=self.source_len, cutoff_len=self.cutoff_len, nrows=train_nrows)

        train_sampler = RandomSampler(train_data)
        # train_dataloader = DataLoader(train_data, sampler=train_sampler,
        #                               batch_size=train_batch_size)
        
        collate_fn = self._create_collate_fn()
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=train_batch_size, collate_fn=collate_fn)

        # Prepare optimizer and schedule (linear warmup and decay)
        t_total = len(train_dataloader) // num_train_epochs
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        num_train_optimization_steps = num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        print("***** Running training *****")
        print("  Num examples = %d", train_example_num)
        print("  Batch size = %d", train_batch_size)
        print("  Batch num = %d", math.ceil(train_example_num / train_batch_size))
        print("  Num epoch = %d", num_train_epochs)

        global_step, best_bleu, best_loss = 0, -1.0, 1e6
        count = 0

        for cur_epoch in range(int(num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            self.model.train()
            for step, (input_ids, token_labels) in enumerate(bar):
                input_ids = input_ids.to(self.device)
                labels = token_labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                tr_loss += loss.item()
                nb_tr_steps += 1

                loss.backward()

                optimizer.step()
                scheduler.step()

                global_step += 1
                train_loss = round(tr_loss * 1 / (nb_tr_steps + 1), 4)
                bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

            if do_eval:
                eval_data = GPTDataset(eval_filename, tokenizer=self.tokenizer, source_len=self.source_len, cutoff_len=self.cutoff_len, nrows=eval_nrows)
                eval_sampler = SequentialSampler(eval_data)
                # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size, collate_fn=collate_fn)

                print("***** Running evaluation  *****")
                print("  Num examples = %d", eval_data.__len__())
                print("  Batch size = %d", eval_batch_size)
                print("  Num epoch = %d", cur_epoch)
                self.model.eval()
                eval_loss, batch_num = 0, 0
                for step, (input_ids, token_labels) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
                    input_ids = input_ids.to(self.device)
                    labels = token_labels.to(self.device)

                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids, labels=labels)
                        loss = outputs.loss
                    eval_loss += loss.mean().item()
                    batch_num += 1
                self.model.train()
                eval_loss = eval_loss / batch_num
                result = {'eval_loss': round(eval_loss, 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    print("  %s = %s", key, str(result[key]))
                print("  " + "*" * 20)
                if do_eval_bleu:
                    if self.sacrebleu_metric is None:
                        self.sacrebleu_metric = evaluate.load("sacrebleu")
                    
                    hyp_list = []
                    datas = pd.read_csv(eval_filename)
                    ref_list_for_bleu = [[r] for r in datas['tgt'].tolist()]
                    src_list = datas['src'].tolist()

                    for i in tqdm(range(len(src_list)), desc="Generating predictions for BLEU"):
                        src = src_list[i]
                        hyp_list.append(self.predict(src))

                    assert len(ref_list_for_bleu) == len(hyp_list)

                    bleu_results = self.sacrebleu_metric.compute(predictions=hyp_list, references=ref_list_for_bleu)
                    current_bleu_score = bleu_results['score'] / 100.0
                    print(f"Epoch {cur_epoch} BLEU score: {current_bleu_score}")

                    if best_bleu < current_bleu_score:
                        best_bleu = current_bleu_score
                        print(f'New best BLEU score: {best_bleu}')
                        count = 0
                        output_dir_bleu = os.path.join(output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir_bleu):
                            os.makedirs(output_dir_bleu)
                        self.model.save_pretrained(output_dir_bleu)
                    else:
                        count += 1
                        if count == early_stop:
                            print(f"Early stopping triggered after {early_stop} epochs without BLEU improvement.")
                            break
            
                        if count == early_stop:
                            break

                print("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()

    def test(self, filename, output_dir, decoding='greedy'):
        if self.sacrebleu_metric is None:
            self.sacrebleu_metric = evaluate.load("sacrebleu")
        if self.rouge_metric is None:
            self.rouge_metric = evaluate.load("rouge")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        hyp_list = []
        datas = pd.read_csv(filename)
        ref_list_strings = datas['tgt'].tolist()
        src_list = datas['src'].tolist()

        for i in tqdm(range(len(src_list)), desc="Generating predictions for test set"):
            src = src_list[i]
            hyp_list.append(self.predict(src, decoding))

        assert len(ref_list_strings) == len(hyp_list)
        df_gold = pd.DataFrame(ref_list_strings)
        df_gold.to_csv(os.path.join(output_dir, "gold.csv"), index=False, header=None)
        df_hyp = pd.DataFrame(hyp_list)
        df_hyp.to_csv(os.path.join(output_dir, "predictions.csv"), index=False, header=None)

        ref_list_for_bleu_test = [[r] for r in ref_list_strings]
        bleu_results = self.sacrebleu_metric.compute(predictions=hyp_list, references=ref_list_for_bleu_test)
        
        rouge_results = self.rouge_metric.compute(predictions=hyp_list, references=ref_list_strings)

        score = {
            "bleu": bleu_results['score'],
            "counts": bleu_results['counts'],
            "totals": bleu_results['totals'],
            "precisions": bleu_results['precisions'],
            "bp": bleu_results['bp'],
            "sys_len": bleu_results['sys_len'],
            "ref_len": bleu_results['ref_len'],
            "rouge1": rouge_results['rouge1'], 
            "rouge2": rouge_results['rouge2'],
            "rougeL": rouge_results['rougeL'],
            "rougeLsum": rouge_results['rougeLsum']
        }
        print("Evaluation scores:")
        for key, value in score.items():
            print(f"  {key}: {value}")

    def predict(self, src, decoding='greedy'):
        messages = cot_prompt_pre(src)
        encoding_dict = self.tokenizer.apply_chat_template(messages, 
                                                           tokenize=True, 
                                                           add_generation_prompt=True, 
                                                           return_tensors="pt",
                                                           max_length=self.source_len,
                                                           truncation=True)
        if isinstance(encoding_dict, dict):
            input_ids = encoding_dict.get("input_ids").to(self.device)
        else:
            input_ids = encoding_dict.to(self.device)

        with torch.no_grad():
            if decoding == 'greedy':
                gen_tokens = self.model.generate(input_ids=input_ids,
                                                 do_sample=False,
                                                 num_beams=1,
                                                 temperature=0.2,
                                                 max_new_tokens=256,
                                                 num_return_sequences=1,
                                                 eos_token_id=self.tokenizer.eos_token_id,
                                                 top_p=0.95)
            elif decoding == 'beam':
                gen_tokens = self.model.generate(input_ids=input_ids,
                                                 do_sample=False,
                                                 num_beams=5,
                                                 temperature=0.2,
                                                 max_new_tokens=256,
                                                 num_return_sequences=1,
                                                 eos_token_id=self.tokenizer.eos_token_id,
                                                 top_p=0.95)
            elif decoding == 'multinomial':
                gen_tokens = self.model.generate(input_ids=input_ids,
                                        do_sample=True,
                                        num_beams=1,
                                        temperature=0.2,
                                        max_new_tokens=256,
                                        num_return_sequences=1,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        top_p=0.95)
            elif decoding == 'contrastive':
                gen_tokens = self.model.generate(input_ids=input_ids,
                                        penalty_alpha=0.6,
                                        top_k=4,
                                        temperature=0.2,
                                        max_new_tokens=256,
                                        num_return_sequences=1,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        top_p=0.95)

        if gen_tokens.shape[-1] > input_ids.shape[-1]:
             gen_tokens = gen_tokens[:, input_ids.shape[-1]:]
        else:
            pass

        gen_seqs = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        completion_seqs = []
        for gen_seq in gen_seqs:
            if self.tokenizer.eos_token in gen_seq:
                gen_seq = gen_seq[:gen_seq.index(self.tokenizer.eos_token)]
            completion_seqs.append(gen_seq)
        return completion_seqs[0]

