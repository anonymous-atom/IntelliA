import os
import torch
import pandas as pd
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, StateDictType, FullStateDictConfig, StateDictConfig
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from tqdm import tqdm
from torch.amp import autocast
import argparse
from transformers import AutoConfig
import os
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp import StateDictType
from torch.distributed.checkpoint.stateful import Stateful
from multi_obj_rm import ChatMultiObjClassifier
from multi_obj_data import MultiObjRMDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=20000)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--min_lr_rate", type=float, default=0.1)
    parser.add_argument("--precomputed_dir", type=str, default=None)
    return parser.parse_args()



options = StateDictOptions(full_state_dict=True, cpu_offload=True)
import time
import torch
import os
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, get_optimizer_state_dict,
    set_model_state_dict, set_optimizer_state_dict,
    StateDictOptions,
)

MODEL_CHECKPOINT = "model_state_dict.pt"
OPTIM_CHECKPOINT = "optim_state_dict.pt"

metrics_history = []
def save_checkpoint(model, optimizer, base_dir: str, step: int):
    """Save FSDP checkpoint with proper synchronization"""
    assert torch.distributed.is_initialized()
    
    rank = dist.get_rank()
    dist.barrier()  # Synchronize

    try:
        if rank == 0:
            os.makedirs(base_dir, exist_ok=True)
            folder = os.path.join(base_dir, f"checkpoint-step{step}")
            os.makedirs(folder, exist_ok=True)

            # Get state dicts
            model_state = get_model_state_dict(
                model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
            )
            optim_state = get_optimizer_state_dict(
                model, optimizers=optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
            )

            # Save to disk
            torch.save(model_state, os.path.join(folder, f"model_state_dict-step{step}.pt"))
            torch.save(optim_state, os.path.join(folder, f"optim_state_dict-step{step}.pt"))

        else:
            get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
            get_optimizer_state_dict(model, optimizers=optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        
        dist.barrier()
        
    except Exception as e:
        print(f"Rank {rank}: Error during checkpoint saving: {e}")
        dist.barrier()
        raise e


def setup_fsdp_env():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def evaluate(model, dataloader, device):
    """Evaluation with proper FSDP synchronization"""
    # Synchronize all processes before evaluation
    dist.barrier()
    
    model.eval()
    correct = [0, 0, 0]
    total = 0

    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False, disable=dist.get_rank() != 0):
                reps = batch["rep"].cuda()
                labels = batch["labels"].cuda()
                
                # print(f"reps.shape: {reps.shape}")
                # print(f"labels.shape: {labels.shape}")
                with autocast(dtype=torch.bfloat16, device_type='cuda'):
                    outputs = model(rep=reps, labels=labels)
                    
                preds = torch.argmax(outputs["logits"], dim=-1)
                correct_batch = (preds == labels).sum(dim=0).tolist()
                for i in range(3):
                    correct[i] += correct_batch[i]
                total += labels.size(0)

        acc = [round(c / total, 4) for c in correct]
        result = {
            "low_effort_acc": acc[0],
            "evidence_acc": acc[1],
            "factual_acc": acc[2]
        }
    except Exception as e:
        print(f"Rank {dist.get_rank()}: Error during evaluation: {e}")
        result = {
            "low_effort_acc": 0.0,
            "evidence_acc": 0.0,
            "factual_acc": 0.0
        }
    finally:
        # Restore train mode and synchronize
        model.train()
        dist.barrier()
    
    return result

def train():
    args = parse_args()
    local_rank = setup_fsdp_env()
    is_main_process = (dist.get_rank() == 0)

    os.makedirs(args.output_dir, exist_ok=True) if is_main_process else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    train_dataset = MultiObjRMDataset(
        tokenizer,
        arrow_dir="train_data",
        split="train",
        precomputed_col="hidden_state_file_last_token",
        base_hidden_dir="hidden_states"
    )
    
    val_dataset = MultiObjRMDataset(
        tokenizer,
        arrow_dir="test_data",
        split="test",
        precomputed_col="hidden_state_file_last_token",
        base_hidden_dir="hidden_states"
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model and FSDP-wrap it
    model = ChatMultiObjClassifier(base_model_name=args.model_name)
    model = fully_shard(model).cuda()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    num_training_steps = len(train_loader) * args.epochs // args.accum_steps

    if args.lr_scheduler_type == "cosine_with_min_lr":
        def lr_lambda(current_step):
            progress = current_step / float(num_training_steps)
            cosine_decay = 0.5 * (1 + torch.cos(torch.pi * progress))
            return args.min_lr_rate + (1.0 - args.min_lr_rate) * cosine_decay
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=num_training_steps,
        )

    step = 0
    model.train()

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}") if is_main_process else train_loader
        total_loss = 0.0

        for i, batch in enumerate(pbar):
            if args.precomputed_dir:
                reps = batch["rep"].cuda()
                labels = batch["labels"].cuda()
                # print(f"reps.shape: {reps.shape}")
                # print(f"labels.shape: {labels.shape}")
                with autocast(dtype=torch.bfloat16, device_type='cuda'):
                    output = model(rep=reps, labels=labels)
                    loss = output["loss"] / args.accum_steps
            else:
                input_ids = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()
                labels = batch["labels"].cuda()
                with autocast(dtype=torch.bfloat16, device_type='cuda'):
                    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = output["loss"] / args.accum_steps

            # with autocast(dtype=torch.bfloat16, device_type='cuda'):
            #     output = model(input_ids, attention_mask, labels)
            #     loss = output["loss"] / args.accum_steps

            loss.backward()
            total_loss += loss.item()

            if (i + 1) % args.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                step += 1

                if is_main_process:
                    avg_loss = round(total_loss / (i + 1), 4)
                    pbar.set_postfix({"step": step, "loss": avg_loss})
                    tqdm.write(f"[Step {step}] Loss: {avg_loss}")

                should_eval = step % args.eval_steps == 0
                should_save = step % args.save_steps == 0
                
                # Evaluation (all processes participate)
                if should_eval:
                    if is_main_process:
                        print(f"Running evaluation at step {step}...")
                    train_metrics = evaluate(model, train_loader, device=torch.device("cuda"))
                    val_metrics = evaluate(model, val_loader, device=torch.device("cuda"))

                    if is_main_process:
                        print(
                            f"[Step {step}] Train Accuracies — Low Effort: {train_metrics['low_effort_acc']:.4f}, "
                            f"Evidence: {train_metrics['evidence_acc']:.4f}, Factual: {train_metrics['factual_acc']:.4f}"
                            )
                        print(
                            f"[Step {step}] Val Accuracies   — Low Effort: {val_metrics['low_effort_acc']:.4f}, "
                            f"Evidence: {val_metrics['evidence_acc']:.4f}, Factual: {val_metrics['factual_acc']:.4f}"
                            )
                    if is_main_process:
                        metrics_history.append({
                            "step": step,
                            "train_low_effort_acc": train_metrics["low_effort_acc"],
                            "train_evidence_acc": train_metrics["evidence_acc"],
                            "train_factual_acc": train_metrics["factual_acc"],
                            "val_low_effort_acc": val_metrics["low_effort_acc"],
                            "val_evidence_acc": val_metrics["evidence_acc"],
                            "val_factual_acc": val_metrics["factual_acc"],
                            "loss": avg_loss
                        })
                    
                        df = pd.DataFrame(metrics_history)
                        df.to_csv(os.path.join(args.output_dir, "training_metrics.csv"), index=False)
                    

                if step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-step{step}")
                    save_checkpoint(model, optimizer, args.output_dir, step)
                    if is_main_process:
                            print(f"Checkpoint saved at step {step}")


        if is_main_process:
            print(f"Epoch {epoch+1} completed. Avg Loss: {round(total_loss / len(train_loader), 4)}")
            train_metrics = evaluate(model, train_loader, device=torch.device("cuda"))
            val_metrics = evaluate(model, val_loader, device=torch.device("cuda"))
            print(
                f"[Epoch {epoch+1}] Final Train Accuracies — Low Effort: {train_metrics['low_effort_acc']:.4f}, "
                f"Evidence: {train_metrics['evidence_acc']:.4f}, Factual: {train_metrics['factual_acc']:.4f}"
            )
            print(
                f"[Epoch {epoch+1}] Final Val Accuracies   — Low Effort: {val_metrics['low_effort_acc']:.4f}, "
                f"Evidence: {val_metrics['evidence_acc']:.4f}, Factual: {val_metrics['factual_acc']:.4f}"
            )


if __name__ == "__main__":
    train()