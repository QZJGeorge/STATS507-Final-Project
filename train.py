# Standard library imports
import os
from pathlib import Path
import warnings
import unicodedata

# External library imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchtext.datasets as datasets
import torchmetrics
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

# Huggingface datasets and tokenizers
from datasets import load_dataset, load_from_disk, concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace, Split

# External libraries
from bert_score import score

# Custom modules
from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path
from filter import clean_chinese_text, clean_english_text


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    batch_size = source.size(0)

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token for all batch elements
    decoder_input = torch.full(
        (batch_size, 1), sos_idx, dtype=source.dtype, device=device
    )

    # Keep track of which sequences have finished
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len):
        # Build mask for target
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # Calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get next token probabilities
        prob = model.project(out[:, -1])  # Shape: (batch_size, vocab_size)
        _, next_word = torch.max(prob, dim=1)  # Shape: (batch_size,)

        # For sequences that have finished, set next_word to eos_idx
        next_word = next_word.masked_fill(finished, eos_idx)

        # Append next_word to decoder_input
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(1)], dim=1)

        # Update 'finished' status
        finished |= next_word.eq(eos_idx)

        # If all sequences are finished, break
        if finished.all():
            break

    return decoder_input  # Shape: (batch_size, sequence_length)


def run_validation(
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_step,
    writer,
    num_examples=2,
):
    model.eval()

    source_texts = []
    expected = []
    predicted = []

    count = 0
    with torch.no_grad():
        for batch in validation_ds:
            source_texts_batch = []
            expected_batch = []
            predicted_batch = []
            count += 1
            print(f"Batch {count}/{len(validation_ds)}")

            encoder_input = batch["encoder_input"].to(
                device
            )  # Shape: (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(
                device
            )  # Shape: (batch_size, 1, 1, seq_len)

            batch_size = encoder_input.size(0)

            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )

            # Iterate over each sequence in the batch
            for i in range(batch_size):
                source_text = batch["src_text"][i]

                target_text = batch["tgt_text"][i]
                target_text = target_text.replace(" ", "")

                output_tokens = model_out[i].detach().cpu().numpy()

                model_out_text = tokenizer_tgt.decode(output_tokens)
                model_out_text = model_out_text.replace(" ", "")

                target_text = target_text.replace(" ", "")

                source_texts.append(source_text)
                expected.append(target_text)
                predicted.append(model_out_text)

    if writer:
        P, R, F1 = score(predicted, expected, lang="zh")

        # Calculate average values
        avg_P = sum(P) / len(P)
        avg_R = sum(R) / len(R)
        avg_F1 = sum(F1) / len(F1)

        print_msg(f"PRECISION: {avg_P}, RECALL: {avg_R}, F1 Score: {avg_F1}")


def get_all_sentences(ds, lang):
    for item in ds:
        text = item["translation"][lang]

        if lang == "en":
            yield clean_english_text(text)
        elif lang == "zh":
            yield clean_chinese_text(text)


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        if lang == "en":
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(
                special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=10
            )
            tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
            tokenizer.save(str(tokenizer_path))
            print("English tokenizer saving complete")

        elif lang == "zh":
            # Using character-level tokenization
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()

            trainer = WordLevelTrainer(
                special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=10
            )
            tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
            tokenizer.save(str(tokenizer_path))
            print("Chinese tokenizer saving complete")

        else:
            raise Exception("language out of range")

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"loading tokenizer from {str(tokenizer_path)}")

    return tokenizer


def get_ds(config):
    # Define the dataset path
    dataset_path = "./dataset"

    ds_train = load_dataset(
        f"{config['datasource']}",
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="train",
    )

    ds_val = load_dataset(
        f"{config['datasource']}",
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="validation",
    )

    ds_test = load_dataset(
        f"{config['datasource']}",
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="test",
    )

    # Merge the datasets
    ds_combined = concatenate_datasets([ds_val, ds_test])

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_train, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_train, config["lang_tgt"])

    train_ds = BilingualDataset(
        ds_train,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        ds_combined,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=16, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
    )
    return model


def train_model(config):
    # Define the device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    )
    print("Using device:", device)
    if device == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(
            f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB"
        )
    elif device == "mps":
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print(
            "      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc"
        )
        print(
            "      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu"
        )
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(
        parents=True, exist_ok=True
    )

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)
    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config["preload"]
    model_filename = (
        latest_weights_file_path(config)
        if preload == "latest"
        else get_weights_file_path(config, preload) if preload else None
    )
    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("No model to preload, starting from scratch")

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (B, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device)  # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (B, seq_len, d_model)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch["label"].to(device)  # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )

        run_validation(
            model,
            val_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            config["seq_len"],
            device,
            lambda msg: batch_iterator.write(msg),
            global_step,
            writer,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
