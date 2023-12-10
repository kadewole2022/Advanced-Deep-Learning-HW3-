{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNzX+u7jt2HozCG4f/HyL1b",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/kadewole2022/Advanced-Deep-Learning-HW3-/blob/main/train.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 384
        },
        "id": "RPobHMyvXWh5",
        "outputId": "08e35424-91ca-4f1b-f17d-b3e4d8bcc41f"
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "ModuleNotFoundError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-1-9b001ca79267>\u001b[0m in \u001b[0;36m<cell line: 1>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0mmodel\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mbuild_transformer\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mdataset\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mBilingualDataset\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcausal_mask\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mconfig\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mget_config\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mget_weights_file_path\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlatest_weights_file_path\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mtorchtext\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdatasets\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mdatasets\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'model'",
            "",
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0;32m\nNOTE: If your import is failing due to a missing package, you can\nmanually install dependencies using either !pip or !apt.\n\nTo view examples of installing some common dependencies, click the\n\"Open Examples\" button below.\n\u001b[0;31m---------------------------------------------------------------------------\u001b[0m\n"
          ],
          "errorDetails": {
            "actions": [
              {
                "action": "open_url",
                "actionText": "Open Examples",
                "url": "/notebooks/snippets/importing_libraries.ipynb"
              }
            ]
          }
        }
      ],
      "source": [
        "from model import build_transformer\n",
        "from dataset import BilingualDataset, causal_mask\n",
        "from config import get_config, get_weights_file_path, latest_weights_file_path\n",
        "\n",
        "import torchtext.datasets as datasets\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "from torch.utils.data import Dataset, DataLoader, random_split\n",
        "from torch.optim.lr_scheduler import LambdaLR\n",
        "\n",
        "import warnings\n",
        "from tqdm import tqdm\n",
        "import os\n",
        "from pathlib import Path\n",
        "\n",
        "# Huggingface datasets and tokenizers\n",
        "from datasets import load_dataset\n",
        "from tokenizers import Tokenizer\n",
        "from tokenizers.models import WordLevel\n",
        "from tokenizers.trainers import WordLevelTrainer\n",
        "from tokenizers.pre_tokenizers import Whitespace\n",
        "\n",
        "import torchmetrics\n",
        "from torch.utils.tensorboard import SummaryWriter\n",
        "\n",
        "def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):\n",
        "    sos_idx = tokenizer_tgt.token_to_id('[SOS]')\n",
        "    eos_idx = tokenizer_tgt.token_to_id('[EOS]')\n",
        "\n",
        "    # Precompute the encoder output and reuse it for every step\n",
        "    encoder_output = model.encode(source, source_mask)\n",
        "    # Initialize the decoder input with the sos token\n",
        "    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)\n",
        "    while True:\n",
        "        if decoder_input.size(1) == max_len:\n",
        "            break\n",
        "\n",
        "        # build mask for target\n",
        "        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)\n",
        "\n",
        "        # calculate output\n",
        "        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)\n",
        "\n",
        "        # get next token\n",
        "        prob = model.project(out[:, -1])\n",
        "        _, next_word = torch.max(prob, dim=1)\n",
        "        decoder_input = torch.cat(\n",
        "            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1\n",
        "        )\n",
        "\n",
        "        if next_word == eos_idx:\n",
        "            break\n",
        "\n",
        "    return decoder_input.squeeze(0)\n",
        "\n",
        "\n",
        "def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):\n",
        "    model.eval()\n",
        "    count = 0\n",
        "\n",
        "    source_texts = []\n",
        "    expected = []\n",
        "    predicted = []\n",
        "\n",
        "    try:\n",
        "        # get the console window width\n",
        "        with os.popen('stty size', 'r') as console:\n",
        "            _, console_width = console.read().split()\n",
        "            console_width = int(console_width)\n",
        "    except:\n",
        "        # If we can't get the console width, use 80 as default\n",
        "        console_width = 80\n",
        "\n",
        "    with torch.no_grad():\n",
        "        for batch in validation_ds:\n",
        "            count += 1\n",
        "            encoder_input = batch[\"encoder_input\"].to(device) # (b, seq_len)\n",
        "            encoder_mask = batch[\"encoder_mask\"].to(device) # (b, 1, 1, seq_len)\n",
        "\n",
        "            # check that the batch size is 1\n",
        "            assert encoder_input.size(\n",
        "                0) == 1, \"Batch size must be 1 for validation\"\n",
        "\n",
        "            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)\n",
        "\n",
        "            source_text = batch[\"src_text\"][0]\n",
        "            target_text = batch[\"tgt_text\"][0]\n",
        "            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())\n",
        "\n",
        "            source_texts.append(source_text)\n",
        "            expected.append(target_text)\n",
        "            predicted.append(model_out_text)\n",
        "\n",
        "            # Print the source, target and model output\n",
        "            print_msg('-'*console_width)\n",
        "            print_msg(f\"{f'SOURCE: ':>12}{source_text}\")\n",
        "            print_msg(f\"{f'TARGET: ':>12}{target_text}\")\n",
        "            print_msg(f\"{f'PREDICTED: ':>12}{model_out_text}\")\n",
        "\n",
        "            if count == num_examples:\n",
        "                print_msg('-'*console_width)\n",
        "                break\n",
        "\n",
        "    if writer:\n",
        "        # Evaluate the character error rate\n",
        "        # Compute the char error rate\n",
        "        metric = torchmetrics.CharErrorRate()\n",
        "        cer = metric(predicted, expected)\n",
        "        writer.add_scalar('validation cer', cer, global_step)\n",
        "        writer.flush()\n",
        "\n",
        "        # Compute the word error rate\n",
        "        metric = torchmetrics.WordErrorRate()\n",
        "        wer = metric(predicted, expected)\n",
        "        writer.add_scalar('validation wer', wer, global_step)\n",
        "        writer.flush()\n",
        "\n",
        "        # Compute the BLEU metric\n",
        "        metric = torchmetrics.BLEUScore()\n",
        "        bleu = metric(predicted, expected)\n",
        "        writer.add_scalar('validation BLEU', bleu, global_step)\n",
        "        writer.flush()\n",
        "\n",
        "def get_all_sentences(ds, lang):\n",
        "    for item in ds:\n",
        "        yield item['translation'][lang]\n",
        "\n",
        "def get_or_build_tokenizer(config, ds, lang):\n",
        "    tokenizer_path = Path(config['tokenizer_file'].format(lang))\n",
        "    if not Path.exists(tokenizer_path):\n",
        "        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour\n",
        "        tokenizer = Tokenizer(WordLevel(unk_token=\"[UNK]\"))\n",
        "        tokenizer.pre_tokenizer = Whitespace()\n",
        "        trainer = WordLevelTrainer(special_tokens=[\"[UNK]\", \"[PAD]\", \"[SOS]\", \"[EOS]\"], min_frequency=2)\n",
        "        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)\n",
        "        tokenizer.save(str(tokenizer_path))\n",
        "    else:\n",
        "        tokenizer = Tokenizer.from_file(str(tokenizer_path))\n",
        "    return tokenizer\n",
        "\n",
        "def get_ds(config):\n",
        "    # It only has the train split, so we divide it overselves\n",
        "    ds_raw = load_dataset(f\"{config['datasource']}\", f\"{config['lang_src']}-{config['lang_tgt']}\", split='train')\n",
        "\n",
        "    # Build tokenizers\n",
        "    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])\n",
        "    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])\n",
        "\n",
        "    # Keep 90% for training, 10% for validation\n",
        "    train_ds_size = int(0.9 * len(ds_raw))\n",
        "    val_ds_size = len(ds_raw) - train_ds_size\n",
        "    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])\n",
        "\n",
        "    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])\n",
        "    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])\n",
        "\n",
        "    # Find the maximum length of each sentence in the source and target sentence\n",
        "    max_len_src = 0\n",
        "    max_len_tgt = 0\n",
        "\n",
        "    for item in ds_raw:\n",
        "        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids\n",
        "        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids\n",
        "        max_len_src = max(max_len_src, len(src_ids))\n",
        "        max_len_tgt = max(max_len_tgt, len(tgt_ids))\n",
        "\n",
        "    print(f'Max length of source sentence: {max_len_src}')\n",
        "    print(f'Max length of target sentence: {max_len_tgt}')\n",
        "\n",
        "\n",
        "    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)\n",
        "    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)\n",
        "\n",
        "    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt\n",
        "\n",
        "def get_model(config, vocab_src_len, vocab_tgt_len):\n",
        "    model = build_transformer(vocab_src_len, vocab_tgt_len, config[\"seq_len\"], config['seq_len'], d_model=config['d_model'])\n",
        "    return model\n",
        "\n",
        "def train_model(config):\n",
        "    # Define the device\n",
        "    device = \"cuda\" if torch.cuda.is_available() else \"mps\" if torch.has_mps or torch.backends.mps.is_available() else \"cpu\"\n",
        "    print(\"Using device:\", device)\n",
        "    if (device == 'cuda'):\n",
        "        print(f\"Device name: {torch.cuda.get_device_name(device.index)}\")\n",
        "        print(f\"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB\")\n",
        "    elif (device == 'mps'):\n",
        "        print(f\"Device name: <mps>\")\n",
        "    else:\n",
        "        print(\"NOTE: If you have a GPU, consider using it for training.\")\n",
        "        print(\"      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc\")\n",
        "        print(\"      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu\")\n",
        "    device = torch.device(device)\n",
        "\n",
        "    # Make sure the weights folder exists\n",
        "    Path(f\"{config['datasource']}_{config['model_folder']}\").mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)\n",
        "    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)\n",
        "    # Tensorboard\n",
        "    writer = SummaryWriter(config['experiment_name'])\n",
        "\n",
        "    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)\n",
        "\n",
        "    # If the user specified a model to preload before training, load it\n",
        "    initial_epoch = 0\n",
        "    global_step = 0\n",
        "    preload = config['preload']\n",
        "    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None\n",
        "    if model_filename:\n",
        "        print(f'Preloading model {model_filename}')\n",
        "        state = torch.load(model_filename)\n",
        "        model.load_state_dict(state['model_state_dict'])\n",
        "        initial_epoch = state['epoch'] + 1\n",
        "        optimizer.load_state_dict(state['optimizer_state_dict'])\n",
        "        global_step = state['global_step']\n",
        "    else:\n",
        "        print('No model to preload, starting from scratch')\n",
        "\n",
        "    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)\n",
        "\n",
        "    for epoch in range(initial_epoch, config['num_epochs']):\n",
        "        torch.cuda.empty_cache()\n",
        "        model.train()\n",
        "        batch_iterator = tqdm(train_dataloader, desc=f\"Processing Epoch {epoch:02d}\")\n",
        "        for batch in batch_iterator:\n",
        "\n",
        "            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)\n",
        "            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)\n",
        "            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)\n",
        "            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)\n",
        "\n",
        "            # Run the tensors through the encoder, decoder and the projection layer\n",
        "            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)\n",
        "            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)\n",
        "            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)\n",
        "\n",
        "            # Compare the output with the label\n",
        "            label = batch['label'].to(device) # (B, seq_len)\n",
        "\n",
        "            # Compute the loss using a simple cross entropy\n",
        "            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))\n",
        "            batch_iterator.set_postfix({\"loss\": f\"{loss.item():6.3f}\"})\n",
        "\n",
        "            # Log the loss\n",
        "            writer.add_scalar('train loss', loss.item(), global_step)\n",
        "            writer.flush()\n",
        "\n",
        "            # Backpropagate the loss\n",
        "            loss.backward()\n",
        "\n",
        "            # Update the weights\n",
        "            optimizer.step()\n",
        "            optimizer.zero_grad(set_to_none=True)\n",
        "\n",
        "            global_step += 1\n",
        "\n",
        "        # Run validation at the end of every epoch\n",
        "        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)\n",
        "\n",
        "        # Save the model at the end of every epoch\n",
        "        model_filename = get_weights_file_path(config, f\"{epoch:02d}\")\n",
        "        torch.save({\n",
        "            'epoch': epoch,\n",
        "            'model_state_dict': model.state_dict(),\n",
        "            'optimizer_state_dict': optimizer.state_dict(),\n",
        "            'global_step': global_step\n",
        "        }, model_filename)\n",
        "\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    warnings.filterwarnings(\"ignore\")\n",
        "    config = get_config()\n",
        "    train_model(config)"
      ]
    }
  ]
}