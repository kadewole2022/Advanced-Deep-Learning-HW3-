{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNgDpXxFz2zmLbbxR8W/CI8",
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
        "<a href=\"https://colab.research.google.com/github/kadewole2022/Advanced-Deep-Learning-HW3-/blob/main/translate.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 384
        },
        "id": "D535tuvRU18L",
        "outputId": "ed8859a5-d5b2-409f-aca8-2302cb04cb33"
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "ModuleNotFoundError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-9-fb7add726200>\u001b[0m in \u001b[0;36m<cell line: 2>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mpathlib\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mPath\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0mconfig\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mget_config\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlatest_weights_file_path\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mmodel\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mbuild_transformer\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mtokenizers\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mTokenizer\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mdatasets\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mload_dataset\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'config'",
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
        "from pathlib import Path\n",
        "from config import get_config, latest_weights_file_path\n",
        "from model import build_transformer\n",
        "from tokenizers import Tokenizer\n",
        "from datasets import load_dataset\n",
        "from dataset import BilingualDataset\n",
        "import torch\n",
        "import sys\n",
        "\n",
        "def translate(sentence: str):\n",
        "    # Define the device, tokenizers, and model\n",
        "    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "    print(\"Using device:\", device)\n",
        "    config = get_config()\n",
        "    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))\n",
        "    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))\n",
        "    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config[\"seq_len\"], config['seq_len'], d_model=config['d_model']).to(device)\n",
        "\n",
        "    # Load the pretrained weights\n",
        "    model_filename = latest_weights_file_path(config)\n",
        "    state = torch.load(model_filename)\n",
        "    model.load_state_dict(state['model_state_dict'])\n",
        "\n",
        "    # if the sentence is a number use it as an index to the test set\n",
        "    label = \"\"\n",
        "    if type(sentence) == int or sentence.isdigit():\n",
        "        id = int(sentence)\n",
        "        ds = load_dataset(f\"{config['datasource']}\", f\"{config['lang_src']}-{config['lang_tgt']}\", split='all')\n",
        "        ds = BilingualDataset(ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])\n",
        "        sentence = ds[id]['src_text']\n",
        "        label = ds[id][\"tgt_text\"]\n",
        "    seq_len = config['seq_len']\n",
        "\n",
        "    # translate the sentence\n",
        "    model.eval()\n",
        "    with torch.no_grad():\n",
        "        # Precompute the encoder output and reuse it for every generation step\n",
        "        source = tokenizer_src.encode(sentence)\n",
        "        source = torch.cat([\n",
        "            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),\n",
        "            torch.tensor(source.ids, dtype=torch.int64),\n",
        "            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),\n",
        "            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)\n",
        "        ], dim=0).to(device)\n",
        "        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)\n",
        "        encoder_output = model.encode(source, source_mask)\n",
        "\n",
        "        # Initialize the decoder input with the sos token\n",
        "        decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)\n",
        "\n",
        "        # Print the source sentence and target start prompt\n",
        "        if label != \"\": print(f\"{f'ID: ':>12}{id}\")\n",
        "        print(f\"{f'SOURCE: ':>12}{sentence}\")\n",
        "        if label != \"\": print(f\"{f'TARGET: ':>12}{label}\")\n",
        "        print(f\"{f'PREDICTED: ':>12}\", end='')\n",
        "\n",
        "        # Generate the translation word by word\n",
        "        while decoder_input.size(1) < seq_len:\n",
        "            # build mask for target and calculate output\n",
        "            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)\n",
        "            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)\n",
        "\n",
        "            # project next token\n",
        "            prob = model.project(out[:, -1])\n",
        "            _, next_word = torch.max(prob, dim=1)\n",
        "            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)\n",
        "\n",
        "            # print the translated word\n",
        "            print(f\"{tokenizer_tgt.decode([next_word.item()])}\", end=' ')\n",
        "\n",
        "            # break if we predict the end of sentence token\n",
        "            if next_word == tokenizer_tgt.token_to_id('[EOS]'):\n",
        "                break\n",
        "\n",
        "    # convert ids to tokens\n",
        "    return tokenizer_tgt.decode(decoder_input[0].tolist())\n",
        "\n",
        "#read sentence from argument\n",
        "translate(sys.argv[1] if len(sys.argv) > 1 else \"I am not a very good a student.\")"
      ]
    }
  ]
}