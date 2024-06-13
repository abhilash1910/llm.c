# llm.sycl

This is an official SYCL fork for [llm.c](https://github.com/karpathy/llm.c) including support for standard 
SYCL runtime across hardwares. The repository utilizes [oneapi toolchain](https://www.oneapi.io/) , [SYCL standards](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html) , [oneDNN](https://oneapi-src.github.io/oneDNN/). Currently supports BF16/FP32 mixed precision on single GPU runtime for train_gpt2.cpp 

## Context of llm.c and LLMs


LLMs in simple, pure C/CUDA with no need for 245MB of PyTorch or 107MB of cPython. Current focus is on pretraining, in particular reproducing the [GPT-2](https://github.com/openai/gpt-2) and [GPT-3](https://arxiv.org/abs/2005.14165) miniseries, along with a parallel PyTorch reference implementation in [train_gpt2.py](train_gpt2.py). You'll recognize this file as a slightly tweaked [nanoGPT](https://github.com/karpathy/nanoGPT), an earlier project of mine. Currently, llm.c is a bit faster than PyTorch Nightly (by about 7%). In addition to the bleeding edge mainline code in [train_gpt2.cu](train_gpt2.cu), we have a simple reference CPU fp32 implementation in ~1,000 lines of clean code in one file [train_gpt2.c](train_gpt2.c). I'd like this repo to only maintain C and CUDA code. Ports to other languages or repos are very welcome, but should be done in separate repos, and I am happy to link to them below in the "notable forks" section. Developer coordination happens in the [Discussions](https://github.com/karpathy/llm.c/discussions) and on Discord, either the `#llmc` channel on the [Zero to Hero](https://discord.gg/3zy8kqD9Cp) channel, or on `#llmdotc` on CUDA MODE Discord.

## Run individual benchmarks on ops

The repository has a [sycl](https://github.com/abhilash1910/llm.sycl/tree/sycl/sycl) folder which contains the 
corresponding kernels and algorithms for llm.sycl.To validate and benchmark kernels across hardwares please use the following points using [MakeFile](https://github.com/abhilash1910/llm.sycl/blob/sycl/sycl/Makefile):

### For Intel GPUs on SYCL

```bash
git clone https://github.com/abhilash1910/llm.sycl.git
cd llm.c/sycl
make llmc/attention
make llmc/adamw
make llmc/gelu
make llmc/dnn_att
make llmc/global_norm
make llmc/layernorm
make llmc/encoder
make llmc/fused_classifier
```
Run them using:

```bash
./llmc/attention.o
```

This will by default compile for spir64_gen runtime, more compiler customizations for spv will be followed.
The corresponding header files for these cpp files contain same code which are used in llm.sycl frontend.
We can manually target SPIRV like so:

```bash
export SPIRV=yes
```

### Run for Nvidia sm architectures

Set the environment variables to target for nvidia runtime like so:

```bash
export CUDA=yes
export CUDA_ARCH=sm_90 // if you would like to change architecture default is sm_70
```

Also for running SYCL code on Nvidia, please install the Nvidia plugin for SYCL from [Codeplay](https://developer.codeplay.com/products/oneapi/nvidia/2023.0.0/guides/get-started-guide-nvidia)

The rest of the commands remain the same for building and running the kernels.


### Run for AMD gcn architectures


Disclaimer: This is currently in experimental phase.


## quick start (1 GPU, fp16/bf16 mix only)

Run the 1 GPU, fp32 code like this:

```bash
pip install -r requirements.txt
python dev/data/tinyshakespeare.py
python train_gpt2.py
make train_gpt2
./train_gpt2.o
```

The above lines (1) download the [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset, tokenize it with the GPT-2 Tokenizer, (2) download and save the GPT-2 (124M) weights, (3) init from them in SYCL and train for one epoch on tineshakespeare with AdamW (using batch size 4, context length 1024, total of 74 steps), evaluate validation loss, and sample some text. Steps for building for Nvidia SYCL runtime is same as in previous section.

## quick start (CPU) on SYCL

The "I am so GPU poor that I don't even have one GPU" section. You can still enjoy seeing llm.sycl train! In this case a small edit needs to be made for onednn device selector for cpu runtime in [train_gpt2.cpp](https://github.com/abhilash1910/llm.sycl/blob/d9944327c02880db8d1934a7c064d628d4f4cd01/sycl/train_gpt2.cpp#L1521). (PS:this will be modified soon for easer usability). For example, instead of training from scratch, you can finetune a GPT-2 small (124M) to output Shakespeare-like text, as an example on CPU:

```bash
pip install -r requirements.txt
python dev/data/tinyshakespeare.py
python train_gpt2.py
make train_gpt2
OMP_NUM_THREADS=8 ./train_gpt2.o
```

## datasets

The data files inside `/dev/data/(dataset).py` are responsible for downloading, tokenizing and saving the tokens to .bin files, readable easily from C. So for example when you run:

```bash
python dev/data/tinyshakespeare.py
```

We download and tokenize the [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset. The output of this looks like this:

```
writing 32,768 tokens to ./dev/data/tinyshakespeare/tiny_shakespeare_val.bin
writing 305,260 tokens to ./dev/data/tinyshakespeare/tiny_shakespeare_train.bin
```

The .bin files contain a short header (1024 bytes) and then a stream of tokens in uint16, indicating the token ids with the GPT-2 tokenizer. More datasets are available in `/dev/data`.

## test

I am also attaching a simple unit test for making sure our C code agrees with the PyTorch code. On the  SYCL CPU/GPU as an example, compile and run with:

```bash
make test_gpt2
./test_gpt2.o
```

This now loads the `gpt2_124M_debug_state.bin` file that gets written by train_gpt2.py, runs a forward pass, compares the logits and loss with the PyTorch reference implementation, then it does 10 iterations of training with Adam and makes sure the losses match PyTorch. 

## further work

This will be continued to add more features. 


## license

MIT
