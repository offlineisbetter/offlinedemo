# _offlineisbetter_

inference for models by _offlineisbetter_.

we believe that you shouldn't give your data to faceless companies, that you deserve to run text models locally, and that you shouldn't need to buy expensive hardware. so we're building _offlineisbetter_.

_offlineisbetter_ models will be for _encoding_ tasks: sentiment analysis, text tagging, document retrieval, etc., rather than for _decoding_ tasks like autoregressive generation. we believe that it's wasteful and dangerous to depend on cloud apis for frontier language models to do these simple tasks, and it should be almost mindless to download a model to _use_ it without dealing with runtimes or quantization formats.

## try it yourself

try our first model yourself. our first model is a small (230m) text model for sentiment analysis called `offline-sentiment-small`.

```bash
pip install offlinedemo
```

download the model archive from the website and unpack the model.

```bash
tar -xvf offline-sentiment-small.tar
```

run the demo, passing the inflated directory containing the model checkpoint.

```bash
offlinedemo offline-sentiment-small
```

## benchmarks

below are benchmarks for `offline-sentiment-small` on binary sentiment classification using the [stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2) validation set. all benchmarks were completed on the ryzen 9950x3d cpu on one thread.

| model | parameters | p95 (ms) | f1 (validation) |
|:---:|:---:|:---:|:---:|
| `offline-sentiment-small` | 230m | 80.32 | 0.9489 |
| `distilbert-base` | 67m | 66.15 | 0.9321 |
| `roberta-base` | 125m | 469.65 | 0.9396 |
| `modernbert-base` | 149m | 530.73 | 0.9396 |

## philosophy

succinctly, the core philosophy of _offlineisbetter_ is that parameter-efficient and low-latency models should be easily accessible to everybody. of course hugging face and `transformers.pipeline` allows you to run sentiment analysis in three lines of python, but for more parameter-efficient models, already quantized and with optimized computation graphs.
