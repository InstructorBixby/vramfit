# how this breaks

A note to the version of me that comes back and adds a fast path.

## the parts that read untrusted input

- **The GGUF and safetensors header parsers.** A model pulled off a
  public index is untrusted input, and this tool's whole pitch is that
  it will tell me about a file before I commit five gigabytes and a
  loader to it. A crafted header that causes an unbounded allocation, a
  huge read, or anything worse than a clean exception is the most
  serious bug class in here. The safetensors reader already rejects an
  implausible header length for that reason, and the GGUF reader does
  not materialise the tokenizer arrays.
- **Subprocess handling in the bench backends.** They run a binary I
  name with arguments built from flags. Argument injection through a
  model path, or anything that escapes the argv list into a shell,
  would be a bug. Nothing here uses `shell=True`, and nothing here
  should start.
- **The ledger.** sqlite, parameterised queries only. An entry imported
  from a JSON file is untrusted data even when I wrote the file, and it
  is validated against `vramfit/schema/ledger-entry.schema.json` before
  it goes anywhere near the database.

## explicitly not this tool's problem

The model file's *contents*. vramfit does not load weights,
deserialise pickles or execute anything from a checkpoint. If that is
the question, it is ckptscan's.

## the quiet failure

Not a crash: a number that is wrong and plausible. Every guard in here
that refuses to guess -- the unknown ggml type id that is fatal, the
unknown card that gets no fraction, the partial offload that reports
tok/s and no ratio -- exists because the alternative is a confident
figure that ends up in a report with my name on it.
