# security

## scope

vramfit reads model metadata and runs benchmark binaries you point it
at. The parts worth reporting on:

- **The GGUF and safetensors header parsers.** These read untrusted
  files -- a model downloaded from a public index is untrusted input.
  A crafted header that causes an unbounded allocation, a huge read, or
  anything worse than a clean exception is a bug I want to hear about.
  The safetensors reader already rejects an implausible header length
  for this reason and the GGUF reader does not materialise the tokenizer
  arrays.
- **Subprocess handling in the bench backends.** They run a binary you
  name with arguments built from your flags. Argument injection through
  a model path, or anything that escapes the argv list into a shell,
  would be a bug; nothing here uses `shell=True`.
- **The ledger.** sqlite, parameterised queries only. A contributed
  ledger entry is untrusted data and is validated against
  `vramfit/schema/ledger-entry.schema.json` before it goes anywhere
  near the database.

Explicitly out of scope: the model file's *contents*. vramfit does not
load weights, deserialise pickles or execute anything from a checkpoint.
If that is the question, it is a different tool's problem.

## reporting

Open a draft security advisory on the repository, or mail
panic@c01n31.net. I will acknowledge within a week. There is no bounty.

## supported versions

The latest minor release gets fixes. Older ones do not.
