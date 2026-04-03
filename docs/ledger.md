# the ledger

Every measurement gets kept. The reason is that every answer to "how
fast is X on Y" is a number somebody measured once and then remembered
wrongly, including me.

    vramfit bench ./model.gguf --context 8192 --record
    vramfit ledger list
    vramfit ledger show --model Meta-Llama-3.1-8B-Instruct
    vramfit ledger diff --model Meta-Llama-3.1-8B-Instruct

## what the key is

`(model, quant, context, backend, gpu)`. Everything else -- driver
version, backend build, cache dtype, batch size -- is provenance, and a
new measurement under the same key is *appended* rather than
overwriting, because overwriting is how you lose the ability to answer
"did that upgrade help".

The `gpu` part is the device tag, which includes the tail of the NVML
UUID. A box with two identical cards once collided on one key and the
diff reported a 30% regression that was really the other slot.

## the diff

`ledger diff` reports measured changes above 2% -- below that is
run-to-run noise here -- and prints the provenance that moved at the
same time *next to* them:

    T1 -> T2

      decode_tok_s             118.40 ->     131.00  (+10.6%)
      achieved_bandwidth_gbs   758.00 ->     838.00  (+10.6%)

      changed at the same time:
        driver                   550.54.14 -> 550.78
        -- attribute the change to these before attributing it to
           anything else

That last line is the whole feature. A speed change reported without
the driver change beside it is an invitation to the wrong conclusion.

## export

`ledger` renders markdown or CSV of the latest measurement per key.
Five runs of the same configuration are interesting to me and noise to
everybody else.
