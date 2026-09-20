# Distilling judgement from a stronger model

> **Status.** The run below was kept as evidence and not released: the student learned its
> teacher without collapsing, and still did not judge the benchmark's hard items better
> than the frozen model, which remains `stable`
> ([results/lora-distill-qwen3.5-4b.md](results/lora-distill-qwen3.5-4b.md)). Read this
> pipeline as the tool it now is: a way to distil a stronger model's judgement **on your
> own workload**, where the states and questions you label are the ones you will serve.
> `reflex-distill label --think N` is offline only, like everything that reasons here.

`reflex-distill` is the training route that survived the lesson in
[results/frozen-vs-trained.md](results/frozen-vs-trained.md): adapters trained on labelled
datasets learn rules for *those datasets' wording* and apply them, over-confidently, to
neighbouring tasks. What did not transfer was the labels. What might transfer is
**judgement**, so the targets here come from a much stronger model answering diverse,
in-the-wild inputs through the same prompt the student is served with.

## The pipeline

    reflex-distill corpus    --per-domain 450 --out runs/distill_states.jsonl
    reflex-distill questions --in runs/distill_states.jsonl --out runs/distill_questions.jsonl \
                             --anchor-out runs/distill_anchor_questions.jsonl --model default   # any OpenAI-compatible chat model
    reflex-distill label     --model Qwen/Qwen3.8-27B --in runs/distill_questions.jsonl --out runs/distill_teacher.jsonl --permutations 2
    reflex-distill label     --model Qwen/Qwen3.5-4B  --in runs/distill_anchor_questions.jsonl --out runs/distill_anchor.jsonl --permutations 2
    reflex-distill mix       --teacher runs/distill_teacher.jsonl --anchor runs/distill_anchor.jsonl \
                             --out runs/distill_train.jsonl --eval-out runs/distill_eval.jsonl
    reflex-calibrate train   --data runs/distill_train.jsonl --val runs/distill_eval.jsonl --out runs/lora-distill --epochs 1 --lr 5e-5 --permutations 2

1. **corpus**: raw states from ten public domains (support tickets, PR hunks, chat prompts,
   reviews, news, contract clauses and terms, long reports, forum posts, assistant replies,
   passages with proposed answers). No labels are read. Nothing overlaps the external test
   sets or the benchmark; `reflex-data check-overlap` is run on the result.
2. **questions**: every state gets a few *bank* questions (the operational ones a decision
   model is actually asked about that kind of input) and a few written by a chat model in
   its own words, with its own option sets. Wording diversity is the point: the student
   must not learn one phrasing per task. A fixed 20 % of states is set aside as the
   **anchor** slice.
3. **label**: the teacher answers through reflex, so the targets are "what a stronger model
   reads off the same evidence" as full distributions, averaged over two option orders.
   The anchor slice is answered by the *frozen student itself*.
4. **mix**: teacher rows are the thing to learn; anchor rows, added at a quarter of their
   count, are a KL pull toward the base model on inputs the teacher never touched. A
   held-out 6 % of teacher rows measures fidelity to the teacher.
5. **train**: the ordinary LoRA trainer with soft targets.

## Gates, in order

* the teacher itself on the four external sets, before it labels anything: if it is not
  clearly better than the frozen student there, there is nothing to distil;
* the student adapter on the same external sets;
* the public benchmark items.

## Results

First run (2026-09-20): the student's held-out agreement with the 27B teacher rose from
0.65 to 0.85 and it moved toward the teacher on all four external sets with no collapse,
the first adapter to do so. On the benchmark's hard tier it did *not* improve (0.613 vs
0.658 frozen, teacher 0.703). Full numbers and the item-level analysis:
[results/lora-distill-qwen3.5-4b.md](results/lora-distill-qwen3.5-4b.md); teacher gates:
[results/teachers-27b-and-4b-think.md](results/teachers-27b-and-4b-think.md).
