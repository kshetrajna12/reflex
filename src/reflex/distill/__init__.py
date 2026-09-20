"""Distillation: a stronger model's typed judgements, as soft targets for the 4B student.

    reflex-distill corpus    -> raw, unlabelled states from many public domains
    reflex-distill questions -> typed questions per state (a fixed bank + LLM-written ones)
    reflex-distill label     -> a teacher model answers them through reflex; the
                                distributions become the training targets
    reflex-distill mix       -> teacher rows + frozen-student "anchor" rows -> train/eval files

Why: adapters trained on labelled datasets learned rules for those datasets' wording and
applied them, over-confidently, to neighbouring tasks (docs/results/frozen-vs-trained.md).
A teacher's answers to *diverse, in-the-wild* inputs are general judgement, so what the
student learns from them has a chance of transferring. The anchor rows (the frozen
student's own answers on a disjoint slice) act as a KL penalty toward the base model.
"""
