r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 512
    hypers['seq_len'] = 64
    hypers['h_dim'] = 128
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.3
    hypers['learn_rate'] = 1e-2
    hypers['lr_sched_factor'] = .5
    hypers['lr_sched_patience'] = 3
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT I. SCENE 1. Marcellus and Bernardo have seen a ghost on the castle battlements"
    temperature = 0.4
    # ========================
    return start_seq, temperature



part1_q1 = r"""
**Your answer:**
We will split the corpus up into subsequences with a fixed length in order to slide a window along with the whole corpus one character at a time. This way, we allow each character a chance to be learned from the last N characters that preceded it, otherwise, it will not impact our learning from considering the whole text.
Additionally, with this suggested approach, the corpus can contain an enormous amount of characters which can cause our performance to be poor.
"""

part1_q2 = r"""
**Your answer:**
It is possible since we pass all the hidden states of the network at the end of the sequence to the next word to generate.
"""

part1_q3 = r"""
**Your answer:**
Our goal is to learn the sequences of a situation and then generate entirely new plausible sequences for this specific domain.
Therefore, the order of the different batches is crucial since we would need to preserve the order of words within the given sentences in the text and to keep the sense of the domain reasonable, otherwise, our training phase will be ineffective.
"""

part1_q4 = r"""
**Your answer:**
1. Lowering the temperature will result in less uniform distributions and lower variance between the generated word and the given sequence.
Therefore, by setting the temperature, we can control the distributions to be less uniform and to increase the chance of sampling the char(s) with the highest scores compared to the others.

2. Higher temperatures can create more noise (unreasonable words, special chars, etc.) and might lead to undesired results due to uniform distribution sampling.

3. Lower temperatures can lead to a variance decreasing which can affect our generation process to outcome more common characters/words, lower the diversity, and increase repetition due to the importance of the letters/words weights.
"""
# ==============

