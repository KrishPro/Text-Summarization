# Text Summarization

> Summarization is the task of condensing a piece of text to a shorter version, reducing the size of the initial text while at the same time preserving key informational elements and the meaning of content. Since manual text summarization is a time expensive and generally laborious task, the automatization of the task is gaining increasing popularity and therefore constitutes a strong motivation for academic research.

## Finalized Approaches
- [x] BART (BERT + GPT-2)
- [ ] LongFormer (Based on BART)
- [ ] BigBird (Based on Pegasus)
- [ ] GPT-2

## Conclusion
BART uses a full attention mask like BERT & also a partially attention mask like GPT. So it consumes (Resources used by BERT) + (Resources used by GPT).

Which means it **can** support sequences about 1024 words long, But at a high cost of resources. I consumes a lot of RAM

Free Google Colab **cannot** train BART for Text Summarization

But still, I will consider this branch as in-complete and I will try use BART for text summarization later on

## Research
Whole research, I did for this project is public [here](https://krishpro.github.io/text-summarization)
