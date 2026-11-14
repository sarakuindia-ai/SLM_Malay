# SLM_Malay
Small Language Model_Malay conversations
Building a 46M Parameter Malay SLM from Scratch
A Case Study in Specialized Tokenization & Scale-Up

This carousel outlines how we built a highly specialized Small Language Model customized specifically for conversational Malay customer support interactions.

The Core Challenge: Standard tokenizers fragment native Malay vocabulary, causing extremely high perplexity and poor phrase structure.
The Solution: Native Byte-Pair Encoding (BPE) vocabulary + architectural scaling.

Performance Breakthrough: The Tokenizer Effect
Our empirical benchmark comparing the standard tokenizers against our customized native configuration:

Model Setup	Vocab Size	Val Loss	Val Perplexity
Baseline GPT-2 Tokenizer	50,257	6.9038	996.03 (Very Poor)
Custom BPE Tokenizer	16,000	5.0074	149.51 (Vast Improvement)
Scaled 46M Model (Our Best)	16,000	2.2632	9.61 (Elite Coherence)
Takeaway: Aligning vocabulary configuration to native grammar dynamics reduces perplexity by over 99%

Model Architecture & Scaling Design
How we designed our model parameter footprint to fit into the consumer-grade deployment space:

Total Parameters: 46,152,704 (46.15M)
Non-embedding Parameters: 37,960,704 (37.96M)
Layers (n_layer): 12 Transformer block layers
Embedding Dimension (n_embd): 512
Attention Heads (n_head): 8 (dimension of 64 per head)
Context Window (block_size): 256 tokens

How to Run Inference (Workflow)
Here is the absolute minimal production workflow setup to interact with the best custom model:
**# 1. Load Custom Tokenizer tokenizer = Tokenizer.from_file("malay_bpe_tokenizer.json") # 
2. Format Input with Template prompt = "<|user|> macam mana saya nak register akaun baru <|assistant|> " input_ids = torch.tensor(tokenizer.encode(prompt).ids).unsqueeze(0).to(device) # 
3. Autoregressive Generation outputs = scaled_model.generate(input_ids, max_new_tokens=100, temperature=0.7) print(tokenizer.decode(outputs[0].tolist()))**

Future Roadmap: How to Fine-Tune on More Data
To continue scaling the capabilities of our native Malay Small Language Model:

Extend Training Subsets: Increase data range extraction past the 20,000-sample limit into the full 137,000+ conversation dataset.
Pack Sequences: Implement structured block padding to prevent wasteful padding tokens when training context windows expand.
Learning Rate Warmup: Employ a Cosine Decay scheduler starting from 3e-4 down to 3e-5 with dynamic weight decay of 0.1 for maximum stabilization.
