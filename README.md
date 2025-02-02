# RLHF Dialogue Summarization with PPO

## Background
Dialogue summarization is a complex task in natural language processing (NLP) that involves generating concise summaries from conversations while retaining key information. Reinforcement Learning with Human Feedback (RLHF) has emerged as a promising technique for refining model-generated summaries based on reward signals. This project employs Proximal Policy Optimization (PPO) to fine-tune a dialogue summarization model, aligning outputs with human preferences.

## Proximal Policy Optimization (PPO)
PPO is a policy optimization algorithm used in reinforcement learning to improve model performance by updating policies within a constrained step size. It utilizes a clipped objective function to balance exploration and exploitation while preventing drastic model updates. In this project, PPO is employed to optimize a summarization model using a reward model trained with human feedback.

## Project Objectives
The primary objectives of this project include:
- Implementing a dialogue summarization model using RLHF and PPO.
- Developing a reward model to assess the quality of generated summaries.
- Fine-tuning the model using human-aligned feedback.
- Evaluating the effectiveness of PPO in improving summarization quality.

## Challenges
During implementation, several challenges were encountered:
- **TRL Package Compatibility Issues**: The latest version of the `trl` package caused difficulties in configuring `PPOTrainer()`. This was eventually resolved by using an older version of the package that provided better compatibility.
- **Understanding PPO and Value Model**: The concepts behind PPO, particularly the **value model** and how it contributes to learning, were initially challenging to grasp. Extensive research and experimentation were required to understand their roles in reinforcement learning.

## Dataset
This project utilizes the **DialogSum** dataset from Hugging Face (`knkarthick/dialogsum`). The dataset consists of dialogue transcripts paired with human-written summaries, making it a suitable benchmark for training and evaluating dialogue summarization models.

## Data Preparation
The dataset is preprocessed to:
- Filter out overly short dialogues.
- Structure inputs with instruction-based formatting.
- Tokenize dialogues and summaries for model training.

## Models
This project uses:
- **Base Model**: PEFT `google/flan-t5-base` model that were trained from previous project.
- **Reward Model**: [Meta AI's RoBERTa-based hate speech model](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target) that output **logits** based on across two classes: `nothate` and `hate`.
- **PPO Policy Model**: A trainable version of PEFT `flan-t5-base` with an added value head for reinforcement learning.
- **Reference Model**: An unmodified version of the base model used for KL-divergence regularization.

## Fine-Tuning Approaches
The model is fine-tuned using the following techniques:
1. **Reward Modeling**: Training a reward model to evaluate summary quality.
2. **PPO Optimization**: Using reinforcement learning to iteratively improve summaries based on reward feedback while constraining divergence from the reference model.

## Evaluation Metrics
The model's performance is assessed using:
- **KL-Divergence (`objective/kl`)**: Ensures the policy model does not deviate excessively from the reference model.
- **PPO Mean Returns (`ppo/returns/mean`)**: Evaluates the average reward obtained during training.
- **Policy Advantage (`ppo/policy/advantages_mean`)**: Indicates how much better selected actions (summaries) are compared to the expected baseline.

## Results
After fine-tuning, the model demonstrated:
- Higher reward scores (mean: +27.66%, std: +43.23%), indicating better alignment with human preferences.
- A balanced trade-off between stability and innovation in generated summaries.

## How to Run
### 1. Clone the repository
```bash
git clone https://github.com/ivanluk914/RLHF-Dialogue-Summarizer-with-PPO-and-Reward-Model
cd RLHF-Dialogue-Summarizer-with-PPO-and-Reward-Model
```

### 2. Install dependencies
```bash
pip install transformers datasets peft torch evaluate numpy pandas tqdm
pip install trl==0.11.4
```

### 3. Run the notebook
```bash
jupyter notebook dialogue_summeries_with_RLHF.ipynb
```


## Acknowledgements
This project is based on research in RLHF and dialogue summarization. Special thanks to Hugging Face for providing the dataset and open-source implementations of RLHF techniques.

