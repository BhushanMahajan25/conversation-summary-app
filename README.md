# Chat Summarizer App

## Introduction

This project showcases a simple yet powerful web application developed using Streamlit, designed to demonstrate the capabilities of a fine-tuned language model for chat conversation summarization. The core focus of the project lies in utilizing the `google/flan-t5-base` pre-trained language model, fine-tuned on `knkarthick/dialogsum` dataset, a text-to-text sequence-to-sequence language model.

### Key Features:

- **Web App:** Experience a simple UI allowing the simulation of chat conversations between two entities. The simplicity of the app highlights the underlying power of the language model.
  
- **Language Model (LLM):** Explore the intricacies of the language model, its fine-tuning process, and the specific approach taken to enhance its text summarization capabilities.

- **Evaluation:** Gain insights into the evaluation metrics, specifically ROUGE (Recall-Oriented Understudy for Gisting Evaluation), to assess the performance improvement achieved through fine-tuning.

## Web App

### Functionality Overview

The Chat Summarizer Web App offers a straightforward platform for simulating chat conversations and generating summaries. The following features are highlighted:

#### Conversation Simulation

- **Toggling Mechanism:** The app incorporates a toggling mechanism that allows a single user to simulate a conversation between two entities. This design choice simplifies the user experience for the purpose of showcasing the language model's summarization capabilities.

#### Summary Generation

- **Generate Summary Button:** Users can initiate the summary generation process by clicking the `Summarize` button. This action formats the simulated conversation as per the requirements of the fine-tuned language model and triggers the model to generate a concise summary.

### Design Philosophy

The web app prioritizes simplicity to draw attention to the core functionality—language model-based chat summarization. While the app may not implement industry best practices for a full-fledged chat application, it serves as a purposeful and illustrative tool for understanding the capabilities of the fine-tuned language model.

### How to Use

1. **Simulate Chat:** Type messages for each user to simulate a chat conversation. The application will toggle between the users internally.
2. **Generate Summary:** Click the `Summarize` button to observe the language model's summarization in action.
3. **View Summary:** The generated summary will be displayed on the screen.

### Note

This web app is intentionally designed for simplicity and educational purposes, focusing on the core functionality of chat summarization using the fine-tuned language model.

## Large Language Model (LLM)

### Base Model

The foundation of the Chat Summarizer project lies in the utilization of the `google/flan-t5-base` language model. This model is a pre-trained text-to-text sequence-to-sequence language model that serves as the starting point for our fine-tuning process.

#### `google/flan-t5-base` model
![google/flan-t5-base](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/flan2_architecture.jpg)

To learn more about the base model, visit here: https://huggingface.co/google/flan-t5-base


#### Significance of the Base Model

The choice of the `google/flan-t5-base` model provides a robust starting point for our chat summarization task. Leveraging a pre-trained model allows us to tap into the wealth of knowledge and linguistic nuances captured during its initial training on diverse text corpora.

#### Why `google/flan-t5-base`?

The base model's selection is motivated by its performance and versatility in handling text generation tasks. The capabilities of the model, combined with the subsequent fine-tuning process, aim to enhance its proficiency in summarizing chat conversations.

### Dataset

- **Source:** The fine-tuning is performed using the `knkarthick/dialogsum` dataset, sourced from Hugging Face.

    > "DialogSum is a large-scale dialogue summarization dataset, consisting of 13,460 (Plus 100 holdout data for topic generation) dialogues with corresponding manually labeled summaries and topics."

To learn more about the dataset, visit here: https://huggingface.co/datasets/knkarthick/dialogsum


### Transfer Learning and Fine-tuning Approach

Since `knkarthick/dialogsum` dataset is relatively small (Size of downloaded dataset files: 13.1 MB), I used transfer learning by fine-tuning the base model `google/flan-t5-base` for the specific task of chat conversation summarization.

#### PEFT (Parameter-Efficient Fine-Tuning)
- The fine-tuning process incorporates PEFT, which stands for Parameter-Efficient Fine-Tuning. This technique focuses on training a minimal subset of the base model's parameters and freezes most of the pre-trained network, making the fine-tuning process more computationally and space efficient while maintaining effective model adaptation.

#### LoRA (Low-Rank Adaptation) for LLM
Here, I implemented a variant of PEFT called Low-Rank Adaption (LoRA).
Hu et al., [“LoRA: Low-Rank Adaptation of Large Language Models”](https://arxiv.org/abs/2106.09685), in ICLR 2021, proposes a technique that freezes all the low-level layers in the pre-trained model, while only training a shallow, adaptable layer near the output.

Fine-tuning only the adaptable (Feed Forward Network) FFN layer minimizes forgetting previous knowledge while allowing model customization for summarization. LoRA also lowers compute requirements compared to full fine-tuning.

![LoRA](https://github.com/BhushanMahajan25/conversation-summary-app/assets/29192863/5d6b80ce-0f23-485e-9fdb-f332bac44f41)

To learn more about "LoRA", visit here: https://huggingface.co/docs/peft/conceptual_guides/lora

### Hyperparameters and LoRA configuration

Due to resource constraints on the available compute following configuration for efficient fine-tuning were selected:

- **LoRA Cofigurations**:
  
    ```
        lora_config = LoraConfig(
            r=32, # Rank
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
        )
    ```

    This configuration results in ***3,538,944*** trainable parameters out of the total ***251,116,800*** parameters in `google/flan-t5-base`.

    So only ***1.41%*** of the parameters are fine-tuned, leading to very efficient and targeted adaption for summarization while retaining prior knowledge in the frozen lower layers.

- **Hyperparameters**:

    ```
    peft_training_args = TrainingArguments(
        auto_find_batch_size=True,
        learning_rate=1e-3, # Higher learning rate than full fine-tuning.
        num_train_epochs=5,
        logging_steps=200
    )
    ```

## Evaluation

To evaluate the quality of the summarization models, I used the standard ROUGE (Recall-Oriented Understudy for Gisting Evaluation) set of metrics.

As per [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013) (Lin, 2004), ROUGE compares the machine generated summaries to human-written reference summaries using n-gram overlap statistics. It measures coverage of key textual elements between the candidates and references.

ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum are reported here in this project. Higher ROUGE scores indicate better correlation with human labels.

* **ROUGE-N**: Overlap of n-grams between candidate and reference summaries. ROUGE-1 uses unigrams, ROUGE-2 uses bigrams.
* **ROUGE-L**: Longest common subsequence overlap based on sentence structure.
* **ROUGE-Lsum**: Modification that adds a weighted sum instead of longest match.

### Results


| Metric | `google/flan-t5-base` model | PEFT model | Abs. Diff | % Improvement |  
|-|-|-|-|-|
| Rouge-1 | 0.23 | 0.41 | +0.18 | +78% |
| Rouge-2 | 0.07 | 0.17 | +0.10 | +143% | 
| Rouge-L | 0.20 | 0.33 | +0.13 | +65% |
| Rouge-Lsum | 0.20 | 0.33 | +0.13 | +65% |

The fine-tuned model seems perform better than the `google/flan-t5-base` across all ROUGE metrics, indicating the Parameter Efficient Fine-Tuning successfully adapts the model to the summarization task:

* ~23% vs ~41% ROUGE-1 shows higher unigram overlap with references.
* ~7% vs ~17% ROUGE-2 suggests the PEFT model captures more key bigrams.
* ~20% vs ~33% ROUGE-L indicates improved sentence structure match.

As the table shows, the PEFT model achieves seems to show improvement over the `google/flan-t5-base` across all ROUGE metrics:

- **+78%** increase in unigram overlap
- **+143%** gain in ngram structure match   
- **+65%** boosted sentence structure correlation

This highlights the gains in summarization performance from this targeted Parameter Efficient Fine-Tuning approach. Freezing unneeded capacity allowed efficiently focusing learning on only the parameters needed to boost summarization quality greatly.

So the PEFT model's generated summaries correlate better with human labels after efficient adaption, while retaining pretrained knowledge.
