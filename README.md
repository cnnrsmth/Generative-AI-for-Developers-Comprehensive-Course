# Generative AI for Developers

## Course Video

[![Generative AI for Developers - freeCodeCamp.org](https://img.youtube.com/vi/F0GQ0l2NfHA/0.jpg)](https://www.youtube.com/watch?v=F0GQ0l2NfHA)

## Project Overview

This repository documents my journey through the "Generative AI for Developers" course by freeCodeCamp.org. The course provides an in-depth exploration of generative AI, covering topics such as large language models, data preprocessing, fine-tuning, Retrieval-Augmented Generation (RAG), and hands-on projects utilizing tools like Hugging Face, OpenAI, and LangChain.

## Course Notes

<details>
<summary><strong>Introduction to Generative AI</strong></summary>

- **Definition and Relationship to Deep Learning**:  
  Generative AI is a subset of deep learning focused on creating new content such as text, images, music, or other data types. The models used in Generative AI, often referred to as Generative Models, learn to produce outputs that resemble the data they were trained on.

- **Training with Large Datasets**:  
  Generative models are trained using vast amounts of data. Unlike traditional supervised learning, where labeled data (input-output pairs) is required, generative models often rely on unlabeled or partially labeled data. This is because their objective is not to classify or predict specific outcomes but to understand and replicate the underlying patterns or distributions within the training data.

- **Learning from Data Distributions**:  
  During training, a generative model analyzes the relationships and patterns in the data. It does not explicitly need labeled examples to perform this task. Instead, it attempts to capture the structure and statistical characteristics of the dataset.

- **Use of Unstructured Data in Generative AI**:  
  Unstructured data—such as text, images, or audio—is a primary source for training Generative AI models. In the case of models like Large Language Models (LLMs), the training involves feeding vast amounts of unstructured data (e.g., books, articles, or web pages). These models learn to generate coherent and contextually relevant outputs by identifying patterns within this unstructured input.

### **What Are Generative Models?**

Generative models try to **understand how data is created**. They don’t just look at patterns—they learn the full story of the data, including both:

1. What the input looks like (e.g., an image of a cat).
2. How the input relates to the output (e.g., "this is a cat").

Once trained, they can create (or "generate") new data that looks like the original.

### **Example: Generative AI (Text or Image Creation)**

- **Case Study:** _ChatGPT (Text Generation)_  
  ChatGPT learned from millions of text samples to understand how words and ideas are related. It doesn’t just predict what comes next—it can generate completely new, coherent responses.
- **Another Example:** _DALL·E (Image Generation)_  
  DALL·E generates realistic images (e.g., "a panda surfing"). It has learned how visual features like shapes, colors, and objects combine to create images.

### **What Can They Do?**

- Generate new content: write poems, create images, compose music.
- Fill in missing information: restore old photos or predict missing text.

---

### **What Are Discriminative Models?**

Discriminative models are **decision-makers**. They focus on solving problems like:

1. "Is this a cat or a dog?"
2. "Will this customer buy a product?"

They don’t try to understand how data is created—they focus on **drawing boundaries** between classes (e.g., separating cats from dogs).

### **Example: Spam Email Classifier**

- **Case Study:** _Gmail Spam Filter_  
  Gmail uses a discriminative model to classify emails as "Spam" or "Not Spam" by looking at features like keywords, sender address, and formatting.

### **What Can They Do?**

- Classify objects (e.g., "cat or dog").
- Predict outcomes (e.g., "Will it rain tomorrow?").
- Rank or sort information (e.g., movie recommendations).

---

### **What is a Large Language Model (LLM)?**

An **LLM** is an AI model trained to understand, generate, and analyze human-like text. Think of it as a machine that predicts and constructs meaningful sentences, paragraphs, or even documents, based on the input it receives. It’s the backbone of tools like ChatGPT, helping to create natural, conversational, and context-aware text.

---

### **How Does an LLM Work?**

At a high level, an LLM predicts the most likely next word in a sequence. If you type "The sky is," the model predicts "blue" because it has seen similar text patterns during training. But this basic prediction scales up to understanding and creating much more complex text structures.

To achieve this, an LLM uses:

1. **Training Data:**  
   Massive datasets that include books, articles, websites, and more. These datasets allow the model to understand vocabulary, grammar, facts, and even cultural nuances.

2. **Patterns and Probabilities:**  
   LLMs don’t "know" language the way humans do. Instead, they rely on probabilities. For example:
   - If the input is "I love eating pizza," the model assigns a high probability to "pizza" after "eating" based on patterns it learned during training.

---

### **Key Architecture: Transformers**

Transformers are the core architecture behind modern LLMs (introduced in the 2017 paper, _Attention Is All You Need_). Here's a simple-to-detailed progression:

1. **The Simple Explanation:**  
   Transformers analyze the entire input (not just the most recent word) and figure out which parts of the input are most important for understanding the text.

2. **The Slightly Technical View:**

   - A transformer processes input in parallel (unlike older models like RNNs or LSTMs, which process word by word).
   - It uses an **attention mechanism** to decide which words or tokens matter most. For example, in the sentence, _"The cat sat on the mat, and it was happy,"_ the word "it" refers to "cat." The attention mechanism helps identify this relationship.

3. **Key Components of Transformers:**

   - **Tokenization:**  
     Breaks text into smaller chunks (tokens). For example, "I’m running" might become ["I", "’m", "running"].

   - **Embeddings:**  
     Converts each token into a vector (a series of numbers). This vector represents the word in a way that captures its meaning and relationships with other words.

   - **Self-Attention Mechanism:**  
     Determines how important each token is in relation to others. For instance, in "She went to the store," the model links "She" to "went" and "store" to create context.

   - **Feedforward Networks:**  
     After applying attention, the transformer processes information through neural layers to learn more abstract relationships.

</details>
