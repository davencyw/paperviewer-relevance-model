# Relevance Model System - Training Code

This repository serves as a case study for the Paper Curation System. You can find more details about the system in the [project article](https://davencyw.github.io/davencyw.net/projects/paperviewer.html).  

**Note:**  
- This code is **not complete** and is intended to demonstrate the mechanisms that I use to train three main models: **generative**, **reward**, and **classification**. While it is not complete, it is runnable and demonstrates all major training and testing implementations.
- Major logic around state-handling (specifically around saving and loading generated and reviewed papers is missing) but is trivial to implement (specifically with a database).
- No real data is included in this repository for privacy reasons. All data has been randomly generated to enable the code to run, but the results may not be as meaningful.

For license information, see the [LICENSE](./LICENSE) file.  

---

## Requirements
- Python 3.11  
- Install dependencies from `requirements.txt` using your preferred Python virtual environment manager.  

```bash
pip install -r requirements.txt
```

---

## Training Modes

**Important:**  
This repository does not include training or test datasets. You will need to create your own data for these tasks.  

### 1. Train the Generative Paper Title Creator

#### Step 1: Initial Training
Train the generator on all papers to help it learn how to generate paper titles and abstracts:  

```bash
./src/main.py generator --verbose experiment-0 config finetune
```

#### Step 2: Fine-Tuning
Fine-tune the generator on positive samples by using the same command with a different dataset configuration.  

---

### 2. Reinforcement Learning with Human Feedback (RLHF)

After initial fine-tuning, you can use RLHF cycles to further refine the generator to align with specific interests, especially when the number of positive samples is limited.  

#### Workflow:
1. Run the RLHF cycle:  

   ```bash
   ./src/main.py generator --verbose experiment-0 config rlhf
   ```

2. This will:
   - Generate `N` samples.
   - Allow you to review them via a simple terminal-based UI (TUI) built on `curses`.
   - Use your annotations to perform a PPO (Proximal Policy Optimization) run.

3. Repeat the cycle until the generator consistently produces a satisfactory percentage of interesting papers.  

---

### 3. Fine-Tune the Classification Model

Once the generator is tuned, you can fine-tune the classification model using both the original annotated true paper samples and a variable amount of generated papers.  

#### Step 1: Training
Run the training process:  

```bash
./src/main.py classification --verbose experiment-0 config finetune
```

#### Step 2: Evaluation
Evaluate the trained classifier:  

```bash
./src/main.py classification --verbose experiment-0 config eval
```

This step generates evaluation plots, which are saved in the output folder.  

---

Feedback, comments, or constructive criticism are always welcome — feel free to reach out via [email](mailto:dave@davencyw.com).


David Schmidig

davencyw.net


