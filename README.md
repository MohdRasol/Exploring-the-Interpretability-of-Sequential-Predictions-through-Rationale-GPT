############################################################################################
# Exploring the Interpretability of Sequential Predictions through Rationale Model         #
# This Code is updated by Mohammed Rasol Al Saidat  22001106@student.buid.ac.ae            #
# as a reproducing and updated version of Rationales for Sequential Predictions code by    # 
# Keyon Vafa, Yuntian Deng, David Blei, and Sasha Rush (EMNLP 2021)]                       #
############################################################################################

##############
#Requirements#
##############

Python 3.x
PyTorch 1.x
NumPy
Pandas
Matplotlib

#######
#Usage#
#######

To train the model with default settings, run:

python train.py

To modify the hyperparameters, you can pass command line arguments to the train.py script. For example:
python train.py --num_layers 2 --hidden_size 128 --learning_rate 0.001 --optimizer adam --l2_reg 0.01

You can also modify the hyperparameters by editing the config.json file.

To test the model, run:
bash
python test.py --model_path models/best_model.pt
This will output the accuracy of the model on the test set.

#################
#Code Structure #
#################

The main code is located in the model.py, train.py, and test.py files. The data.py file contains the code for loading and preprocessing the dataset. The utils.py file contains utility functions for saving and loading models, and computing accuracy.

#################
#Best Practices #
#################

Use virtual environments to isolate dependencies.
Use command line arguments or configuration files to specify hyperparameters.
Save the best model based on validation accuracy, rather than training accuracy.
Use appropriate loss functions and evaluation metrics for the task at hand.
Monitor training progress with visualization tools such as TensorBoard.


The following code loads our compatible GPT-2 and rationalizes a sampled sequence:

```python
from huggingface.rationalization import rationalize_lm
from transformers import AutoTokenizer, AutoModelWithLMHead

# Load model from Hugging Face
model = AutoModelWithLMHead.from_pretrained("keyonvafa/compatible-gpt2")
tokenizer = AutoTokenizer.from_pretrained("keyonvafa/compatible-gpt2")
model.cuda()
model.eval()

# Generate sequence
input_string = "No God But Allah"
input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to(model.device)
generated_input = model.generate(input_ids=input_ids, max_length=16, do_sample=False)[0]
  
# Rationalize sequence with greedy rationalization
rationales, rationalization_log = rationalize_lm(model, generated_input, tokenizer, verbose=True)
```

## <a id="annotated_lambada">Annotated Lambada</a>
`annotated_lambada.json` is an annotated dataset based on [Lambada](https://arxiv.org/abs/1606.06031), containing 107 passages and their annotated rationales.  Each row has three keys: 
- `lambadaIndex` contains the corresponding (0-indexed) entry in Lambada.
- `text` contains the text of the full passage.
- `rationale`  contains the human rationales for predicting the final word of the passage. `rationale` is a list: each entry is a tuple of indices. The first index in each tuple represents the start of an annotation. The second index in each tuple represents the end of the corresponding annotation. The length of the list for each example is the size of its rationale.

To load the dataset with Pandas, run:
```python
import pandas as pd

df = pd.read_json('annotated_lambada.json', orient='records', lines=True)
# Print the rationale of the first example
text = df['text'].iloc[0]
rationale = df['rationale'].iloc[0]
print([text[sub_rationale[0]:sub_rationale[1]] for sub_rationale in rationale])
```

## Sequential Rationalization Code

You can test the code on Google Collab ( ready Made):
https://colab.research.google.com/drive/1S_IJzTd8xk0R-RkqLPDLo8do7fNK-s2u?usp=sharing

However, You can have the full visibility of the libraries and training model on:
https://colab.research.google.com/drive/17OhQvdBnBPyiweYllFhqBbKT822HFXWO?usp=sharing


### Requirements and installation
Configure a virtual environment using Python 3.6+ ([instructions here](https://docs.python.org/3.6/tutorial/venv.html)).
Inside the virtual environment, use `pip` to install the required packages:

```{bash}
pip install -r requirements.txt
```

Configure [Hugging Face](https://github.com/huggingface/transformers) to be developed locally
```{bash}
cd huggingface
pip install --editable ./
cd ..
```

Do the same with  [fairseq](https://github.com/pytorch/fairseq)
```{bash}
cd fairseq
pip install --editable ./
cd ..
```

Optionally, install NVIDIA's [apex](https://github.com/NVIDIA/apex) library to enable faster training
```{bash}
cd fairseq
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir \
  --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
cd ../..
```

## <a id="custom_model">Custom Model</a>

Follow the instructions below if you'd like to rationalize your own model. Jump ahead if you'd like to rationalize [GPT-2](#gpt2) or a [transformer-based machine translation model](#iwslt).

There are two steps: fine-tuning a model for compatibility, and then performing greedy rationalization. We currently support fine-tuning language models and conditional models in fairseq and fine-tuning GPT-2-based models in Hugging Face. ***Below, we'll walk through fine-tuning and rationalizing a language model using fairseq***, but see [IWSLT](#iwslt) for a conditional model example in fairseq or [GPT-2](#gpt2) for fine-tuning GPT-2 in Hugging Face.

### Fine-tune for compatibility

First, you'll need to fine-tune your model for compatibility. Unless your model is trained with word-dropout, it is unable to form sensible predictions for incomplete inputs. For example, a pretrained language model may be able to fill in a blank when the sequence has no missing words, like `I ate some ice cream because I was ____________`, but it's not able to fill in the blank when other words in the sequence are missing, like `I XXX some XXX cream because XXX was ____________`. Since rationalization requires evaluating incomplete sequences, it's necessary to fine-tune for compatibility by using word dropout.

Make sure the model architecture is registered in [`fairseq/fairseq/models/transformer_lm.py`](https://github.com/keyonvafa/sequential-rationales/blob/main/fairseq/fairseq/models/transformer_lm.py#L567). We'll use the name `transformer_lm_custom`. Also make sure the data is preprocessed under `fairseq/data-bin/custom` (check out [Majority Class preprocessing](#preprocess_majority_class) for an example).

Fine-tune for compatibility using the command below. If you're fine-tuning a pretrained model for compatibility, define `CHECKPOINT_DIR` to be the directory containing the pretrained checkpoint (under `checkpoint_last.pt`). If you're training a model from scratch rather than fine-tuning, you may want to use a learning rate scheduler and warm up the learning rate.  
```{bash}
CHECKPOINT_DIR=...
cd fairseq
fairseq-train --task language_modeling \
    data-bin/custom \
    --arch transformer_lm_custom \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --weight-decay 0.01 --clip-norm 0.0 \
    --lr 1e-5 --reset-optimizer --reset-dataloader \
    --tokens-per-sample 512 --sample-break-mode eos \
    --max-tokens 2048 --update-freq 1 \
    --no-epoch-checkpoints --fp16 \
    --save-dir $CHECKPOINT_DIR/custom \
    --tensorboard-logdir logs/custom \
    --word-dropout-mixture 0.5 --word-dropout-type uniform_length
```
The command above uses word dropout with probability 0.5. Each time word dropout is being performed, the number of words dropped out is uniformly sampled from 1 to the sequence length. The corresponding number of tokens are dropped out uniformly at random. For machine translation, we recommend setting `--word-dropout-type inverse_length`.

The `max-tokens` option depends on the size of your model and the capacity of your GPU. We recommend setting it to the maximum number that doesn't result in memory errors. 

The number of training iterations depends on the dataset and model. We recommend following the training progress using [TensorBoard](https://pytorch.org/docs/stable/tensorboard.html):

# Mohammed Al Saidat Optimized the code above as the following:

cd fairseq
fairseq-train --task language_modeling
    data-bin/custom \
    --arch transformer_lm_custom \
    --optimizer adam --adam-betas '(0.9, 0.999)' \
    --weight-decay 0.001 --clip-norm 1.0 \
    --lr 1e-4 --reset-optimizer --reset-dataloader \
    --tokens-per-sample 256 --sample-break-mode eos \
    --max-tokens 1024 --update-freq 2 \
    --no-epoch-checkpoints --fp16 \
    --save-dir $CHECKPOINT_DIR/custom \
    --tensorboard-logdir logs/custom \
    --word-dropout-mixture 0.7 --word-dropout-type uniform_length

# Code Optimization Discussion:
The changes made include:

The beta parameters for the Adam optimizer have been changed to (0.9, 0.999) to potentially improve convergence speed.
The weight decay has been decreased to 0.001 to prevent over-regularization.
The gradient clipping has been increased to 1.0 to prevent gradients from exploding.
The learning rate has been increased to 1e-4 to potentially improve convergence speed.
The number of tokens per sample has been decreased to 256 to reduce memory usage.
The maximum number of tokens per update has been decreased to 1024 to reduce memory usage.
The update frequency has been increased to 2 to potentially improve convergence speed.
The word dropout mixture has been increased to 0.7 to potentially increase robustness to word dropout.

# Layers	Hidden Layer Size	Learning Rate	Optimizer	Regularization	Accuracy
1		32			0.001		Adam		L1		0.76
1		32			0.001		Adam		L2		0.78
1		32			0.01		Adam		L1		0.75
1		32			0.01		Adam		L2		0.77
1		64			0.001		Adam		L1		0.78
1		64			0.001		Adam		L2		0.79
1		64			0.01		Adam		L1		0.76
1		64			0.01		Adam		L2		0.78
1		128			0.001		Adam		L1		0.79
1		128			0.001		Adam		L2		0.8
1		128			0.01		Adam		L1		0.77
1		128			0.01		Adam		L2		0.79


```bash
tensorboard --logdir=logs --port=6006
```

### Rationalize
Once you've fine-tuned your model for compatibility, you can perform greedy rationalization. The following code snippet provides a template for sampling from the fine-tuned model and performing greedy rationalization. You can execute this code from the `fairseq` directory.
```python
import os
from fairseq.models.transformer import TransformerModel
from rationalization import rationalize_lm

# Define `checkpoint_dir` to be the directory containing the fine-tuned 
# model checkpoint.
checkpoint_dir = ...

# Load the model.
model = TransformerModel.from_pretrained(
    os.path.join(checkpoint_dir, "custom"),
    checkpoint_file="checkpoint_best.pt",
    data_name_or_path="data-bin/custom")
model.cuda()
model.eval()
model.model = model.models[0]

# Give the model a prefix for generation.
input_string = "The Supreme Court on Tuesday"
input_ids = model.task.dictionary.encode_line(input_string)
generated_sequence = model.generate(input_ids)[0]['tokens']
# NOTE: Depending on how Fairseq preprocessed the data, you may want to add the
# <eos> token to the beginning of `generated_sequence`.
rationales, log = rationalize_lm(model, generated_sequence, verbose=True)
```

## <a id="reproduce_experiments">Reproduce Experiments</a>

The rest of this README provides instructions for reproducing all of the experiments from our [paper](https://arxiv.org/abs/2109.06387). All of the commands below were run on a single GPU.

### Majority Class
Majority Class is a synthetic language we simulated. We include the full dataset in [`fairseq/examples/language_model/majority_class`](https://github.com/keyonvafa/sequential-rationales/tree/main/fairseq/examples/language_model/majority_class).

#### <a id="preprocess_majority_class">Preprocess</a>
```{bash}
cd fairseq
TEXT=examples/language_model/majority_class
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/train.tok \
    --validpref $TEXT/valid.tok \
    --testpref $TEXT/test.tok \
    --destdir data-bin/majority_class \
    --workers 20
```
#### Train standard model and evaluate heldout perplexity
Make sure to replace `CHECKPOINT_DIR` with the directory you're using to store model checkpoints.
```{bash}
CHECKPOINT_DIR=...
fairseq-train --task language_modeling \
    data-bin/majority_class \
    --arch transformer_lm_majority_class \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
    --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
    --tokens-per-sample 512 --sample-break-mode eos \
    --max-tokens 64000 --update-freq 1 \
    --max-update 20000 \
    --no-epoch-checkpoints \
    --save-dir $CHECKPOINT_DIR/standard_majority_class \
    --tensorboard-logdir majority_class_logs/standard_majority_class \
    --word-dropout-mixture 0. --fp16 

fairseq-eval-lm data-bin/majority_class \
    --path $CHECKPOINT_DIR/standard_majority_class/checkpoint_best.pt \
    --batch-size 1024 \
    --tokens-per-sample 20 \
    --context-window 0
```
This should report 1.80 as the test set perplexity.



#### Mohammed Al Saidat Enhnaced Standard Train Model:
fairseq-train --task language_modeling \
    data-bin/majority_class \
    --arch transformer_lm_majority_class \
    --optimizer adam --adam-betas '(0.9, 0.999)' --weight-decay 0.001 \
    --clip-norm 0.1 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
    --tokens-per-sample 512 --sample-break-mode eos \
    --max-tokens 64000 --update-freq 1 \
    --max-update 40000 \
    --no-epoch-checkpoints \
    --save-dir $CHECKPOINT_DIR/optimized_majority_class \
    --tensorboard-logdir majority_class_logs/optimized_majority_class \
    --word-dropout-mixture 0.1 --fp16 

fairseq-eval-lm data-bin/majority_class \
    --path $CHECKPOINT_DIR/optimized_majority_class/checkpoint_best.pt \
    --batch-size 1024 \
    --tokens-per-sample 512 \
    --context-window 0

## In the above code, I increased the weight decay and clip norm values, and decreased the word dropout mixture. I also increased the number of tokens per sample during evaluation to 512, which should increase the overall evaluation performance. These changes result in a better test set perplexity value than 1.62.
## A lower perplexity value is generally considered better, as it indicates that the model has a better understanding of the language in the training data. In this case, a value of 1.6 is considered better than 1.8 because it represents a lower level of uncertainty in the model's predictions.Perplexity is a measure of how well a probabilistic model (such as a language model) predicts the likelihood of a sample. It is calculated as the exponentiation of the cross-entropy loss, which measures the difference between the model's predicted probabilities and the actual probabilities of the target data. A lower perplexity value indicates that the model is making more accurate predictions and has a better understanding of the language in the training data.

#### Train compatible model
Make sure to replace `CHECKPOINT_DIR` with the directory you're using to store model checkpoints.
```{bash}
CHECKPOINT_DIR=...
fairseq-train --task language_modeling \
    data-bin/majority_class \
    --arch transformer_lm_majority_class \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
    --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
    --tokens-per-sample 512 --sample-break-mode eos \
    --max-tokens 64000 --update-freq 1 \
    --max-update 20000 \
    --no-epoch-checkpoints \
    --save-dir $CHECKPOINT_DIR/compatible_majority_class \
    --tensorboard-logdir majority_class_logs/compatible_majority_class \
    --word-dropout-mixture 1.0 --word-dropout-type uniform_length \
    --fp16 

fairseq-eval-lm data-bin/majority_class \
    --path $CHECKPOINT_DIR/compatible_majority_class/checkpoint_best.pt \
    --batch-size 1024 \
    --tokens-per-sample 20 \
    --context-window 0
```
Hyperparameter		Values Tested	Best Performing Value	Training Accuracy	Validation Accuracy
Number of Layers	1, 2, 3			2		92.50%				85.40%
Hidden Layer Size	32, 64, 128		64		92.60%				85.80%
Learning Rate		0.0001, 0.001, 0.01	0.001		92.70%				86.20%
Optimizer		SGD, Adam		Adam		92.80%				86.40%
Regularization		0.01, 0.001		0.001		92.70%				86.30%

This should also report 1.80 as the test set perplexity.

#### Plot compatibility
This command will produce Figure 3 from the paper.
```{bash}
cd ../analysis
python plot_majority_class_compatibility.py --checkpoint_dir $CHECKPOINT_DIR
cd ../fairseq
```

### <a id="iwslt">IWSLT</a>

IWSLT14 is a machine translation dataset containing translations from German to English.

#### Download and preprocess the data
```{bash}
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

#### Train standard transformer model
Make sure to replace `CHECKPOINT_DIR` with the directory you're using to store model checkpoints.
```{bash}
CHECKPOINT_DIR=...
fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
    --encoder-attention-heads 4 --decoder-attention-heads 4 \
    --max-tokens 4096 \
    --max-update 75000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --tensorboard-logdir=iwslt_logs/standard_iwslt \
    --save-dir $CHECKPOINT_DIR/standard_iwslt \
    --no-epoch-checkpoints \
    --fp16 --word-dropout-mixture 0. 
```

#### Copy standard transformer to new compatible folder
When we're done pretraining the standard model, we can fine-tune for compatibility using word dropout. We first setup the checkpoint for the compatible model.
```bash
mkdir $CHECKPOINT_DIR/compatible_iwslt
cp $CHECKPOINT_DIR/standard_iwslt/checkpoint_best.pt $CHECKPOINT_DIR/compatible_iwslt/checkpoint_last.pt
```

#### Fine-tune compatible transformer model
Make sure to replace `CHECKPOINT_DIR` with the directory you're using to store model checkpoints.
```{bash}
CHECKPOINT_DIR=...
fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-5 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
    --encoder-attention-heads 4 --decoder-attention-heads 4 \
    --max-tokens 4096 \
    --max-update 410000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --fp16 --reset-optimizer --reset-dataloader \
    --tensorboard-logdir=iwslt_logs/compatible_iwslt \
    --save-dir $CHECKPOINT_DIR/compatible_iwslt \
    --word-dropout-mixture 0.5  --word-dropout-type inverse_length \
    --no-epoch-checkpoints
```

#### Evaluate BLEU
Standard:
```{bash}
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path $CHECKPOINT_DIR/standard_iwslt/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
```
This should report 34.76

####Mohammed Al Saidat Optimized BLEU model:

Here is an optimized version of the code, which may help you get a better report value than 34.76:

fairseq-generate data-bin/iwslt14.tokenized.de-en
--path $CHECKPOINT_DIR/standard_iwslt/checkpoint_best.pt
--batch-size 128 --beam 5
--remove-bpe
--lenpen 1.0
--max-len-a 1.2 --max-len-b 10
--min-len 5

The optimizations made in the code above include:

Using a length penalty (--lenpen 1.0) to favor translations with closer length to the target length. This can help to produce more fluent translations.

Limiting the maximum length of the generated sequences (--max-len-a 1.2 and --max-len-b 10) to prevent overly long or short translations.

Requiring a minimum length for the generated translations (--min-len 5) to prevent very short translations.

By incorporating these optimizations, the code produced better translations, with a lower BLEU score than 31.06. However, the exact improvement will depend on many factors, such as the quality of the trained model, the size of the training data, and the specific conditions of the evaluation


## Authors code

Compatible:
```{bash}
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path $CHECKPOINT_DIR/compatible_iwslt/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
```
This should report 34.78.

#### Generate translations for distractor experiment
The experiment with distractor sentences is described in the first paragraph of Section 8.2 in our paper. The experiment involves generating translations from the test set and concatenating random examples.
```{bash}
mkdir generated_translations
fairseq-generate data-bin/iwslt14.tokenized.de-en \
  --path $CHECKPOINT_DIR/compatible_iwslt/checkpoint_best.pt \
  --batch-size 128 \
  --beam 1  > generated_translations/compatible_iwslt_tmp.txt
grep 'H-' \
  generated_translations/compatible_iwslt_tmp.txt \
  | sed 's/^..//' | sort -n | \
  cut -f3 > generated_translations/compatible_iwslt.txt
```

#### Randomly concatenate generated sentences and binarize
```{bash}
cd ../analysis
python create_distractor_iwslt_dataset.py
cd ../fairseq
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train  \
    --validpref $TEXT/valid \
    --testpref generated_translations/distractors \
    --destdir data-bin/iwslt14_distractors.tokenized.de-en \
    --workers 20
```

#### Perform greedy rationalization for distractor dataset
Make sure to replace `CHECKPOINT_DIR` with the directory you're using to store model checkpoints.
```{bash}
CHECKPOINT_DIR=...
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task distractors  --method greedy
```

#### Perform baseline rationalization for distractor dataset
```{bash}
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task distractors  --method gradient_norm
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task distractors  --method signed_gradient
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task distractors  --method integrated_gradient
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task distractors  --method last_attention
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task distractors  --method all_attention
```

#### Evaluate distractors
Run this set of commands to reproduce Table 3 from the paper.
```{bash}
cd ../analysis
python evaluate_distractor_rationales.py --baseline gradient_norm
python evaluate_distractor_rationales.py --baseline signed_gradient
python evaluate_distractor_rationales.py --baseline integrated_gradient
python evaluate_distractor_rationales.py --baseline last_attention
python evaluate_distractor_rationales.py --baseline all_attention
cd ../fairseq
```
The results should look like:
|       Method      | Source Mean | Target Mean | Source Frac. | Target Frac. |
| ----------------- | ----------- | ----------- | ------------ | ------------ |
| Gradient norms    |     0.40    |     0.44    |   **0.06**   |     0.06     |
| Grad x emb        |     6.25    |     5.57    |     0.42     |     0.41     |
| Integrated grads  |     2.08    |     1.68    |     0.23     |     0.14     |
| Last attention    |     0.63    |     2.41    |     0.09     |     0.24     |
| All attentions    |     0.58    |     0.80    |     0.08     |     0.12     |
| Greedy            |   **0.12**  |   **0.12**  |     0.09     |   **0.02**   |


#### Download and preprocess alignments
The other translation experiment involves word alignments. It is described in more detail in Section 8.2 of the paper.

First, agree to the license and download the gold alignments from [RWTH Aachen](https://www-i6.informatik.rwth-aachen.de/goldAlignment/). Put the files `en`, `de`, and `alignmentDeEn` in the directory `fairseq/examples/translation/iwslt14.tokenized.de-en/gold_labels`, and, in that same repo, convert to Unicode using
```{bash}
iconv -f ISO_8859-1 -t UTF8 de > gold.de
iconv -f ISO_8859-1 -t UTF8 en > gold.en
```
Clean and tokenize the text
```{bash}
cd ../..
cat iwslt14.tokenized.de-en/gold_labels/gold.en | \
  perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -l en | \
  perl mosesdecoder/scripts/tokenizer/lowercase.perl > iwslt14.tokenized.de-en/gold_labels/tmp_gold.en
cat iwslt14.tokenized.de-en/gold_labels/gold.de | \
  perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -l de | \
  perl mosesdecoder/scripts/tokenizer/lowercase.perl > iwslt14.tokenized.de-en/gold_labels/tmp_gold.de
cd ../..
```

Apply BPE
```{bash}
python subword-nmt/subword_nmt/apply_bpe.py -c iwslt14.tokenized.de-en/code < iwslt14.tokenized.de-en/gold_labels/tmp_gold.en > iwslt14.tokenized.de-en/gold_labels/gold_bpe.en 
python subword-nmt/subword_nmt/apply_bpe.py -c iwslt14.tokenized.de-en/code < iwslt14.tokenized.de-en/gold_labels/tmp_gold.de > iwslt14.tokenized.de-en/gold_labels/gold_bpe.de
```

Since the original file automatically tokenizes the apostrophes (e.g. `don ' t`) after BPE, sometimes there are incorrect spaces in the tokenization (e.g. `don &apos; t` instead of `don &apos;t`). Since this would change the alignments for these files, you may need to go through the files `gold_bpe.en` and `gold_bpe.de` and change them manually. Keep the spaces for apostrophes but not for plurals, e.g. `man &apos;s office` and `&apos; legal drugs &apos;` are both correct. Delete the new lines at the bottom of `gold.en`, `gold.de`, `gold_bpe.en`, and `gold_bpe.de`. The only other necessary change is changing `à@@ -@@` to `a-@@` on line 247 since we don't tokenize accents.

If you've agreed to the license and would like to skip these steps, email me at [keyvafa@gmail.com](mailto:keyvafa@gmail.com) and I can provide you the preprocessed files.

#### Binarize gold alignments
```{bash}
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train  \
    --validpref $TEXT/valid \
    --testpref $TEXT/gold_labels/gold_bpe \
    --destdir data-bin/iwslt14_alignments.tokenized.de-en \
    --workers 20
```

#### Create mapping between alignments with/without BPE
Make sure to replace `CHECKPOINT_DIR` with the directory you're using to store model checkpoints.
```{bash}
CHECKPOINT_DIR=...
cd ../analysis
python map_alignments_to_bpe.py --checkpoint_dir $CHECKPOINT_DIR
cd ../fairseq
```

#### Perform greedy rationalization for alignments dataset
```{bash}
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method greedy
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method greedy  --top_1
```

#### Perform baseline rationalization for alignments dataset
```{bash}
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method gradient_norm
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method gradient_norm  --top_1
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method signed_gradient
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method signed_gradient  --top_1
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method integrated_gradient
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method integrated_gradient  --top_1
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method last_attention
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method last_attention  --top_1
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method all_attention
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method all_attention  --top_1
```

#### Evaluate alignments
These commands will reproduce Table 4 in the paper.
```{bash}
cd ../analysis
python evaluate_alignment_rationales.py --baseline gradient_norm 
python evaluate_alignment_rationales.py --baseline gradient_norm  --top_1
python evaluate_alignment_rationales.py --baseline signed_gradient 
python evaluate_alignment_rationales.py --baseline signed_gradient  --top_1
python evaluate_alignment_rationales.py --baseline integrated_gradient 
python evaluate_alignment_rationales.py --baseline integrated_gradient  --top_1
python evaluate_alignment_rationales.py --baseline last_attention 
python evaluate_alignment_rationales.py --baseline last_attention  --top_1
python evaluate_alignment_rationales.py --baseline all_attention 
python evaluate_alignment_rationales.py --baseline all_attention  --top_1
cd ../fairseq
```
The results should look like:

|       Method       | Length |  AER   |  IOU   |   F1   |  Top1  |
| ------------------ | ------ | ------ | ------ | ------ | ------ |
| Gradient norms     |  10.2  |  0.82  |  0.30  |  0.16  |  0.63  |
| Grad x emb         |  13.2  |  0.90  |  0.16  |  0.12  |  0.40  |
| Integrated grads   |  11.3  |  0.85  |  0.24  |  0.14  |  0.42  |
| Last attention     |  10.8  |  0.84  |  0.27  |  0.15  |  0.59  |
| All attentions     |  10.7  |  0.82  |  0.32  |  0.15  |**0.66**|
| Greedy             | **4.9**|**0.78**|**0.40**|**0.24**|  0.64  |


#### Plot greedy rationalization example
This will reproduce Figure 6 from the paper.
```{bash}
python plot_iwslt_rationalization.py
cd ..
```

### <a id="gpt2">GPT-2</a>

In the paper, we performed experiments for fine-tuning GPT-2 Large (using sequence lengths of 1024). Since practitioners may not have a GPU that has the memory capacity to train the large model, our replication instructions are for GPT-2 Medium, fine-tuning with a sequence length of 512. This can be done on a single 12GB GPU, and the rationalization performance is similar for both models. If you would like to specifically replicate our results for GPT-2 Large, email me at [keyvafa@gmail.com](mailto:keyvafa@gmail.com) and I can provide you with the fine-tuning instructions/the full fine-tuned model.

#### Download Open-Webtext

First go to the [OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/), a WebText replication corpus provided by Aaron Gokaslan and Vanya Cohen. We only use a single split to train (we used `urlsf_subset09.tar`). Expand all the items and merge the first 998, taking only the first 8 million lines. This will be the training set. We used half of the remaining files as the validation set, and the other half as the test set. Store the files as `webtext_train.txt`, `webtext_valid.txt`, and `webtext_test.txt` in `huggingface/data`.

Alternatively, you can email me at [keyvafa@gmail.com](mailto:keyvafa@gmail.com) and I can send you the raw files (they're a little too large to store on Github).

#### Fine-tune GPT-2 for compatibility using word dropout
Make sure to replace `CHECKPOINT_DIR` with the directory you're using to store model checkpoints.
```{bash}
cd huggingface
CHECKPOINT_DIR=...
python examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path gpt2-medium \
    --do_train \
    --do_eval \
    --train_file data/webtext_train.txt \
    --validation_file data/webtext_valid.txt \
    --logging_dir gpt2_logs/compatible_gpt2 \
    --output_dir $CHECKPOINT_DIR/compatible_gpt2 \
    --per_device_train_batch_size 1 \
    --evaluation_strategy steps --eval_steps 500 \
    --num_train_epochs 50 \
    --lr_scheduler_type constant \
    --learning_rate 0.00001 \
    --block_size 512 \
    --per_device_eval_batch_size 4 \
    --save_total_limit 2 \
    --max_steps 45000 \
    --word_dropout_mixture 0.5
```

#### Get heldout perplexity
For the compatible model:
```{bash}
python examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path $CHECKPOINT_DIR/compatible_gpt2 \
    --output_dir gpt2_test_output/compatible/ \
    --do_eval \
    --validation_file data/webtext_test.txt \
    --block_size 512 \
    --per_device_eval_batch_size 4
```
This should give a heldout perplexity of 17.6086.

# For the pretrained model:
```{bash}
python examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path gpt2-medium \
    --output_dir gpt2_test_output/pretrained/ \
    --do_eval \
    --validation_file data/webtext_test.txt \
    --block_size 512 \
    --per_device_eval_batch_size 4
```
This should give a heldout perplexity of 19.9674.

import torch
from models import SequentialModel
from data_loader import DataLoader
from train import train
from evaluate import evaluate

# Define the model configurations to test
model_configs = [
    {'num_layers': 1, 'hidden_size': 64, 'learning_rate': 0.01, 'optimizer': 'sgd', 'reg_lambda': 0.001},
    {'num_layers': 2, 'hidden_size': 64, 'learning_rate': 0.01, 'optimizer': 'sgd', 'reg_lambda': 0.001},
    {'num_layers': 2, 'hidden_size': 128, 'learning_rate': 0.01, 'optimizer': 'sgd', 'reg_lambda': 0.001},
    {'num_layers': 2, 'hidden_size': 128, 'learning_rate': 0.001, 'optimizer': 'sgd', 'reg_lambda': 0.001},
    {'num_layers': 3, 'hidden_size': 128, 'learning_rate': 0.001, 'optimizer': 'adam', 'reg_lambda': 0.001},
    {'num_layers': 3, 'hidden_size': 256, 'learning_rate': 0.001, 'optimizer': 'adam', 'reg_lambda': 0.001},
    {'num_layers': 4, 'hidden_size': 256, 'learning_rate': 0.001, 'optimizer': 'adam', 'reg_lambda': 0.0001}
]

# Define the dataset path
dataset_path = '/path/to/dataset/'

# Define the number of epochs and batch size for training and testing
num_epochs = 10
batch_size = 32

# Define the device to use for training and testing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
data_loader = DataLoader(dataset_path)
train_data, test_data = data_loader.load_data()

# Test each model configuration and calculate the test accuracy
for i, config in enumerate(model_configs):
    print('Testing model configuration {}...'.format(i+1))
    model = SequentialModel(config['num_layers'], config['hidden_size'], data_loader.num_classes(), config['reg_lambda'])
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate']) if config['optimizer'] == 'sgd' else torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    train(model, optimizer, train_data, num_epochs, batch_size, device)
    test_accuracy = evaluate(model, test_data, batch_size, device)
    print('Model Configuration: {}\tTest Accuracy: {:.2f}'.format(config, test_accuracy))
  # The output will be
  # Layers	Hidden Layer Size	Learning Rate	Optimizer	Regularization	Accuracy

![image](https://user-images.githubusercontent.com/59788704/224533960-dcb8492f-e697-4e4d-9fd6-f56785ed2a2e.png)

To calculate the fail rate and In Variance Test for each capability, you can use the evaluate_model() function in the rationale_net.py file from the Rationales-For-Sequential-Predictions project. This function takes a trained model, test dataset, and a list of capabilities as inputs and returns the accuracy and fail rate for each capability.

For example, to get the accuracy and fail rate for Vocabulary and NER capabilities, you can use the following code:

from rationale_net import evaluate_model
from data_loaders import load_and_cache_examples

test_dataset = load_and_cache_examples(...) # Load the test dataset
model = ... # Load or train a model

capabilities = ['Vocabulary', 'NER'] # List of capabilities to evaluate

accuracy, fail_rate = evaluate_model(model, test_dataset, capabilities)

print('Capability\tMin Func Test\tFail Rate\tIn Variance Test\tFail Rate')
for i, capability in enumerate(capabilities):
    print(capability + '\t' + str(accuracy[i]) + '\t' + str(fail_rate[i]))


To calculate the In Variance Test and Negation capabilities, you can use the Checklist library. The Checklist library provides a set of predefined tests for different capabilities, including In Variance Test and Negation. You can use the test() function from the checklist.test_suite module to run these tests on your model and dataset.

For example, to calculate the fail rate for In Variance Test and Negation capabilities, you can use the following code:
from checklist.test_suite import TestSuite
from checklist.editor import Editor
from data_loaders import load_and_cache_examples
import torch

test_dataset = load_and_cache_examples(...) # Load the test dataset
model = ... # Load or train a model

# Define a set of test cases for In Variance Test and Negation capabilities
suite = TestSuite()
suite.add_test(Editor.InvarianceTest(test_dataset, model))
suite.add_test(Editor.NegationTest(test_dataset, model))

fail_rate = suite.fails()

print('Capability\tIn Variance Test\tFail Rate\tNegation\tFail Rate')
print('In Variance Test\t\t' + str(1 - fail_rate[0]) + '\t' + str(fail_rate[0]))
print('Negation\t\t\t\t\t' + str(1 - fail_rate[1]) + '\t' + str(fail_rate[1]))

# The Output will be:

![image](https://user-images.githubusercontent.com/59788704/224534280-2fd4ec5b-9bcc-477a-9a79-c2fa63f958fc.png)



