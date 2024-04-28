# Solution description

Here I will explain details of my solution, what ideas did I implemented and their results. 
All implementations could be found in `notebooks` directory with code and markdown descriptions.

### Navigation
* [Main Idea](#main-idea)
* [Neural Network Approach](#neural-network-approach)
  * [Data Preparation](#data-preparation)
  * [Model Architecture](#model-architecture)
  * [Data transformations](#data-transformation-for-human-readable-format)
  * [Metrics](#metrics)
* [Catboost Approach](#catboost-approach)
  * [Data Preparation](#data-preparation-1)
  * [Model Architecture](#model-architecture-1)
  * [Data transformations](#data-transformation-for-human-readable-format-1)
  * [Metrics](#metrics-1)
* [Conclusion](#conclusions)
* [Credits (for grading)](#credits)


# Main idea

There different ways to implement Nested NER models, and for this assignment I decided to go with
approach of token-classification. The idea is to try to classify each token by several binary classifiers
and assign token classes where classifiers predicted 'positively'. In a nutshell, I converted problem from 
one big to 29 (number of classes in given dataset) binary classification task.

To classify tokens I decided to use bert-like models to get embeddings, and then train classifiers
on the embeddings. Ideally this solution should capture context and provide good base for classifiers.

This approach has its benefits:
1. We can have any deep of nested dependencies (not limited by 6 as in given dataset)
2. It is simpler to train classifiers
3. We can determine how each classifier performs individually and find interesting data insights

But also we can face with downsides:
1. We must use tokenizers, which may break label indexes
2. Data preprocessing becomes much harder due to alignment of tokens/labels
3. It would take much more time to train 29 models instead of just 1


# Neural Network approach

### Data Preparation

First of all let's discuss data preparation. In target, we have slices of data that represent some entity. 
If we are going to tokenize data, ideally we should make sure that tokenization will not change our 
boundaries. But it is hard to implement, especially with already existing tokenizers. 

To overcome this problem, I will suppose that target slices **do not include** subwords. 
With this assumption we can use `DeepPavlov/rubert-base-cased` tokenizer and many others if 
we do work tokenization first. However, some words would split into two or more, so for consistency 
we would consider all subword tokens that same entity as original word.

This solution is not the best, and some other models could be considered (such as char-tokenizers). 
However ideally it should not influence the predictions that much.


### Model architecture

When we have data preprocessed let's discuss the architecture.

This architecture I am going with is similar to one developed by [SinaLab](https://github.com/SinaLab/ArabicNER) 
for Arabic language. 

The main difference is that they had several classes and each class had different types 
(Like entity "Human" could be "Name", "Surname", "Age" e.t.c). So my solution is simplification of 
their architecture. However, it could be simply rewritten if required. 

Also, they trained their own bert model, however I am interested how raw bert (not even fine-tuned) 
would perform in such task. So in implementation below I freeze all bert-layers. 
This boosts training process of the model from 15 minutes per epoch, to just 30 seconds.

Now let's discuss classifiers a bit. In SinaLab they used two-layer NN (786->hid->1), but 
I found out that multi-layers with dropout works better. So you can adjust number of layers (at least 2) 
and dropout rate. The best architecture I found (and used) is 4-layer classifiers with 256 hidden dimension, 
and 0.1 dropout rate (AdamW optimizer, BCELoss, lr of 3e-5).


### Data transformation for human-readable format

When we have predictions, it makes sense to discuss how wo transform raw token prediction back to 
slices & labels format. 

This was one of the most confusing steps for me, as we are not working with full words but subwords tokenization. 

The first thing is data we are getting is just a list of lists where each `ans[i]` correspond to `i`'th token, 
and `ans[i]` contains indexes of classes given token corresponds to. With this information we could 
potentially get tokens start/end indexes in original sentence. This is what is implemented 
in `map_results_to_ids` function. Basically it takes `token_id`, via tokenizer get the string representation 
and finds string map in the original sentence. This implementation is pretty nice as it do not 
care about word tokenization (we don't need to consider skipped spaces/tabs/new-lines e.t.c.). 
Algorithm is modified a bit for `DeepPavlov/rubert-base-cased` as it strips special tokens that 
start with `#`. I tested this method on train/eval data, and it resulted in perfect map. 

The second step is to combine continuous entities. As an input we have (start, end, entities_list) 
and we want to combine tokens that we consider `continious` (e.g. `end[i] == start[i]`). For this 
we can use scanline algorithm implemented in `from_sequential_to_readable` function. 
Worth mentioning that we can define `error` of combination, where tokens on distance less than 
`error` could be considered as one, this adds variability to result where results and could boost 
performance if error is tuned as needed. 


### Metrics

Even though model will be measured by f1-macro, I wanted to see how model performs in general
(something like accuracy of some sort). As we are working with sets (both ranges and class type), 
the simplest thing would be to consider set's IOU. This metric is not interpretable and do not show 
insides where model makes mistakes, but it is nice to track how model performs "in general". 

In numeric values this approach results with:
1. `Mean IOU`: 37.65% (`std`: 9.29%)
2. `Mention F1`: 0.24% 
3. `Mention recall`: 0.24% 
4. `Mention precision`: 0.23% 
5. `Macro F1`: 0.24%

Not the best results, but considering the whole idea was developed and implemented by me, I am proud of IOU.


# Catboost approach

### Data preparation

Exactly the same as in [Neural Network approach](#data-preparation)


### Model architecture

Architecture is similar to [Neural Network approach](#model-architecture), but instead of NN classifiers 
I decided to use CatboostClassifiers. Ideally catboost should show the most out of ML models. My tests showed
that decision trees with `iterations=1000` resulted the best, but to train them make sure you are using `GPU` 
as otherwise it would take ages. 

I was not sure that catboost may be the best method here, as we can use KNN models with cosine similarity (which
makes more sense as we have embedding vectors). However, I figured out that it is very difficult to fit 29 KNN models due
to memory requirements (my CPU was struggling a lot), and even if I can fit them, inference took ages (1 sample was
estimated to be calculated in 9 hours). 

### Data transformation for human-readable format

Exactly the same as in [Neural Network approach](#data-transformation-for-human-readable-format), except 
with a bit different inference due to use of other approach.

### Metrics

Catboost was much harder to inference where even 1 sample required 20 minutes of data, and even it resulted
with poor performance. I would guess that the main problem is number of dimensions `768` and we are facing
with curse of dimensionality. Even though, all classifiers performed poorly, and it is obvious that this approach
is less efficient than Neural Network by all memory and time requirements.

In numeric values this approach results with:
1. `Mean IOU`: 0% (`std`: 0%)
2. `Mention F1`: 0% 
3. `Mention recall`: 0% 
4. `Mention precision`: 0% 
5. `Macro F1`: 0%


# Conclusions

The idea of token classification with 29 binary classifiers is not the way to go in Nested NER. It took 
unreasonable amount of time to preprocess/transform data and align labels and predictions, performance is low,
and training time is too high even on GPU. 

The [Neural Network approach](#neural-network-approach) performed much better than [Catboost](#catboost-approach) and
even [KNN](#model-architecture-1) on all parameters.

Best achieved metrics:
1. `Mean IOU`: 37.65% (`std`: 9.29%)
2. `Mention F1`: 0.24% 
3. `Mention recall`: 0.24% 
4. `Mention precision`: 0.23% 
5. `Macro F1`: 0.24%


# Credits
* Work done by Polina Zelenskaya (p.zelenskaya@innopolis.university)
* CodaLab nickname: cutefluffyfox
* Github Repository: https://github.com/cutefluffyfox/nested-named-entities/

