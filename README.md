# Word2Vec Negative_Sampling

Note: This implemented on Windows OS, please find all path strings and change \\ with / if running on linux or Mac

### Dependecies:
1- Tensorflow
2- Python 3
3- Numpy
4- os
5- argpase
6- glob


This is an efficient implementation of Word2vec on game of thrones textbooks

### Train From Scratch
if you would like to run the model your self and configure the hyper-parameters specified in main.py please do delete the following folders first to avoid conflicts when running tensorflow:
1- visualizations
2- graph
3- checkpoints

To train from scratch run and of course feel free to provide your own hyper-parameters, you can find them in main.py:
```
python main.py --data-dir data\\ --vocab-dir vocab\\
```

### Just-Visualization
you can use my trained model and run tensorboard to visualize the word vectors generated to do so:

```
1- open terminal (cmd on windows) 
2- Navigate to visualizations folder
3- run
tensorboard --logdir=visualizations`
4- copy and paste the url provided by tensorboard
5- load the vocab_3000.tsv in tensorboard to identify each word

```

### Evaluation
you may as well run evaluate.py to find analogies and nearest words regarding game of thrones
my favourite one is <br />
**Mother is to Joffrey as "ghost/Sam" is to Jon**

