Ancient History LM

Rik Koncel-Kedziorski and Noah A. Smith
https://arxiv.org/abs/2104.08742

Download selector model trained on Wikitext-2 at https://drive.google.com/file/d/1NVXrnrztxiS-hT128AhD5bVAgKfSgIN_/view?usp=sharing

Developed with:

pytorch 1.6
pytorch-lightning 1.1.2
transformers 4.2.2




Inference:

Place saved model in models/wiki_gpt_ranker.ckpt

Run 03-Inference.py 


Training on other data: 

01-CreatePPLTrainingData.py -- modify to create training data for the selector model. Currently uses Wikipedia and GPT2-large. 

02-TrainSelector.py -- trains model on data from step 01
