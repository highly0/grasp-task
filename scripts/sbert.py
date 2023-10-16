#!/usr/bin/env python
# coding: utf-8

# In[26]:


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[27]:


import pandas as pd


# ### Getting our Data

# In[28]:


train_path = '/workspace/grasp-data-hometask-semantic-similarity-master/data/train.data'
val_path = '/workspace/grasp-data-hometask-semantic-similarity-master/data/dev.data'


# In[29]:


col_names = ['Topic_Id', 'Topic_Name', 'Sent_1', 'Sent_2', 'Label', 'Sent_1_tag', 'Sent_2_tag']
train_df = pd.read_csv(train_path, sep='\t', lineterminator='\n', names=col_names, header=None)
train_df.head(3)


# In[30]:


val_df = pd.read_csv(val_path, sep='\t', lineterminator='\n', names=col_names, header=None)
val_df.head(3)


# ##### converting all label to binary

# In[32]:


def preproc(df):
    '''convert our label to 0-5'''
    df.loc[df['Label']== '(0, 5)', 'Label'] = 0
    df.loc[df['Label']== '(1, 4)', 'Label'] = 1
    df.loc[df['Label']== '(2, 3)', 'Label'] = 2
    df.loc[df['Label']== '(3, 2)', 'Label'] = 3
    df.loc[df['Label']== '(4, 1)', 'Label'] = 4
    df.loc[df['Label']== '(5, 0)', 'Label'] = 5
    return df

train_df = preproc(train_df)
val_df = preproc(val_df)


# In[33]:


train_df


# #### Preparing Data

# In[34]:


from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample, models


# In[35]:


def prepare_samples(df): 
    res = []
    for _, row in df.iterrows():
        score = float(row['Label']) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row['Sent_1'], row['Sent_2']], label=score)
        res.append(inp_example)
    return res


# In[36]:


train_samples = prepare_samples(train_df)
val_samples = prepare_samples(val_df)


# In[12]:


model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)


# In[13]:


train_loss = losses.CosineSimilarityLoss(model=model)

train_dataloader = torch.utils.data.DataLoader(train_samples, shuffle=True, batch_size=16)
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_samples)


# ####  Training the network

# In[16]:


import math

num_epochs = 10
# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up


# In[17]:


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path='./results_minilm_2')


# #### Inference

# In[14]:


model_save_path = './results_minilm_2'


# In[15]:


model = SentenceTransformer(model_save_path)


# In[16]:


sentences = ["All the home alones watching 8 mile", "8 mile is on thats my movie"]
encodings = model.encode(sentences)


# In[17]:


from sentence_transformers import SentenceTransformer, util
cosine_scores = util.cos_sim(encodings[0], encodings[1])
cosine_scores


# In[34]:


cosine_scores.item()


# In[18]:


test_path = '/workspace/grasp-data-hometask-semantic-similarity-master/data/test.data'


# In[23]:


results = []
with open(test_path) as tf:
    for tline in tf:
        lines = tline.split('\t')
        sentences = lines[2:4]
        encodings = model.encode(sentences)
        cosine_scores = util.cos_sim(encodings[0], encodings[1]).item()
        
        if cosine_scores >= 0.6:
            results.append("true\t" + "{0:.4f}".format(cosine_scores) + "\n")
        else: #if cosine_scores <= 0.4: 
            results.append("false\t" + "{0:.4f}".format(cosine_scores) + "\n")
        


# In[25]:


results


# In[24]:


res_path = '/workspace/grasp-data-hometask-semantic-similarity-master/systemoutputs/PIT2015_BASELINE_01_sbert_minilm.output'
with open(res_path, 'w+') as f:
    for line in results:
        f.write(line)


# In[ ]:




