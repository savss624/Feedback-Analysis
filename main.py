import pandas as pd
import pickle
import numpy as np
import math
from flask import Flask

app = Flask(__name__)

@app.route('/')
def root():
  df = pd.read_csv('ForumPostsWithEmbeds.csv')
  sample_posts = df.Message
  num_of_posts = sample_posts.shape[0]
  number_clusters = math.floor(math.sqrt(num_of_posts))
  doc_embeddings = df.to_numpy()[:,1:]

  with open('./SpectralClustering.pkl', 'rb') as f:
    clustering = pickle.load(f)

    clustering.set_params(n_clusters=number_clusters, 
                          assign_labels="discretize",
                          n_neighbors=number_clusters)
    clustering.fit(doc_embeddings)
    
  return [(number, count) for number, count in pd.Series(clustering.labels_).value_counts().to_dict().items()][0][0]
