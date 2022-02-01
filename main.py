import pandas as pd
import pickle
import numpy as np
import math
from flask import Flask, request, jsonify

app = Flask(__name__)

def get_post_info_by_cluster(number, 
                             data,
                             cluster):
  return(data[cluster.labels_ == number])

@app.route('/', methods=['POST'])
def root():
  data = request.data
  return jsonify(data)

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
    
  clusters = {}
  for i in range(number_clusters):
    clusters[f"Cluster {i}"] = tuple(get_post_info_by_cluster(i, 
                                   data = sample_posts,
                                   cluster = clustering))
    
  return clusters
  return pd.Series(clustering.labels_).value_counts().to_dict()
