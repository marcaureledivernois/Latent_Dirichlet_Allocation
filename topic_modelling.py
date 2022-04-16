import pandas as pd
# import datefinder
# import numpy as np
# import win32com.client
# import codecs
# import os
# import re
#import itertools
import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from nltk.tokenize import MWETokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import os  # for os.path.basename
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import ward, dendrogram
import string
from nltk.tag import pos_tag
from gensim import corpora, models, similarities

stemmer = SnowballStemmer("english")



df = pd.read_pickle('df_save.pck')
df.head()

df = df[df['Language'].isin(['en'])]
# Combine title and contents
df['Text'] = df['Title'] + ' ' + df['Content']

df[df['Text'].str.contains('S.Korea')].head()

# Normalize words

replacements = {
    'Text': {
        r'\'s': '',
        'Indian': 'India',
        'nextgeneration': 'next generation',
        '//iconnect\.wto\.org/': '',
        '-': ' ',
        'U.S.': 'United States',
        'US': 'United States',
        'S.Korea': 'South Korea',
        'S. Korea': 'South Korea',
        'WTO': 'world trade organization',
        '‘': '',
        'imports': 'import',
        'Imports': 'import',
        'exports': 'export',
        'Exports': 'export',
        'NZ ': 'New Zealand ',
        '\"': '',
        '\'': '',
    }
}

df.replace(replacements, regex=True, inplace=True)

# Test
df[df['Text'].str.contains('S.Korea')].head()

texts = df['Text'].tolist()
titles = df['Title'].tolist()
dates = df['Date'].tolist()
articlecodes = df['ArticleCode'].tolist()

print(str(len(texts)) + ' texts')
print(str(len(titles)) + ' titles')
print(str(len(dates)) + ' dates')
print(str(len(articlecodes)) + ' articlecodes')

# load nltk's English stopwords as variable called 'stopwords'
my_stop_words = nltk.corpus.stopwords.words('english')

# Add my stopwords
my_stop_words = my_stop_words + ['world_trade_organization','years','year','said','important',
                                 'new','would','','','','']

# Stopwords of country names  NOT WORKING ????
#my_stop_words = my_stop_words + ['world_trade_organization','years','year','said','important','new','would','united_states',
#                                   'japan','india','obama','canada','mexico','russia','eu','european','china','chinese','would']


tokenizer = MWETokenizer([('world', 'bank'), ('world', 'trade', 'organization'), ('doha', 'round'),
                          ('united', 'states'), ('european', 'union'), ('new', 'zealand'),
                          ('per', 'cent'),('south', 'korea'),
                          ])

# Test the tokenizer
tokenizer.tokenize('In a little or a european union little bit world trade organization'.split())

# Test the function
#tokenize_and_stem('In World Bank or a_little. bit  _ World Trade Organization. United States')

# Use WordNetLemmatizer instead of stemmer
#from nltk.stem import WordNetLemmatizer
#lemmatizer = WordNetLemmatizer()
#df['Lemmatized'] = df['StopRemoved'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])

# load nltk's SnowballStemmer as variabled 'stemmer'

# ORIGINAL CODE
# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# Define a tokenizer and stemmer which returns the set of stems in the text that it is passed
def tokenize_and_stem(text):
    # Remove punctuation
    #    text = text.translate(str.maketrans('','',string.punctuation))

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    #    tokens = [word for sent in nltk.sent_tokenize(text) for word in tokenizer.tokenize(sent.lower().split())]

    # MWETokenizer: manually link words, when disabled, use n-gram range in TF-IDF
    #    tokens = tokenizer.tokenize(text.lower().split())
    tokens = tokenizer.tokenize(tokens)

    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if (re.search('[a-zA-Z]', token)):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    # WordNetLemmatizer
    #    stems = [lemmatizer.lemmatize(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # Remove punctuation
    #    text = text.translate(str.maketrans('','',string.punctuation))

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    #    tokens = [word for sent in nltk.sent_tokenize(text) for word in tokenizer.tokenize(sent.lower().split())]

    # MWETokenizer: manually link words, when disabled, use n-gram range in TF-IDF
    #    tokens = tokenizer.tokenize(text.lower().split())
    tokens = tokenizer.tokenize(tokens)

    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if (re.search('[a-zA-Z]', token)):
            filtered_tokens.append(token)
    stems = filtered_tokens
    #    stems = [stemmer.stem(t) for t in filtered_tokens]
    # WordNetLemmatizer
    #    stems = [lemmatizer.lemmatize(t) for t in filtered_tokens]
    return stems


totalvocab_stemmed = []
totalvocab_tokenized = []
for i in texts:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

vocab_frame.head(29)


tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

print(tfidf_matrix.shape)


tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
                                   min_df=0.1, stop_words=my_stop_words,
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()
print(terms)

dist = 1 - cosine_similarity(tfidf_matrix)


num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()


# joblib.dump(km,  'doc_cluster.pkl')
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

df_tfidf_matrix = pd.DataFrame(tfidf_matrix.toarray())
# df_tfidf_matrix.head()

news = {'date': dates,'articlecode': articlecodes, 'title': titles,'text': texts,'cluster': clusters}
frame = pd.DataFrame(news, index = [clusters], columns = ['date','articlecode','title','text','cluster'])

frame['cluster'].value_counts() #number of films per cluster (clusters from 0 to 4)

# Export to Access to analyze clusters
# frame.to_csv('newsclusters.txt', sep='^')


print("Top terms per cluster:")
print()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    for ind in order_centroids[i, :6]:
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()
    print()
#     print("Cluster %d titles:" % i, end='')
#     for title in frame.ix[i]['title'].values.tolist():
#         print(' %s,' % title, end='')
#     print()
#     print()

#export tables to HTML
print(frame[['Rank', 'Title']].loc[frame['cluster'] == 1].to_html(index=False))

MDS()

# two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]

#strip any proper nouns (NNP) or plural proper nouns (NNPS) from a text

def strip_proppers_POS(text):
    tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns

#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

#set up cluster names using a dict
cluster_names = {0: 'Family, home, war',
                 1: 'Police, killed, murders',
                 2: 'Father, New York, brothers',
                 3: 'Dance, singing, love',
                 4: 'Killed, soldiers, captain'}

# create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))

# group by cluster
groups = df.groupby('label')

# set up plot
fig, ax = plt.subplots(figsize=(17, 9))  # set size
ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

# iterate through groups to layer the plot
# note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params( \
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params( \
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelleft='off')

ax.legend(numpoints=1)  # show legend with only 1 point

# add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

plt.show()  # show the plot

# uncomment the below to save the plot if need be
# plt.savefig('clusters_small_noaxes.png', dpi=200)

#define custom toolbar location
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}


# create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))

# group by cluster
groups = df.groupby('label')

# define custom css to format the font and to remove the axis labeling
css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family:Arial, Helvetica, sans-serif;
}

g.mpld3-xaxis, g.mpld3-yaxis {
display: none; }
"""

# Plot
fig, ax = plt.subplots(figsize=(14, 6))  # set plot size
ax.margins(0.03)  # Optional, just adds 5% padding to the autoscaling

# iterate through groups to layer the plot
# note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, label=cluster_names[name], mec='none',
                     color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.title]

    # set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                             voffset=10, hoffset=10, css=css)
    # connect tooltip to fig
    mpld3.plugins.connect(fig, tooltip, TopToolbar())

    # set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    # set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

ax.legend(numpoints=1)  # show legend with only one dot

mpld3.display()  # show the plot

# uncomment the below to export to html
# html = mpld3.fig_to_html(fig)
# print(html)


linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters

#strip any proper names from a text...unfortunately right now this is yanking the first word from a sentence too.

def strip_proppers(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.islower()]
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()

#strip any proper nouns (NNP) or plural proper nouns (NNPS) from a text

def strip_proppers_POS(text):
    tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns

#Latent Dirichlet Allocation implementation with Gensim


#remove proper names
preprocess = [strip_proppers(doc) for doc in texts]

tokenized_text = [tokenize_and_stem(text) for text in preprocess]

texts = [[word for word in text if word not in my_stop_words] for text in tokenized_text]

#print(len([word for word in texts[0] if word not in stopwords]))
print(len(texts[0]))

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=1, no_above=0.8)
corpus = [dictionary.doc2bow(text) for text in texts]
len(corpus)

lda = models.LdaModel(corpus, num_topics=5, id2word=dictionary, update_every=5, chunksize=10000, passes=100)
lda = models.LdaModel(corpus, num_topics=5,
                            id2word=dictionary,
                            update_every=5,
                            chunksize=10000,
                            passes=100)

lda.show_topics()
print(lda[corpus[0]])

topics = lda.print_topics(5, num_words=20)

topics_matrix = lda.show_topics(formatted=False, num_words=20)
topics_matrix = np.array(topics_matrix)
topics_matrix.shape

topic_words = topics_matrix[:,:,1]
for i in topic_words:
    print([str(word) for word in i])
    print()