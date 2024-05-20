# %%
import pandas as pd
import numpy as np

# %%
# loading both the csv files
credit = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')


## merging and selecting useful features for my project
movies = movies.merge(how = 'inner' , left_on='title' , right_on= 'title' , right= credit)
removing_col = ['budget' , 'homepage' , 'original_language' , 'original_title' , 'popularity' , 'production_companies' , 'production_countries' , 'revenue' , 'runtime' , 'spoken_languages' , 'tagline' , 'vote_average' , 'release_date' , 'status' , 'vote_count']
movies = movies.drop(removing_col , axis = 1)

# %%
movies.dropna(inplace= True)
movies.sample(5)

# %%
print(movies.shape)
credit.shape

# %% [markdown]
# ## extracting the tags from the important columns

# %%
def mine(obj):
    '''this function is to mine the necessary stuff from the list of dictionaries in some important columns of the dataset'''
    obj = eval(obj)
    core = []
    for i in range(len(obj)):
        core.append(obj[i]['name'])
        
    return core

def mine_2(obj):
    '''for cast column'''
    obj = eval(obj)
    core_2 = []
    try:
        for i in range(3):
            core_2.append(obj[i]['name'])
            
    except:
        try:
            for i in range(2):
                core_2.append(obj[i]['name'])
        except:
            try:
                for i in range(1):
                    core_2.append(obj[i]['name'])
            except:
                pass
        
    return core_2

def mine_3(obj):
    '''for crew column as i want to get the name of only with job vlaue director'''
    obj = eval(obj)
    core_3 = []
    a = 0
    try:
        while True:
            if obj[a]['job'] == 'Director':
                core_3.append(obj[a]['name'])
            a +=1
    except:
        pass
    
    return core_3

movies['genres'] = movies['genres'].apply(mine)
movies['keywords'] = movies['keywords'].apply(mine)
movies['cast'] = movies['cast'].apply(mine_2)
movies['crew'] = movies['crew'].apply(mine_3)
movies['overview'] = movies['overview'].apply(lambda x : x.split())
movies['genres'] = movies['genres'].apply(lambda x: [txt.replace(" ",'') for txt in x])
movies['cast'] = movies['cast'].apply(lambda x: [txt.replace(" ",'') for txt in x])
movies['crew'] = movies['crew'].apply(lambda x: [txt.replace(" ",'') for txt in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [txt.replace(" ",'') for txt in x])
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# %% [markdown]
# ## getting and working with the new dataframe 'df'

# %%
new_df = movies.loc[:,['movie_id' , 'title' , 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: (' '.join(x)).lower())

# %%
new_df.head(5)

# %% [markdown]
# ## feature extraction

# %%
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize , sent_tokenize
from nltk.stem import PorterStemmer

# %%
ps = PorterStemmer()
def stemming(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
        
    return ' '.join(y)

new_df['tags'] = new_df['tags'].apply(stemming)

cv = CountVectorizer(stop_words= 'english' , max_features= 4000)
vectors = cv.fit_transform(new_df['tags']).toarray()

# %%
from sklearn.metrics.pairwise import cosine_similarity
cos_theta = cosine_similarity(vectors)

# %%
def recommend(movie):
    index = (new_df[new_df['title'] == movie].index)
    type(index)
    similar_5_index = list(np.argsort(cos_theta[index]))[0][::-1][1:6]
    rec_movies = [new_df['title'][ind] for ind in similar_5_index]
    return rec_movies

recommend('Terminator 3')

# %%


# %%


# %%


# %%



