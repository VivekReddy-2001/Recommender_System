#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mlp 


# In[2]:


books=pd.read_csv('books.csv')
users=pd.read_csv('users.csv')
ratings=pd.read_csv('ratings.csv')


# In[3]:


books.head(5)


# In[4]:


users.head()


# In[5]:


ratings.head()


# In[6]:


#Finding shape of the datasets
data=[('books', books), ('users', users), ('ratings', ratings)]
for name,df in data:
    print(f'Dataset:{name}, Shape:{df.shape}')


# In[7]:


#Checking for missing values
print('No of values missing in books dataset are', books.isnull().sum())
print('No of values missing in books dataset are', users.isnull().sum())
print('No of values missing in books dataset are', ratings.isnull().sum())


# In[8]:


#Checking for duplicates
books.duplicated().sum()
users.duplicated().sum()
ratings.duplicated().sum()


# In[10]:


books_ratings=pd.merge(books,ratings,on='ISBN')


# In[12]:


books_ratings


# In[27]:


br=books_ratings.groupby('Book-Title')['Book-Rating'].count().sort_values(ascending=False)


# In[32]:


#There are 647924 records with no rating.
books_ratings[books_ratings['Book-Rating']==0]


# In[28]:


br.head(10)


# # Popularity Based Recommender System

# In[35]:


num_rating_df=books_ratings.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_rating'},inplace=True)


# In[37]:


num_rating_df


# In[38]:


avg_rating_df=books_ratings.groupby('Book-Title').mean()['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating':'avg_rating'},inplace=True)


# In[40]:


avg_rating_df


# In[42]:


popular_df=num_rating_df.merge(avg_rating_df,on='Book-Title')
popular_df


# In[47]:


popular_df=popular_df[popular_df['num_rating']>250].sort_values('avg_rating',ascending=False).head(50)


# In[52]:


popular_df=popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_rating','avg_rating']]


# In[55]:


#Top 50 famous books by avgerage rating and number of rating > 500
popular_df.reset_index(drop='index')


# # Collaborative Filtering Based Recommender System

# In[60]:


#Considering Users who have rated more than 200 books 
x=books_ratings.groupby('User-ID').count()['Book-Rating']>200
#getting index of users
rated_users=x[x].index


# In[62]:


filtered_rating=books_ratings[books_ratings['User-ID'].isin(rated_users)]


# In[66]:


#Considering Books with more than 50 ratings
y=filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books=y[y].index


# In[68]:


final_ratings=filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]


# In[71]:


#Checking for duplicates: There are no Duplicates
final_ratings.duplicated().sum()


# In[73]:


fr=final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')


# In[75]:


fr.fillna(0,inplace=True)


# In[76]:


fr


# In[81]:


#Calculating Similarity Scores
from sklearn.metrics.pairwise import cosine_similarity
similarity_score=cosine_similarity(fr)


# In[86]:


similarity_score.shape


# In[95]:


def recommend_books(book_name):
    index=np.where(fr.index==book_name)[0][0]
    similiar_items=sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:6]
    for i in similiar_items:
        print(fr.index[i[0]])


# In[89]:


np.where(fr.index=='Zoya')


# In[96]:


recommend_books('Message in a Bottle')


# In[ ]:




