from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import SKLearnVectorStore

from langchain_openai import OpenAIEmbeddings


db = SQLDatabase.from_uri("sqlite:///chinook.db")
print(db.get_usable_table_names())

artists = db._execute("select * from artists")
songs = db._execute("select * from tracks")
genres = db._execute("select * from genres")

artist_retriever = SKLearnVectorStore.from_texts(
    [a['Name'] for a in artists],
    OpenAIEmbeddings(), 
    metadatas=artists
).as_retriever()
song_retriever = SKLearnVectorStore.from_texts(
    [a['Name'] for a in songs],
    OpenAIEmbeddings(), 
    metadatas=songs
).as_retriever()
genre_retriever = SKLearnVectorStore.from_texts(
    [a['Name'] for a in genres],
    OpenAIEmbeddings(), 
    metadatas=genres
).as_retriever()