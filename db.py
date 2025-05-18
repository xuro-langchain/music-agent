from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import SKLearnVectorStore

from langchain_openai import OpenAIEmbeddings


db = SQLDatabase.from_uri("sqlite:///chinook.db")
print("Loading Database with columns: ")
print(db.get_usable_table_names())

print("\nLoading artists, songs, and genres")
artists = db._execute("select * from artists")
songs = db._execute("select * from tracks")
genres = db._execute("select * from genres")
print("Initializing retrievers...")

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
print("Done initializing retrievers\n")

print("Loading users eligible for upselling...")
upsell_users = db._execute("""
    WITH HighValueUsers AS (
        SELECT invoices.CustomerId, invoices.Total FROM invoices
        WHERE invoices.Total > 5
    ), 
    FirstTimers AS (
        SELECT invoices.CustomerId, 
        COUNT(invoices.InvoiceId) as purchase_count
        FROM invoices
        GROUP BY invoices.CustomerId
        HAVING purchase_count == 0
    )
    SELECT customers.CustomerId, customers.FirstName, customers.LastName, customers.PostalCode
    FROM customers
    WHERE customers.CustomerId IN (
        SELECT CustomerId
        FROM HighValueUsers
    ) OR customers.CustomerId IN (
        SELECT CustomerId
        FROM FirstTimers
    )
    LIMIT 3
    """)
print(upsell_users)

print("\nLoading users ineligible for upselling...")
non_upsell_users = db._execute("""
    WITH HighValueUsers AS (
        SELECT invoices.CustomerId, invoices.Total FROM invoices
        WHERE invoices.Total > 5
    ), 
    FrequentBuyers AS (
        SELECT invoices.CustomerId, 
        COUNT(invoices.InvoiceId) as purchase_count
        FROM invoices
        GROUP BY invoices.CustomerId
        HAVING purchase_count > 0
    )
    SELECT customers.CustomerId, customers.FirstName, customers.LastName, customers.PostalCode
    FROM customers
    WHERE customers.CustomerId NOT IN (
        SELECT CustomerId
        FROM HighValueUsers
    ) AND customers.CustomerId IN (
        SELECT CustomerId
        FROM FrequentBuyers
    )
    LIMIT 3
    """)
print(non_upsell_users)
print("\nAll data loaded. Starting graph...")
