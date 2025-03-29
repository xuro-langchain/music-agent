import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from functools import partial

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_community.vectorstores import SKLearnVectorStore

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessageGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver


load_dotenv()

db = SQLDatabase.from_uri("sqlite:///chinook.db")
print(db.get_usable_table_names())


# We will set streaming=True so that we can stream tokens
# See the streaming section for more information on this.
model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo-preview")

artists = db._execute("select * from artists")
songs = db._execute("select * from tracks")
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

# Tools
@tool
def get_customer_info(customer_id: int):
    """Look up customer info given their ID. ALWAYS make sure you have the customer ID before invoking this."""
    return db.run(f"SELECT * FROM customers WHERE CustomerID = {customer_id};")

@tool
def get_albums_by_artist(artist):
    """Get albums by an artist (or similar artists)."""
    docs = artist_retriever.get_relevant_documents(artist)
    artist_ids = ", ".join([str(d.metadata['ArtistId']) for d in docs])
    return db.run(f"SELECT Title, Name FROM albums LEFT JOIN artists ON albums.ArtistId = artists.ArtistId WHERE albums.ArtistId in ({artist_ids});", include_columns=True)

@tool
def get_tracks_by_artist(artist):
    """Get songs by an artist (or similar artists)."""
    docs = artist_retriever.get_relevant_documents(artist)
    artist_ids = ", ".join([str(d.metadata['ArtistId']) for d in docs])
    return db.run(f"SELECT tracks.Name as SongName, artists.Name as ArtistName FROM albums LEFT JOIN artists ON albums.ArtistId = artists.ArtistId LEFT JOIN tracks ON tracks.AlbumId = albums.AlbumId WHERE albums.ArtistId in ({artist_ids});", include_columns=True)

@tool
def check_for_songs(song_title):
    """Check if a song exists by its name."""
    return song_retriever.get_relevant_documents(song_title)



customer_prompt = """Your job is to help a user update their profile.
You only have certain tools you can use. These tools require specific input. If you don't know the required input, then ask the user for it.
If you are unable to help the user, you can """

def get_customer_messages(messages):
    return [SystemMessage(content=customer_prompt)] + messages

customer_chain = get_customer_messages | model.bind_tools([get_customer_info])



song_system_message = """Your job is to help a customer find any songs they are looking for. 

You only have certain tools you can use. If a customer asks you to look something up that you don't know how, politely tell them what you can help with.

When looking up artists and songs, sometimes the artist/song will not be found. In that case, the tools will return information \
on simliar songs and artists. This is intentional, it is not the tool messing up."""
def get_song_messages(messages):
    return [SystemMessage(content=song_system_message)] + messages

song_recc_chain = get_song_messages | model.bind_tools([get_albums_by_artist, get_tracks_by_artist, check_for_songs])

msgs = [HumanMessage(content="hi! can you help me find songs by amy whinehouse?")]
song_recc_chain.invoke(msgs)


class Router(BaseModel):
    """Call this if you are able to route the user to the appropriate representative."""
    choice: str = Field(description="should be one of: music, customer")

system_message = """Your job is to help as a customer service representative for a music store.

You should interact politely with customers to try to figure out how you can help. You can help in a few ways:

- Updating user information: if a customer wants to update the information in the user database. Call the router with `customer`
- Recomending music: if a customer wants to find some music or information about music. Call the router with `music`

If the user is asking or wants to ask about updating or accessing their information, send them to that route.
If the user is asking or wants to ask about music, send them to that route.
Otherwise, respond."""
def get_messages(messages):
    return [SystemMessage(content=system_message)] + messages

chain = get_messages | model.bind_tools([Router])
msgs = [HumanMessage(content="hi! can you help me find a good song?")]
chain.invoke(msgs)
msgs = [HumanMessage(content="hi! whats the email you have for me?")]
chain.invoke(msgs)

def add_name(message, name):
    _dict = message.dict()
    _dict["name"] = name
    return AIMessage(**_dict)


def _get_last_ai_message(messages):
    for m in messages[::-1]:
        if isinstance(m, AIMessage):
            return m
    return None


def _is_tool_call(msg):
    return hasattr(msg, "additional_kwargs") and 'tool_calls' in msg.additional_kwargs


def _route(messages):
    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        if not _is_tool_call(last_message):
            return END
        else:
            if last_message.name == "general":
                tool_calls = last_message.additional_kwargs['tool_calls']
                if len(tool_calls) > 1:
                    raise ValueError
                tool_call = tool_calls[0]
                return json.loads(tool_call['function']['arguments'])['choice']
            else:
                return "tools"
    last_m = _get_last_ai_message(messages)
    if last_m is None:
        return "general"
    if last_m.name == "music":
        return "music"
    elif last_m.name == "customer":
        return "customer"
    else:
        return "general"

tools = [get_albums_by_artist, get_tracks_by_artist, check_for_songs, get_customer_info]
tool_node = ToolNode(tools=tools)

def _filter_out_routes(messages):
    ms = []
    for m in messages:
        if _is_tool_call(m):
            if m.name == "general":
                continue
        ms.append(m)
    return ms

general_node = _filter_out_routes | chain | partial(add_name, name="general")
music_node = _filter_out_routes | song_recc_chain | partial(add_name, name="music")
customer_node = _filter_out_routes | customer_chain | partial(add_name, name="customer")


def make_graph():
    graph = MessageGraph()
    nodes = {"general": "general", "music": "music", END: END, "tools": "tools", "customer": "customer"}
    # Define a new graph
    workflow = MessageGraph()
    workflow.add_node("general", general_node)
    workflow.add_node("music", music_node)
    workflow.add_node("customer", customer_node)
    workflow.add_node("tools", tool_node)
    workflow.add_conditional_edges("general", _route, nodes)
    workflow.add_conditional_edges("tools", _route, nodes)
    workflow.add_conditional_edges("music", _route, nodes)
    workflow.add_conditional_edges("customer", _route, nodes)
    workflow.set_conditional_entry_point(_route, nodes)
    graph = workflow.compile()
    return graph

memory = SqliteSaver.from_conn_string(":memory:")
graph = make_graph()

async def run():
    history = []
    while True:
        user = input('User (q/Q to quit): ')
        if user in {'q', 'Q'}:
            print('AI: Byebye')
            break
        history.append(HumanMessage(content=user))
        async for output in graph.astream(history):
            if END in output or START in output:
                continue
            # stream() yields dictionaries with output keyed by node name
            for key, value in output.items():
                print(f"Output from node '{key}':")
                print("---")
                print(value)
            print("\n---\n")
        history.append(AIMessage(content=value.content))
