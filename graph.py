import asyncio
from operator import eq
import random
from functools import partial
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command, Interrupt, interrupt

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from typing import Annotated, Optional
from datetime import datetime
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages import ToolMessage
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings


# We will set streaming=True so that we can stream tokens
# See the streaming section for more information on this.
model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo-preview")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    customer_id: Optional[int]


# -----------------------------------------------------------------------
# DB SETUP  -------------------------------------------------------------
# -----------------------------------------------------------------------

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


# -----------------------------------------------------------------------
# TOOLS  ----------------------------------------------------------------
# -----------------------------------------------------------------------

@tool
def verify_customer_info(customer_id: int, tool_call_id: Annotated[str, InjectedToolCallId] ):
    """Verify customer info given a provided customer ID."""
    if customer_id:
        customer = db.run(f"SELECT * FROM customers WHERE CustomerID = {customer_id};")
        human_response = interrupt({"query": "Please enter your first name, last name, and zip code. Please use the format 'First Last Zip'"})       
        parts = human_response["data"].split()
        if len(parts) != 3:
            tool_message = ToolMessage(
                content="Invalid format. Please provide your first name, last name, and zip code separated by spaces.",
                tool_call_id=tool_call_id,
                name="verify_customer_info",
                artifact={"type": "transfer_to_customer", "customer_id": None},
            )
            return Command(goto='customer', update={"messages": [tool_message]})
            
        first, last, zip = parts
        columns = [column.strip("'") for column in customer.split(", ")]
        stored_first = columns[1]
        stored_last = columns[2]
        stored_zip = columns[8]
        
        if first == stored_first and last == stored_last and zip == stored_zip:
            tool_message = ToolMessage(
                content="Successfully verified customer information",
                tool_call_id=tool_call_id,
                name="verify_customer_info",
                artifact={"type": "transfer_to_customer", "customer_id": customer_id,},
            )
            return Command(
                goto='customer',
                update={
                    "messages": [tool_message],
                    "customer_id": customer_id,
                },
            )
    tool_message = ToolMessage(
        content="Failed to verify customer information",
        tool_call_id=tool_call_id,
        name="verify_customer_info",
        artifact={"type": "transfer_to_customer", "customer_id": None},
    )
    return Command(goto='customer', update={"messages": [tool_message], "customer_id": None})


@tool
def check_for_songs(song_title):
    """Check if a song exists by its name."""
    return song_retriever.invoke(song_title, k=1)


@tool
def recommend_songs_by_genre(genre, customer_id: int):
    """Get songs by a genre (or similar genres) that the customer hasn't already purchased."""
    if not customer_id:
        return "Error: Customer ID is required to provide personalized recommendations"
    
    docs = genre_retriever.invoke(genre, k=2)
    genre_ids = ", ".join([str(d.metadata['GenreId']) for d in docs])
    
    # Query to get tracks by genre that aren't in customer's purchase history
    query = """
        SELECT DISTINCT t.Name as SongName, ar.Name as ArtistName, al.Title as AlbumName, g.Name as GenreName
        FROM tracks t
        JOIN albums al ON t.AlbumId = al.AlbumId
        JOIN artists ar ON al.ArtistId = ar.ArtistId
        JOIN genres g ON t.GenreId = g.GenreId
        WHERE t.GenreId IN ({})
        AND NOT EXISTS (
            SELECT 1 
            FROM invoice_items ii
            JOIN invoices i ON ii.InvoiceId = i.InvoiceId
            WHERE ii.TrackId = t.TrackId
            AND i.CustomerId = {}
        )
        ORDER BY RANDOM()
        LIMIT 5;
    """.format(genre_ids, customer_id)
    
    return db.run(query, include_columns=True)

@tool
def create_invoice(songs: list[str], customer_id: int, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Create a new invoice for the given songs and customer.
    Args:
        songs: List of song names to purchase
        customer_id: ID of the customer making the purchase
    """
    if not customer_id:
        return "Transaction failed: Customer ID is required to create an invoice"
    if not songs:
        return "Transaction failed: No songs provided for invoice"
    
    # Find track IDs for the songs
    tracks = []
    for song in songs:
        # Get exact or similar matches for the song
        matches = song_retriever.invoke(song, k=1)
        if matches:
            # Get track info
            track_query = """
                SELECT TrackId, Name, UnitPrice 
                FROM tracks 
                WHERE TrackId = {};
            """.format(matches[0].metadata['TrackId'])
            track = db.run(track_query, include_columns=True)
            columns = track.split(", ")
            track_id = columns[0].split(": ")[1]
            track_name = columns[1].split(": ")[1].strip("'")
            price = columns[2].split(": ")[1].strip("}]")
            if track:
                tracks.append((track_id, track_name, price))
    
    if not tracks:
        return "Transaction failed: No matching tracks found for the provided songs"
    
    interrupt_message =  "You will be purchasing the following songs: {}".format([track[1] for track in tracks]) + \
    ". Your total is: {}".format(sum([float(track[2]) for track in tracks])) + \
    "\nPlease confirm by typing 'yes' or 'no'"
    human_response = interrupt(
        {"query": interrupt_message}
    )       
    response = human_response["data"]

    if response.lower() != "yes":
        return "Transaction failed: cancelled by customer"
    try:
        # Create new invoice
        invoice_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        invoice_query = """
            INSERT INTO invoices (CustomerId, InvoiceDate, Total)
            VALUES ({}, '{}', {})
            RETURNING InvoiceId;
        """.format(customer_id, invoice_date, sum(float(price) for _, _, price in tracks))
        invoice_id = db.run(invoice_query, include_columns=True)
        invoice_id = invoice_id.split(": ")[1].strip("}]")
        
        # Add invoice lines
        for track_id, _, unit_price in tracks:
            line_query = """
                INSERT INTO invoice_items (InvoiceId, TrackId, UnitPrice, Quantity)
                VALUES ({}, {}, {}, 1);
            """.format(invoice_id, track_id, unit_price)
            db.run(line_query)
        
        # Get invoice details for confirmation
        details_query = """
            SELECT t.Name as SongName, i.UnitPrice, inv.InvoiceDate
            FROM invoice_items i
            JOIN tracks t ON i.TrackId = t.TrackId
            JOIN invoices inv ON i.InvoiceId = inv.InvoiceId
            WHERE inv.InvoiceId = {};
        """.format(invoice_id)
        details = db.run(details_query, include_columns=True)
        rows = details.split("}, {")
        if (len(rows) == len(songs)):
            tool_message = ToolMessage(
                content="Transaction successfully completed!",
                tool_call_id=tool_call_id,
                name="create_invoice",
                artifact={"type": "transfer_to_sales", "customer_id": customer_id},
            )
            return Command(goto='sales', update={"messages": [tool_message]})
        return "Transaction failed: database error."
    except Exception as e:
        return f"Transaction failed: runtime error {str(e)}"

@tool
def check_upsell_eligibility(customer_id: int):
    """Check if the customer meets the conditions to upsell:
    - Has made at least one purchase above $5, OR
    - Has only made one purchase total
    """
    if not customer_id:
        return "Error: Customer ID is required"
    
    query = """
        SELECT COUNT(*) as total_purchases, MAX(Total) as highest_purchase
        FROM invoices
        WHERE CustomerId = {} AND Total > 0
    """.format(customer_id)
        
    result = db.run(query, include_columns=True)
    fragments = result.split(", ")
    total_purchases = fragments[0].split(": ")[1]
    highest_purchase = fragments[1].split(": ")[1].strip("}]")
    if int(total_purchases) != 1 and float(highest_purchase) < 5:
        return False
    return True

@tool
def get_recommended_upsells(customer_id: int, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Get recommended upsells based on customer's most recent invoice's most common genre.
    Args:
        customer_id: ID of the customer to get recommendations for
    Returns:
        The most common genre in the customer's most recent invoice
    """
    query = """
        WITH LastInvoice AS (
            SELECT InvoiceId
            FROM invoices
            WHERE CustomerId = {}
            ORDER BY InvoiceDate DESC
            LIMIT 1
        ),
        GenreCounts AS (
            SELECT g.Name as GenreName, COUNT(*) as GenreCount
            FROM LastInvoice li
            JOIN invoice_items ii ON li.InvoiceId = ii.InvoiceId
            JOIN tracks t ON ii.TrackId = t.TrackId
            JOIN genres g ON t.GenreId = g.GenreId
            GROUP BY g.GenreId, g.Name
            ORDER BY GenreCount DESC
            LIMIT 1
        )
        SELECT GenreName, GenreCount
        FROM GenreCounts;
    """.format(customer_id)
    
    result = db.run(query, include_columns=True)
    if result:
        # Result will be in format: "[{'GenreName': 'Rock', 'GenreCount': 3}]"
        genre = result.split("'")[3] if "'" in result else None
        
        tool_message = ToolMessage(
            content=f"Recommended genre for customer {customer_id}: {genre}. Handing off to music agent for recommendations",
            tool_call_id=tool_call_id,
            name="get_recommended_upsells",
            artifact={"type": "transfer_to_music", "genre": genre},
        )
        return Command(
            goto='music',
            update={ "messages": [tool_message]},
        )
    tool_message = ToolMessage(
        content=f"No recommended genres found for customer {customer_id}. Handing back to sales to reject upsell.",
        tool_call_id=tool_call_id,
        name="get_recommended_upsells",
        artifact={"type": "transfer_to_sales", "genre": None},
    )
    return Command(
        goto='sales',
        update={ "messages": [tool_message]},
    )

@tool
def finalize_upsell_decision(upsell: bool, song: str | None, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Finalize whether or not to upsell the customer. Song must be provided if upselling - if it is not, no upsell will be made"""
    message = "No, do not upsell"
    if upsell and song is not None:
        message = "Yes, offer to create an additional invoice for {}".format(song)
    
    tool_message = ToolMessage(
        content=message,
        tool_call_id=tool_call_id,
        name="finalize_upsell_decision",
        artifact={"type": "transfer_to_customer", "song": song},
    )
    return Command(
        goto='customer',
        update={ "messages": [tool_message]},
    )


def make_handoff_tool(*, agent_name: str):
    """Create a tool that can return handoff via a Command"""
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def handoff(
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Transfer customer to another agent."""
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=tool_name,
            tool_call_id=tool_call_id,
            artifact={"type": "transfer_to_" + agent_name},
        )
        return Command(
            goto=agent_name,
            update={"messages": [tool_message]},
        )
    return handoff

# -----------------------------------------------------------------------
# NODE DEFINITIONS ------------------------------------------------------
# -----------------------------------------------------------------------

invoice_handoff = make_handoff_tool(agent_name="invoice")
music_handoff = make_handoff_tool(agent_name="music")

# Helper to specify which node is returning a response
def add_name(message, name):
    _dict = message.model_dump()
    _dict["name"] = name
    return AIMessage(**_dict)

    
async def customer_node(state: State):
    if "customer_id" not in state or not state["customer_id"]:
        customer_prompt = """ 
        You are an AI Agent assisting a customer. You must request their ID before helping them.

        Once the customer has provided their ID, you should use the following tool to verify their identity:
        - Verifying the user: you MUST obtain the customers ID and verify their info before helping them.
        
        Otherwise, respond politely to the customer.
        """
    else:
        customer_prompt = """You are an AI Agent assisting a customer.
        Your job is to help as a customer service representative for a music store.
        
        You should interact politely with customers to figure out how you can help. 
        You have ONLY these tools available:
        - Transfer to invoice: transfers to the invoice agent, which will handle making new invoices.
        - Transfer to music: transfers to the music agent, which will handle music recommendations.

        DO NOT attempt to call any other tools that might be mentioned in the conversation history.

        IMPORTANT: You may not be able to fully complete the customer's request in one go. In such cases, you should break down the task step by step using your tools.
        
        When recommending songs based on genre, sometimes the genre will not be found. Instead, you will be returned information on similar songs and artists. This is intentional, it IS NOT a mistake.
        If you have finalized the decision to upsell, never use the word "upsell". Instead, offer to create another invoice to buy the new song.
        """
    
    formatted = [SystemMessage(content=customer_prompt)] + state["messages"]

    if "customer_id" not in state or not state["customer_id"]:
        chain = model.bind_tools([
            verify_customer_info,
        ], tool_choice="auto", parallel_tool_calls=False) | partial(add_name, name="customer")
    else:
        chain = model.bind_tools([
            music_handoff,
            invoice_handoff,
        ], tool_choice="auto", parallel_tool_calls=False) | partial(add_name, name="customer")
    response = await chain.ainvoke(formatted)

    return {"messages": [response]}


async def music_node(state: State):
    music_prompt = """
    You are an agent being directed by an AI supervisor.
    You have in memory the conversation history between the supervisor and a customer. 
    You should help the supervisor handle any requests related to music.
    
    You have tools available to help recommend songs. You can: 
    - Check if a song exists
    - Recommend songs based on genre

    IMPORTANT: If you do not have the customer's ID, reply that you need the customer's ID.
    IMPORTANT: Only take actions related to music. Ignore requests in the conversation history unrelated to music.
    """
    
    formatted = [SystemMessage(content=music_prompt)] + state["messages"]

    chain = model.bind_tools([
        recommend_songs_by_genre, 
        check_for_songs,
    ], tool_choice="any", parallel_tool_calls=False) | partial(add_name, name="music")
    response = await chain.ainvoke(formatted)
    return {"messages": [response]}


async def invoice_node(state: State):
    invoice_prompt = """
    You are an agent being directed by an AI supervisor.
    You have in memory the conversation history between the supervisor and a customer. 
    You should help the supervisor handle any requests related to invoices and purchases.
    
    You have tools available to help take actions on invoices. You can:
    - Make a new invoice, which represents selling one or more songs to a customer
    
    IMPORTANT: If you do not have the customer's ID, reply that you need the customer's ID.
    IMPORTANT: Only take actions related to invoices. Ignore requests in the conversation history unrelated to invoices.
    """
    
    formatted = [SystemMessage(content=invoice_prompt)] + state["messages"]
    
    # Get response from model
    chain = model.bind_tools([
        create_invoice,
    ], tool_choice="any", parallel_tool_calls=False) | partial(add_name, name="invoice")
    response = await chain.ainvoke(formatted)
    return {"messages": [response]}


async def sales_node(state: State):
    sales_prompt = """Your job is to determine whether to sell additional songs (upsell) to the customer after a purchase.
    
    You must call one of the following tools:
    - Check customer eligibility for upselling. 
    - Get recommended songs to upsell. This will handoff to the music agent.
    - Finalize decision on whether or not to upsell. This will handoff to the invoice agent.
    
    IMPORTANT: You should only upsell to each customer ONCE per session.
    """
    formatted = [SystemMessage(content=sales_prompt)] + state["messages"]
    
    # Get response from model
    chain = model.bind_tools([
        check_upsell_eligibility,
        get_recommended_upsells,
        finalize_upsell_decision,
    ], tool_choice="any", parallel_tool_calls=False) | partial(add_name, name="sales")
    response = await chain.ainvoke(formatted)
    return {"messages": [response]}


# Define available tools
customer_tools = [verify_customer_info, music_handoff, invoice_handoff]
customer_tool_node = ToolNode(tools=customer_tools)

music_tools = [recommend_songs_by_genre, check_for_songs]
music_tool_node = ToolNode(tools=music_tools)

invoice_tools = [create_invoice]
invoice_tool_node = ToolNode(tools=invoice_tools)

sales_tools = [check_upsell_eligibility, get_recommended_upsells, finalize_upsell_decision]
sales_tool_node = ToolNode(tools=sales_tools)


# -----------------------------------------------------------------------
# ROUTING HELPERS -------------------------------------------------------
# -----------------------------------------------------------------------

def _get_last_ai_message(messages):
    for m in messages[::-1]:
        if isinstance(m, AIMessage):
            return m
    return None

def _is_tool_call(msg):
    return hasattr(msg, "additional_kwargs") and 'tool_calls' in msg.additional_kwargs

def _is_internal_transfer(msg):
    return isinstance(msg, ToolMessage) and hasattr(msg, "artifact") and \
        msg.artifact and "type" in msg.artifact and \
        msg.artifact["type"].startswith("transfer_to")

def _get_internal_transfer_source(messages, agent_name):
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            return None
        if isinstance(m, AIMessage):
            if m.name != agent_name:
                return m.name
    return None

# -----------------------------------------------------------------------
# ROUTING FUNCTIONS -----------------------------------------------------
# -----------------------------------------------------------------------

def customer_route(state: State):
    messages = state["messages"]
    last_ai_message = _get_last_ai_message(messages)
    if last_ai_message is None:
        return "customer"
    if not _is_tool_call(last_ai_message):
        return END
    else:
        return "customer_tools"

def customer_tools_route(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if _is_internal_transfer(last_message):
        return last_message.artifact["type"][12:] # parse out transfer_to_ prefix
    return "customer"
    
def music_route(state: State):
    return "music_tools"

def music_tools_route(state: State):
    messages = state["messages"]
    transfer_node = _get_internal_transfer_source(messages, "music")
    if transfer_node and transfer_node != "customer":
        return transfer_node
    return "customer"

def invoice_route(state: State):
    return "invoice_tools"

def invoice_tools_route(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if _is_internal_transfer(last_message):
        return last_message.artifact["type"][12:] # parse out transfer_to_ prefix
    return "customer"

def sales_route(state: State):
    return "sales_tools"

def sales_tools_route(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if _is_internal_transfer(last_message):
        return last_message.artifact["type"][12:] # parse out transfer_to_ prefix
    return "sales"

# -----------------------------------------------------------------------
# GRAPH DEFINITION ------------------------------------------------------
# -----------------------------------------------------------------------

def make_graph(memory):
    """Create the graph."""
    workflow = StateGraph(State)

    # Add the nodes
    workflow.add_node("customer", customer_node)
    workflow.add_node("music", music_node)
    workflow.add_node("invoice", invoice_node)
    workflow.add_node("sales", sales_node)

    workflow.add_node("customer_tools", customer_tool_node)
    workflow.add_node("music_tools", music_tool_node)
    workflow.add_node("invoice_tools", invoice_tool_node)
    workflow.add_node("sales_tools", sales_tool_node)

    workflow.add_conditional_edges("customer", customer_route, 
        {"customer": "customer", "customer_tools": "customer_tools", END: END})
    workflow.add_conditional_edges("music", music_route, {"music_tools": "music_tools"})
    workflow.add_conditional_edges("invoice", invoice_route, {"invoice_tools": "invoice_tools"})
    workflow.add_conditional_edges("sales", sales_route, {"sales_tools": "sales_tools"})

    workflow.add_conditional_edges("customer_tools", customer_tools_route, {"customer": "customer",  "music": "music", "invoice": "invoice"})
    workflow.add_conditional_edges("music_tools", music_tools_route, {"sales": "sales", "customer": "customer"})
    workflow.add_conditional_edges("invoice_tools", invoice_tools_route, {"sales": "sales", "customer": "customer"})
    workflow.add_conditional_edges("sales_tools", sales_tools_route, {"sales": "sales", "music": "music", "customer": "customer"})
   
    workflow.set_entry_point("customer")
    return workflow.compile(checkpointer=memory)

# -----------------------------------------------------------------------
# RUN FUNCTIONS ---------------------------------------------------------
# -----------------------------------------------------------------------
def print_messages(response):
    if isinstance(response, tuple) and isinstance(response[0], Interrupt):
        message = response[0].value["query"]
        if message:
            print("AI: " + message)
    elif isinstance(response, dict) and "messages" in response:
        messages = response["messages"]
        for message in messages:
            if isinstance(message, AIMessage) and message.content:
                print(f"AI: {message.content}")
            if isinstance(message, ToolMessage):
                print(f"Tool called: {message.name}")

async def run(graph: StateGraph):
    state: State = {
        "messages": [],
    }

    thread_id = random.randint(0, 1000000)
    config = {
        "configurable": {
            "thread_id": str(thread_id),
            "checkpoint_ns": "music_store",
        }
    }
    interrupted = False
    while True:
        user = input('User (q to quit): ')
        if user in {'q', 'Q'}:
            print('AI: Goodbye!')
            break
        
        if interrupted:
            turn_input = Command(resume={"data": user})
            interrupted = False
        else:
            # Add user message to state
            state["messages"] = [HumanMessage(content=user)]
            turn_input = state
        try:
            # Stream responses
            async for output in graph.astream(turn_input, config, stream_mode="updates"):
                if END in output or START in output:
                    continue
                # Print any node outputs
                for key, value in output.items():
                    print_messages(value)

                    if key == "__interrupt__":
                        interrupted = True
        except Exception as e:
            print(f"Error: {str(e)}")
            raise e

async def main():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as memory:
        graph = make_graph(memory)
        await run(graph)

if __name__ == "__main__":
    asyncio.run(main())
