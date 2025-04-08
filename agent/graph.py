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

from agent.types import State
from agent.tools import (
    verify_customer_info,
    recommend_songs_by_genre, 
    check_for_songs, 
    check_upsell_eligibility, 
    get_recommended_upsells, 
    finalize_upsell_decision, 
    create_invoice, 
    make_handoff_tool
)

# We will set streaming=True so that we can stream tokens
# See the streaming section for more information on this.
model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo-preview")

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
