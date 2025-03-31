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

sales_handoff = make_handoff_tool(agent_name="sales")
customer_handoff = make_handoff_tool(agent_name="customer")
invoice_handoff = make_handoff_tool(agent_name="invoice")
music_handoff = make_handoff_tool(agent_name="music")

# Helper to specify which node is returning a response
def add_name(message, name):
    _dict = message.model_dump()
    _dict["name"] = name
    return AIMessage(**_dict)


async def customer_node(state: State):
    customer_prompt = """Your job is to help as a customer service representative for a music store.
    
    You should interact politely with customers to try to figure out how you can help. You can help in a few ways:
    - Identifying the user: you MUST obtain and the customer's ID and verify their identity before helping them.
    - Buying or refunding: if a customer wants make a new invoice, or refund an invoice. Handoff to the invoice agent `invoice`
    - Recomending music: if a customer wants to find some music or information about music. Handoff to the music agent `music`

    If the user is asking about music and has verified their customer ID, send them to that route.
    If the user is asking about invoices and has verified their customer ID, send them to that route.
    Otherwise, respond."""
    
    formatted = [SystemMessage(content=customer_prompt)] + state["messages"]
    
    chain = model.bind_tools([
        verify_customer_info,
        music_handoff,
        invoice_handoff,
    ]) | partial(add_name, name="customer")
    response = await chain.ainvoke(formatted)
    return {"messages": [response]}


async def music_node(state: State):
    song_system_message = """Your job is to recommend songs. Requests may come directly from a customer, or on their behalf by another AI agent.
    
    You have tools available to help recommend songs. You can: 
    - Check if a song exists
    - Recommend songs based on genre

    You also have tools to route to other agents, which you should use for any tasks you can't accomplish with your tools:
    - `customer`: if you do not have the customer's ID, or the customer asks for ANYTHING not related to music recommendations.

    When recommending songs based on genre, sometimes the genre will not be found. In that case, the tools will return information \
    on simliar songs and artists. This is intentional, it IS NOT the tool messing up.
    """
    
    formatted = [SystemMessage(content=song_system_message)] + state["messages"]

    chain = model.bind_tools([
        recommend_songs_by_genre, 
        check_for_songs,
        customer_handoff,
    ]) | partial(add_name, name="music")
    response = await chain.ainvoke(formatted)
    return {"messages": [response]}


async def invoice_node(state: State):
    invoice_prompt = """Your job is to help a customer with anything related to invoices or billing. 
    
    You have tools available to help take actions on invoices. You can:
    - Make a new invoice, which represents selling one or more songs to a customer
    - Refund an invoice, which represents giving a refund to a customer
    
    You also have tools to route to other agents, which you should use for any tasks you can't accomplish with your tools:
    - `customer`: if you do not have the customer's ID, or if you are asked for something you don't know how to help with.
    """
    
    formatted = [SystemMessage(content=invoice_prompt)] + state["messages"]
    
    # Get response from model
    chain = model.bind_tools([
        create_invoice,
        sales_handoff,
        customer_handoff,
    ]) | partial(add_name, name="invoice")
    response = await chain.ainvoke(formatted)
    return {"messages": [response]}


async def sales_node(state: State):
    sales_prompt = """Your job is to help a customer with anything related to sales or upselling. 
    
    You have tools available to help take actions on sales. You can:
    - Check if a customer is eligible for upselling
    - Get recommended songs to upsell. This will handoff to the music agent.
    - Decide whether or not to upsell. This will route back to the invoice agent.
    
    IMPORTANT: If a customer is eligible for upselling, you MUST get recommended songs to upsell before you decide whether or not to upsell.
    IMPORTANT: You should only upsell to each customer ONCE per session. You should take into account the customer's behavior and the popularity of the recommended songs when deciding whether or not to upsell.
    IMPORTANT: NEVER mention the word "upsell" in your response - use language like "I think you might like this song" instead.
    """
    formatted = [SystemMessage(content=sales_prompt)] + state["messages"]
    
    # Get response from model
    chain = model.bind_tools([
        check_upsell_eligibility,
        get_recommended_upsells,
        finalize_upsell_decision,
    ]) | partial(add_name, name="sales")
    response = await chain.ainvoke(formatted)
    return {"messages": [response]}


# Define available tools
tools = [
    verify_customer_info,
    recommend_songs_by_genre, 
    check_for_songs, 
    check_upsell_eligibility,
    get_recommended_upsells,
    finalize_upsell_decision,
    create_invoice,
    music_handoff,
    invoice_handoff,
    customer_handoff,
    sales_handoff,
]
tool_node = ToolNode(tools=tools)


# -----------------------------------------------------------------------
# ROUTING HELPERS -------------------------------------------------------
# -----------------------------------------------------------------------

def _is_tool_call(msg):
    return hasattr(msg, "additional_kwargs") and 'tool_calls' in msg.additional_kwargs

def _get_last_ai_message(messages):
    for m in messages[::-1]:
        if isinstance(m, AIMessage):
            return m
    return None

def _get_internal_transfer_source(messages, agent_name):
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            return None

        if isinstance(m, AIMessage) and _is_tool_call(m):
            tool_calls = m.additional_kwargs["tool_calls"]
            for tool_call in tool_calls:
                if tool_call["function"]["name"] == f"transfer_to_{agent_name}":
                    return m.name
    return None


def tool_route(state: State):
    messages = state["messages"]
    last_ai_message = _get_last_ai_message(messages)
    last_message = messages[-1]
    if isinstance(last_message, ToolMessage) and last_message.name.startswith("transfer_to_"):
        return last_message.name[12:]
    if last_ai_message is None:
        return "customer"
    else:
        if last_ai_message.name == "music":
            transfer_node = _get_internal_transfer_source(messages, "music")
            if transfer_node != "customer" and transfer_node != "music":
                return transfer_node
        return last_ai_message.name

def general_route(state: State):
    messages = state["messages"]
    last_ai_message = _get_last_ai_message(messages)
    if last_ai_message is None:
        return "customer"
    if not _is_tool_call(last_ai_message):
        return END
    else:
        return "tools"

def sales_route(state: State):
    messages = state["messages"]
    last_ai_message = _get_last_ai_message(messages)
    if last_ai_message is None:
        return "customer"
    if not _is_tool_call(last_ai_message):
        return "sales"
    else:
        return "tools"

def entry(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        if not _is_tool_call(last_message):
            return "sales" if last_message.name == "sales" else END
        else:
            return "tools"
    last_m = _get_last_ai_message(messages)
    if last_m is None:
        return "customer"
    if last_m.name == "music":
        return "music"
    if last_m.name == "invoice":
        return "invoice"
    if last_m.name == "sales":
        return "sales"
    return "customer"

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
    workflow.add_node("tools", tool_node)
    workflow.add_node("sales", sales_node)

    workflow.add_conditional_edges("tools", tool_route, {"customer": "customer", "music": "music", "invoice": "invoice", "sales": "sales"})
    workflow.add_conditional_edges("customer", general_route, {"customer": "customer", "tools": "tools", END: END})
    workflow.add_conditional_edges("music", general_route, {"customer": "customer", "tools": "tools", END: END})
    workflow.add_conditional_edges("invoice", general_route, {"customer": "customer", "tools": "tools", END: END})
    workflow.add_conditional_edges("sales", sales_route, {"customer": "customer", "sales": "sales", "tools": "tools"})
    workflow.set_conditional_entry_point(entry, {"customer": "customer", "tools": "tools", "music": "music", "invoice": "invoice", "sales": "sales", END: END})
    return workflow.compile(checkpointer=memory)

# -----------------------------------------------------------------------
# RUN FUNCTIONS ---------------------------------------------------------
# -----------------------------------------------------------------------
def print_messages(response):
    if isinstance(response, Interrupt):
        message = response.value["query"]
        if message:
            print("AI:" + message)
    elif isinstance(response, dict) and "messages" in response:
        messages = response["messages"]
        for message in messages:
            if isinstance(message, AIMessage) and message.content:
                print(f"{message.name.upper()} AI: {message.content}")
            if isinstance(message, ToolMessage):
                print(f"Tool called: {message.name}")

async def run(graph: StateGraph):
    state: State = {
        "messages": [],
        "customer_id": None
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
            state["messages"].append(HumanMessage(content=user))
            turn_input = state
        try:
            # Stream responses
            async for output in graph.astream(turn_input, config):
                if END in output or START in output:
                    continue
                # Print any node outputs
                for key, value in output.items():
                    print(f"Output from node '{key}':")
                    print("---")
                    # print(value)
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
