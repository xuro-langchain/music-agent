from typing import Annotated
from datetime import datetime
from agent.types import State
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages import ToolMessage

from agent.db import db, song_retriever, genre_retriever

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
                name="verify_customer_info"
            )
            return Command(goto='customer', graph=Command.PARENT, update={"messages": [tool_message]})
            
        first, last, zip = parts
        columns = [column.strip("'") for column in customer.split(", ")]
        stored_first = columns[1]
        stored_last = columns[2]
        stored_zip = columns[8]
        
        if first == stored_first and last == stored_last and zip == stored_zip:
            tool_message = ToolMessage(
                content="Successfully verified customer information",
                tool_call_id=tool_call_id,
                name="verify_customer_info"
            )
            return Command(
                goto='customer',
                update={
                    "messages": [tool_message],
                    "customer_id": customer_id
                },
            )
    tool_message = ToolMessage(
        content="Failed to verify customer information",
        tool_call_id=tool_call_id,
        name="verify_customer_info"
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
def create_invoice(songs: list[str], customer_id: int):
    """Create a new invoice for the given songs and customer.
    Args:
        songs: List of song names to purchase
        customer_id: ID of the customer making the purchase
    """
    if not customer_id:
        return "Error: Customer ID is required to create an invoice"
    if not songs:
        return "Error: No songs provided for invoice"
    
    # Find track IDs for the songs
    track_ids = []
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
            
            if track:
                track_ids.append((track[0]['TrackId'], track[0]['UnitPrice']))
    
    if not track_ids:
        return "Error: No matching tracks found for the provided songs"
    
    try:
        # Create new invoice
        invoice_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        invoice_query = """
            INSERT INTO invoices (CustomerId, InvoiceDate, Total)
            VALUES ({}, '{}', {})
            RETURNING InvoiceId;
        """.format(customer_id, invoice_date, sum(price for _, price in track_ids))
        invoice_id = db.run(invoice_query)[0][0]
        
        # Add invoice lines
        for track_id, unit_price in track_ids:
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
        
        return {
            "message": "Invoice created successfully",
            "invoice_id": invoice_id,
            "date": details[0]['InvoiceDate'],
            "items": [{"song": d['SongName'], "price": d['UnitPrice']} for d in details],
            "total": sum(d['UnitPrice'] for d in details)
        }
        
    except Exception as e:
        return f"Error creating invoice: {str(e)}"

@tool
def refund_invoice(invoice_id: int, customer_id: int):
    """Refund an existing invoice for a customer.
    Args:
        invoice_id: ID of the invoice to refund
        customer_id: ID of the customer requesting the refund
    """
    if not invoice_id or not customer_id:
        return "Error: Both invoice ID and customer ID are required"
    
    # Verify invoice belongs to customer
    verify_query = """
        SELECT InvoiceId, Total, InvoiceDate
        FROM invoices
        WHERE InvoiceId = {} AND CustomerId = {};
    """.format(invoice_id, customer_id)
    
    invoice = db.run(verify_query, include_columns=True)
    if not invoice:
        return "Error: No matching invoice found for this customer"
    
    try:
        # Create refund invoice (negative amount)
        refund_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        refund_query = """
            INSERT INTO invoices (CustomerId, InvoiceDate, Total)
            VALUES ({}, '{}', {})
            RETURNING InvoiceId;
        """.format(customer_id, refund_date, -invoice[0]['Total'])
        refund_id = db.run(refund_query)[0][0]
        
        # Copy invoice items with negative amounts
        items_query = """
            INSERT INTO invoice_items (InvoiceId, TrackId, UnitPrice, Quantity)
            SELECT {}, TrackId, -UnitPrice, Quantity
            FROM invoice_items
            WHERE InvoiceId = {};
        """.format(refund_id, invoice_id)
        db.run(items_query)
        
        return {
            "message": "Refund processed successfully",
            "original_invoice": invoice_id,
            "refund_invoice": refund_id,
            "amount": -invoice[0]['Total'],
            "date": refund_date
        }
        
    except Exception as e:
        return f"Error processing refund: {str(e)}"

@tool
def check_upsell_eligibility(customer_id: int):
    """Check if the customer meets the conditions to upsell:
    - Has made at least one purchase above $10, OR
    - Has only made one purchase total
    """
    if not customer_id:
        return "Error: Customer ID is required"
    
    query = """
        WITH customer_stats AS (
            SELECT COUNT(*) as total_purchases, MAX(Total) as highest_purchase
            FROM invoices
            WHERE CustomerId = {} AND Total > 0
        )
        SELECT 
            total_purchases, highest_purchase,
            CASE 
                WHEN total_purchases = 1 THEN true  -- First-time buyer
                WHEN highest_purchase >= 10 THEN true  -- Has made a $10+ purchase
                ELSE false
            END as should_upsell
        FROM customer_stats;
    """.format(customer_id)
        
    result = db.run(query, include_columns=True)
    if not result:
        return False
    return True

@tool
def get_recommended_upsells():
    """Get recommended upsells based on customer's purchase history."""
    return None
    

@tool
def finalize_upsell_decision():
    """Finalize whether or not to upsell the customer."""
    return None


def make_handoff_tool(*, agent_name: str):
    """Create a tool that can return handoff via a Command"""
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def handoff(
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Ask another agent for help."""
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=tool_name,
            tool_call_id=tool_call_id,
        )
        return Command(
            goto=agent_name,
            update={"messages": [tool_message]},
        )
    return handoff