import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.
from smolagents import ToolCallingAgent, OpenAIServerModel, tool

dotenv.load_dotenv()  # reads .env from the project folder

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=os.getenv("UDACITY_OPENAI_API_KEY"),
)


# =====================================================================
# SHARED HELPERS (plain code, no LLM). Used by every agent's tools.
# =====================================================================
import json
import re
import difflib

REAM_SIZE = 500  # sheets per ream; prices are per sheet

# Exact catalog lookup: item_name -> {category, unit_price}
CATALOG = {item["item_name"]: item for item in paper_supplies}

# Common customer phrases that clearly mean one catalog item
PHRASE_SYNONYMS = {
    "printer paper": "Standard copy paper",
    "copy paper": "Standard copy paper",
    "8.5x11": "Letter-sized paper",
    '8.5"x11"': "Letter-sized paper",
    "letter size": "Letter-sized paper",
    "washi tape": "Decorative adhesive tape (washi tape)",
    "streamers": "Party streamers",
    "napkins": "Paper napkins",
    "name tags": "Name tags with lanyards",
}

# Paper sizes. A requested size that no catalog item has (e.g. A3) means we don't carry it.
SIZE_TOKENS = {"a3", "a4", "a5", "letter", "legal", "tabloid"}

# Words that carry no meaning for matching
STOPWORDS = {"of", "the", "a", "an", "and", "in", "with", "for", "sheet", "sheets", "paper",
             "various", "assorted", "colors", "color", "high", "quality", "size", "sized",
             "pack", "packs", "unit", "units", "roll", "rolls", "white"}

# In-memory stores shared across agents
QUOTE_STORE: Dict[str, Dict] = {}   # quote_id -> saved quote (sales only accepts these)
RUN_LOG: List[Dict] = []            # one record per request outcome (read by the business advisor)

# Line results for the request currently being processed. The orchestrator reads THIS list,
# not the agent's wording. Cleared at the start of each request.
CURRENT_LINES: List[Dict] = []

# The quote created for the request currently being processed (at most one per request).
CURRENT_QUOTE: Dict[str, Optional[str]] = {"quote_id": None, "sale_failed": False}

# Money spent on restocks while processing the current request (for row-by-row reconciliation)
CURRENT_SPEND: Dict[str, float] = {"restock": 0.0}

# The customer's ORIGINAL request text for the current request (agents may rephrase lines;
# code checks their wording against what the customer actually wrote).
CURRENT_REQUEST: Dict[str, str] = {"text": ""}


def _normalize_text(text_value: str) -> str:
    """Lowercase, drop thousands separators ("10,000" -> "10000"), collapse whitespace."""
    text_value = re.sub(r"(?<=\d),(?=\d)", "", str(text_value).lower())
    return re.sub(r"\s+", " ", text_value).strip()


GENERIC_WORDS = {"of", "the", "a", "an", "and", "in", "for", "with", "sheet", "sheets", "paper"}

# A quantity that starts an item line: a whole number followed by a word ("500 sheets",
# "200 balloons"). Not part of a size or date ("8.5x11", "24\" x 36\"", "April 15, 2025", "100%").
LINE_START = re.compile(r'(?<![\w.])(\d+)(?=\s+[A-Za-z])')       # not after a letter: "A4"
MONTH_DATE = re.compile(r"(january|february|march|april|may|june|july|august|september|october|"
                        r"november|december)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", re.I)


def parse_request_lines(request_text: str) -> List[Dict]:
    """Split the customer's request into its own item lines, in the customer's words.
    Each line runs from one quantity to the next (or to the end of the sentence)."""
    text_value = re.sub(r"(?<=\d),(?=\d)", "", str(request_text))       # "10,000" -> "10000"
    starts = list(LINE_START.finditer(text_value))
    lines = []
    for i, match in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text_value)
        segment = text_value[match.start():end]
        segment = re.split(r"\.\s|\n|\(date of request|\s+for (?:our|the|an?)\b", segment, flags=re.I)[0]
        segment = re.sub(r"[\s,;:-]*(\band\b|\balong with\b)?[\s,;:.-]*$", "", segment.strip(),
                         flags=re.I)
        lines.append({"text": segment, "qty": int(match.group(1))})
    return lines


def parse_deadline(request_text: str, default_deadline: str) -> str:
    """The delivery date written in the request ("April 15, 2025"), else the default."""
    match = MONTH_DATE.search(str(request_text))
    if not match:
        return default_deadline
    return datetime.strptime(f"{match.group(1)} {match.group(2)} {match.group(3)}",
                             "%B %d %Y").strftime("%Y-%m-%d")


def _stems4(text_value: str) -> set:
    """First four letters of each descriptive word ("rolls"/"roll", "colored"/"colors")."""
    words = re.findall(r"[a-z0-9]+", str(text_value).lower())
    return {w[:4] for w in words if not w.isdigit() and w not in GENERIC_WORDS}


def match_request_line(requested_as: str, quantity: float) -> Optional[int]:
    """Which of the customer's own lines this call is about: same quantity, best word overlap.
    None means the call matches no line the customer wrote."""
    segments = CURRENT_REQUEST.get("lines") or []
    candidates = [i for i, s in enumerate(segments) if s["qty"] == int(round(quantity))]
    if not candidates:
        return None
    overlap = {i: len(_stems4(requested_as) & _stems4(segments[i]["text"])) for i in candidates}
    best = max(candidates, key=lambda i: overlap[i])
    if len(candidates) > 1 and overlap[best] == 0:
        return None
    return best


def _word_stems(text_value: str) -> set:
    """First five letters of each descriptive word ("colored"/"colors" -> "color"). Uses its own
    short list of generic words: descriptive words like "colors" must count here."""
    words = re.findall(r"[a-z0-9]+", str(text_value).lower())
    return {w[:5] for w in words if not w.isdigit() and w not in GENERIC_WORDS}


def normalize_date(date_value: Union[str, datetime]) -> str:
    """Return a date as plain 'YYYY-MM-DD'.

    The database compares dates as text, so '2025-04-05T00:00:00' would sort AFTER
    '2025-04-05' and be silently excluded. Every date written or queried goes through here.
    """
    if isinstance(date_value, datetime):
        return date_value.strftime("%Y-%m-%d")
    text_value = str(date_value).strip()[:10]
    datetime.strptime(text_value, "%Y-%m-%d")  # raises ValueError if not a real date
    return text_value


def to_units(quantity: float, unit_word: str = "") -> int:
    """Convert a requested quantity to sellable units. Reams become sheets (x500)."""
    if "ream" in unit_word.lower():
        return int(quantity) * REAM_SIZE
    return int(quantity)


def stocked_items() -> Dict[str, Dict]:
    """The items the company carries (the inventory table), with price and reorder level."""
    rows = pd.read_sql("SELECT item_name, unit_price, min_stock_level FROM inventory", db_engine)
    return {r["item_name"]: {"unit_price": float(r["unit_price"]), "min_stock_level": int(r["min_stock_level"])}
            for _, r in rows.iterrows()}


def available_units(item_name: str, as_of_date: str) -> int:
    """Units that can be promised on as_of_date without stock going negative later.

    Takes the LOWEST projected stock level from this date onward, not just today's level,
    so stock already promised to any later-dated transaction is never sold twice.
    """
    as_of = normalize_date(as_of_date)
    later_dates = pd.read_sql(
        "SELECT DISTINCT substr(transaction_date, 1, 10) AS d FROM transactions "
        "WHERE item_name = :item AND substr(transaction_date, 1, 10) > :d",
        db_engine, params={"item": item_name, "d": as_of})["d"].tolist()
    levels = [int(get_stock_level(item_name, d)["current_stock"].iloc[0]) for d in [as_of] + later_dates]
    return min(levels)


def _meaningful_tokens(text_value: str) -> set:
    """Lowercase words and numbers, minus filler words."""
    return {t for t in re.findall(r"[a-z0-9]+", text_value.lower()) if t not in STOPWORDS}


def match_catalog_item(description: str, top_n: int = 3) -> List[Dict]:
    """Match a customer's item description to the closest REAL catalog names.

    Returns up to top_n candidates, best first, each with a score (0-1) and whether the
    company currently stocks it. An agent can only choose from these, so it can never
    invent an item name. An empty list means nothing in the catalog is a reasonable match.
    """
    desc_lower = description.lower()
    desc_tokens = _meaningful_tokens(description)
    desc_words = re.findall(r"[a-z0-9]+", desc_lower)  # in order, for finding the head noun
    stocked = stocked_items()

    # 0. A size the catalog doesn't carry (e.g. "A3 matte paper") is not carried in any form
    catalog_tokens = set().union(*(_meaningful_tokens(name) for name in CATALOG))
    if (desc_tokens & SIZE_TOKENS) - catalog_tokens:
        return []

    scores = {}

    # 1. Known phrases are a strong signal ("printer paper" -> Standard copy paper)
    for phrase, item_name in PHRASE_SYNONYMS.items():
        if phrase in desc_lower:
            scores[item_name] = 1.0

    # 2. Word overlap plus overall text similarity for every catalog item
    for item_name in CATALOG:
        item_tokens = _meaningful_tokens(item_name)
        union = item_tokens | desc_tokens
        overlap = len(desc_tokens & item_tokens) / len(union) if union else 0
        similarity = difflib.SequenceMatcher(None, desc_lower, item_name.lower()).ratio()
        score = round(0.7 * overlap + 0.3 * similarity, 3)

        # 3. The item the customer NAMED outranks a generic phrase: every distinguishing word
        #    of the catalog name (ignoring details in parentheses) appears in the request
        #    ("A4 printer paper" names A4 paper). A product type ("glossy") outranks a size
        #    ("A4"), a more specific name ("large poster") outranks a shorter one, and the
        #    product named LAST is the one wanted: "kraft paper envelopes" are envelopes.
        #    A name in parentheses counts as an alternative name ("washi tape").
        aliases = [re.sub(r"\(.*?\)", "", item_name)] + re.findall(r"\((.*?)\)", item_name)
        for alias in aliases:
            name_tokens = _meaningful_tokens(alias)
            if name_tokens and name_tokens <= desc_tokens:
                last_position = max(i for i, w in enumerate(desc_words) if w in name_tokens)
                named_score = ((1.05 if name_tokens <= SIZE_TOKENS else 1.1)
                               + 0.02 * len(name_tokens)
                               + 0.03 * last_position / max(len(desc_words), 1)
                               + 0.001 * similarity)
                score = max(score, named_score)
        scores[item_name] = max(scores.get(item_name, 0), score)

    # Best first; on an exact tie prefer an item we stock
    ranked = sorted(scores.items(), key=lambda pair: (round(pair[1], 2), pair[0] in stocked), reverse=True)
    return [{"item_name": name, "score": round(min(score, 1.0), 3), "stocked": name in stocked,
             "unit_price": CATALOG[name]["unit_price"]}
            for name, score in ranked[:top_n] if score >= 0.35]


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent
@tool
def find_catalog_item(description: str) -> str:
    """
    Find the closest REAL catalog items for a customer's item description.

    Args:
        description: The item as the customer wrote it, e.g. "A4 glossy paper".

    Returns:
        JSON list of candidates (best first) with item_name, score, stocked, unit_price.
        An empty list means the catalog has nothing that reasonably matches.
    """
    return json.dumps(match_catalog_item(description))


@tool
def check_inventory_snapshot(as_of_date: str) -> str:
    """
    List every stocked item and its quantity on a given date.

    Args:
        as_of_date: Date in YYYY-MM-DD format.

    Returns:
        JSON object mapping item_name to units in stock.
    """
    snapshot = get_all_inventory(normalize_date(as_of_date))
    return json.dumps({name: int(qty) for name, qty in snapshot.items()})


@tool
def check_item_stock(item_name: str, as_of_date: str) -> str:
    """
    Get the stock level of one exact catalog item on a given date.

    Args:
        item_name: Exact catalog item name (use find_catalog_item first).
        as_of_date: Date in YYYY-MM-DD format.

    Returns:
        JSON with item_name and units_in_stock.
    """
    level = get_stock_level(item_name, normalize_date(as_of_date))
    return json.dumps({"item_name": item_name, "units_in_stock": int(level["current_stock"].iloc[0])})


def _record_line(requested_as: str, result: Dict, requested_units: int = 0) -> str:
    """Save one line's outcome in code and return it as JSON for the agent."""
    record = {"requested": requested_as, **result, "requested_units": requested_units}
    CURRENT_LINES.append(record)
    return json.dumps(record)


@tool
def order_stock(requested_as: str, item_name: str, quantity: float, unit: str,
                request_date: str, deadline_date: str) -> str:
    """
    Secure stock for ONE order line: confirm it is in stock, or restock from the supplier
    if cash allows and the delivery arrives by the deadline. Check first, then commit.
    Call this for EVERY requested line, including lines with no catalog match.

    Args:
        requested_as: The line exactly as the customer wrote it, e.g. "200 sheets of A4 glossy paper".
        item_name: Exact catalog item name from find_catalog_item, or "NONE" if nothing fits.
        quantity: Quantity the customer asked for.
        unit: Unit the customer used, e.g. "sheets" or "reams" (reams are converted to sheets).
        request_date: Date of the request, YYYY-MM-DD.
        deadline_date: Customer's required delivery date, YYYY-MM-DD.

    Returns:
        JSON with requested, item_name, status ("available", "restock_ordered", "declined"),
        units, ready_date, reason.
    """
    stocked = stocked_items()
    requested_units = to_units(quantity, unit)

    # The deadline is the one the CUSTOMER wrote, read by code, never the agent's.
    if CURRENT_REQUEST.get("deadline"):
        deadline_date = CURRENT_REQUEST["deadline"]

    # Every call must be one of the customer's OWN lines (split from the request by code).
    # A paraphrase maps back to the customer's line; a substitute the customer never asked
    # for ("5000 sheets of A4 paper" for "5,000 sheets of A3 paper") is the same line, and a
    # call matching no line is refused without being recorded.
    segment = None
    if CURRENT_REQUEST.get("lines"):
        segment = match_request_line(requested_as, quantity)
        if segment is None:
            return json.dumps({"requested": requested_as, "status": "rejected",
                               "reason": "not_a_line_in_the_customer_request",
                               "customer_lines": [s["text"] for s in CURRENT_REQUEST["lines"]]})
        requested_as = CURRENT_REQUEST["lines"][segment]["text"]      # the customer's words
    for existing in CURRENT_LINES:
        if existing["requested"] == requested_as or (
                segment is not None and existing.get("segment") == segment):
            return json.dumps(existing)

    # Code decides the item from the CUSTOMER's wording, not the agent's choice. If the wording
    # names a catalog item, that item is used ("A4 printer paper" -> A4 paper; "kraft paper
    # envelopes" -> Envelopes). A stocked item the matcher does not offer for this wording is
    # refused ("A3 glossy" is not Glossy paper, because we don't carry A3).
    matches = match_catalog_item(requested_as)
    if matches and matches[0]["score"] >= 1.0:
        item_name = matches[0]["item_name"]
    elif item_name not in {m["item_name"] for m in matches}:
        item_name = "NONE"

    # Idempotency: a line already handled is never processed (or restocked) twice.
    # Match on WHAT is ordered (catalog item + quantity), not on wording: agents rephrase.
    for existing in CURRENT_LINES:
        if (existing["item_name"] == item_name and item_name != "NONE"
                and existing.get("requested_units") == requested_units):
            return json.dumps(existing)

    # ---------- CHECK (nothing changes here) ----------
    if item_name not in stocked:
        return _record_line(requested_as, {"segment": segment, "item_name": item_name, "status": "declined",
                                           "reason": "not_carried"}, requested_units)
    units_needed = requested_units
    if units_needed <= 0:
        return _record_line(requested_as, {"segment": segment, "item_name": item_name, "status": "declined",
                                           "reason": "invalid_quantity"}, requested_units)
    req_date, deadline = normalize_date(request_date), normalize_date(deadline_date)

    on_hand = available_units(item_name, req_date)
    if on_hand >= units_needed:
        return _record_line(requested_as, {"segment": segment, "item_name": item_name, "status": "available",
                                           "units": units_needed, "ready_date": req_date,
                                           "reason": "in_stock"}, requested_units)

    # Short: order the shortfall plus the reorder buffer
    reorder_qty = (units_needed - on_hand) + stocked[item_name]["min_stock_level"]
    cost = round(reorder_qty * stocked[item_name]["unit_price"], 2)
    if cost > get_cash_balance(req_date):
        return _record_line(requested_as, {"segment": segment, "item_name": item_name, "status": "declined",
                                           "units": units_needed,
                                           "reason": "insufficient_funds_for_restock"}, requested_units)
    arrival = get_supplier_delivery_date(req_date, reorder_qty)
    if arrival > deadline:
        return _record_line(requested_as, {"segment": segment, "item_name": item_name, "status": "declined",
                                           "units": units_needed,
                                           "reason": f"restock_arrives_{arrival}_after_deadline_{deadline}"}, requested_units)

    # ---------- COMMIT (every check passed) ----------
    # Booked on the request date, so each request's cash effect appears in its own row of the
    # results; the supplier's arrival date still sets when the goods are ready to ship.
    create_transaction(item_name, "stock_orders", reorder_qty, cost, req_date)
    CURRENT_SPEND["restock"] = round(CURRENT_SPEND["restock"] + cost, 2)
    return _record_line(requested_as, {"segment": segment, "item_name": item_name, "status": "restock_ordered",
                                       "units": units_needed, "ready_date": arrival,
                                       "reason": "restocked_from_supplier"}, requested_units)


@tool
def replenish_low_stock(as_of_date: str) -> str:
    """
    Reorder every stocked item that has fallen below its minimum stock level.

    Args:
        as_of_date: Date in YYYY-MM-DD format.

    Returns:
        JSON list of reorders placed (item_name, units, arrival_date) and any skipped for cash.
    """
    as_of = normalize_date(as_of_date)
    snapshot = get_all_inventory(as_of)   # items at zero are missing here, so default to 0
    placed, skipped = [], []
    for item_name, info in stocked_items().items():
        on_hand = int(snapshot.get(item_name, 0))
        if on_hand >= info["min_stock_level"]:
            continue
        reorder_qty = info["min_stock_level"] * 2 - on_hand   # refill to twice the minimum
        cost = round(reorder_qty * info["unit_price"], 2)
        if cost > get_cash_balance(as_of):
            skipped.append(item_name)
            continue
        arrival = get_supplier_delivery_date(as_of, reorder_qty)
        create_transaction(item_name, "stock_orders", reorder_qty, cost, as_of)
        CURRENT_SPEND["restock"] = round(CURRENT_SPEND["restock"] + cost, 2)
        placed.append({"item_name": item_name, "units": reorder_qty, "arrival_date": arrival})
    return json.dumps({"reorders_placed": placed, "skipped_for_cash": skipped})


class InventoryAgent(ToolCallingAgent):
    """Worker agent: decides whether each order line can be supplied, and by when."""

    def __init__(self, model):
        super().__init__(
            tools=[find_catalog_item, check_inventory_snapshot, check_item_stock,
                   order_stock, replenish_low_stock],
            model=model,
            name="inventory_agent",
            description="""Inventory specialist. For EACH requested line:
            1. Call find_catalog_item. Choose ONE candidate: size words (A4, 8.5x11, letter) describe size,
               so match the paper type; if several fit, prefer a stocked one. Never substitute a different
               product. If nothing fits, use item_name "NONE".
            2. Call order_stock ONCE for the line, passing the line exactly as the customer wrote it
               as requested_as. Do not call check_item_stock first; order_stock checks stock itself.
            When every line is done, give a one-sentence summary. The system reads the line results
            directly from order_stock, so your summary does not need to repeat them.""",
        )


# Tools for quoting agent

# Bulk discount tiers: (minimum total units, discount rate). Checked from largest down.
DISCOUNT_TIERS = [(10000, 0.10), (1000, 0.05), (0, 0.0)]


def bulk_discount_rate(total_units: int) -> float:
    """Return the bulk discount rate for an order's total units."""
    for minimum_units, rate in DISCOUNT_TIERS:
        if total_units >= minimum_units:
            return rate
    return 0.0


@tool
def search_past_quotes(search_term: str) -> str:
    """
    Look up similar historical quotes for pricing precedent.

    Args:
        search_term: ONE short keyword, e.g. "cardstock" or "ceremony". The history search
            requires every term to match, so use a single term per call.

    Returns:
        JSON list of up to 3 past quotes with total_amount, order_size, event_type and explanation.
    """
    matches = search_quote_history([search_term.strip()], limit=3)
    return json.dumps([{"total_amount": m["total_amount"], "order_size": m["order_size"],
                        "event_type": m["event_type"], "explanation": m["quote_explanation"][:300]}
                       for m in matches])


def describe_discount(total_units: int, rate: float) -> str:
    """Customer-facing explanation of which bulk discount tier applied, and why."""
    if rate >= 0.10:
        return f"Your order totals {total_units:,} units, so our 10% bulk discount for orders of 10,000+ units applies."
    if rate >= 0.05:
        return f"Your order totals {total_units:,} units, so our 5% bulk discount for orders of 1,000-9,999 units applies."
    return (f"Your order totals {total_units:,} units; bulk discounts start at 1,000 units (5%), "
            f"so standard pricing applies.")


@tool
def calculate_quote(request_date: str) -> str:
    """
    Price every line the inventory agent confirmed (available or restock_ordered) for the
    current request, apply the bulk discount, and save the quote under a quote ID.
    Lines are read from the system's records, not passed in, so nothing can be mispriced.

    Args:
        request_date: Date of the request, YYYY-MM-DD.

    Returns:
        JSON with quote_id, priced lines, total_units, subtotal, discount_rate, discount_amount,
        total, discount_note and precedent_note.
    """
    # Idempotency: at most one quote per request; a repeat call returns the existing quote
    existing_id = CURRENT_QUOTE["quote_id"]
    if existing_id in QUOTE_STORE:
        return json.dumps(QUOTE_STORE[existing_id])

    fillable = [line for line in CURRENT_LINES if line["status"] in ("available", "restock_ordered")]
    if not fillable:
        return json.dumps({"quote_id": None, "message": "No lines can be supplied, so there is nothing to quote."})

    total_units = sum(line["units"] for line in fillable)
    rate = bulk_discount_rate(total_units)

    priced_lines = []
    for line in fillable:
        unit_price = CATALOG[line["item_name"]]["unit_price"]
        list_total = round(line["units"] * unit_price, 2)
        priced_lines.append({"item_name": line["item_name"], "units": line["units"],
                             "unit_price": unit_price, "list_total": list_total,
                             "net_total": round(list_total * (1 - rate), 2),
                             "ready_date": line["ready_date"]})

    # Pricing precedent from quote history (RAG), always checked in code
    keyword = fillable[0]["item_name"].split()[0].lower()
    past_quotes = search_quote_history([keyword], limit=5)
    precedent_note = (f"Our pricing is consistent with {len(past_quotes)} similar past orders."
                      if past_quotes else "")

    subtotal = round(sum(p["list_total"] for p in priced_lines), 2)
    total = round(sum(p["net_total"] for p in priced_lines), 2)
    quote_id = f"Q-{len(QUOTE_STORE) + 1:04d}"
    QUOTE_STORE[quote_id] = {"quote_id": quote_id, "request_date": normalize_date(request_date),
                             "lines": priced_lines, "total_units": total_units, "subtotal": subtotal,
                             "discount_rate": rate, "discount_amount": round(subtotal - total, 2),
                             "total": total, "discount_note": describe_discount(total_units, rate),
                             "precedent_note": precedent_note, "used": False}
    CURRENT_QUOTE["quote_id"] = quote_id
    return json.dumps(QUOTE_STORE[quote_id])


class QuotingAgent(ToolCallingAgent):
    """Worker agent: prices the confirmed lines and explains the pricing."""

    def __init__(self, model):
        super().__init__(
            tools=[search_past_quotes, calculate_quote],
            model=model,
            name="quoting_agent",
            description="""Quoting specialist.
            1. Call search_past_quotes once or twice, each time with ONE short keyword
               (the event type or the main item), to find pricing precedent.
            2. Call calculate_quote ONCE with the request date. It prices the confirmed lines itself.
            3. Give a short pricing explanation for the customer: the quote ID, the total, which bulk
               discount tier applied and why (under 1,000 units: none; 1,000-9,999: 5%; 10,000+: 10%),
               and, only if past quotes were found, that the pricing is consistent with similar past orders.
            Use the numbers exactly as calculate_quote returned them. Never invent or change a price.""",
        )


# Tools for ordering agent
@tool
def finalize_sale(quote_id: str, request_date: str) -> str:
    """
    Turn a saved quote into recorded sales. Check first, then commit.
    Only quotes created by calculate_quote are accepted, each one only once.

    Args:
        quote_id: The quote ID returned by calculate_quote, e.g. "Q-0001".
        request_date: Date of the request, YYYY-MM-DD.

    Returns:
        JSON with success, quote_id, total, and per-line delivery dates; or the reason it was refused.
    """
    # ---------- CHECK (nothing changes here) ----------
    quote = QUOTE_STORE.get(quote_id)
    if quote is None:
        return json.dumps({"success": False, "reason": "unknown_quote_id"})
    if quote["used"]:
        return json.dumps({"success": False, "reason": "quote_already_finalized"})

    # Re-check stock on the request date (stock may have changed since the quote)
    sale_date = quote["request_date"]
    units_by_item: Dict[str, int] = {}
    for line in quote["lines"]:
        units_by_item[line["item_name"]] = units_by_item.get(line["item_name"], 0) + line["units"]
    shortages = []
    for item_name, units in units_by_item.items():
        if available_units(item_name, sale_date) < units:
            shortages.append(item_name)
    if shortages:
        if quote_id == CURRENT_QUOTE["quote_id"]:
            CURRENT_QUOTE["sale_failed"] = True
        return json.dumps({"success": False, "reason": "stock_changed_since_quote", "items": shortages})

    # ---------- COMMIT (every check passed) ----------
    delivered = []
    for line in quote["lines"]:
        # Booked on the request date so the sale appears in this request's row; ships on ready_date
        create_transaction(line["item_name"], "sales", line["units"], line["net_total"], sale_date)
        delivered.append({"item_name": line["item_name"], "units": line["units"],
                          "delivery_date": line["ready_date"]})
    quote["used"] = True
    return json.dumps({"success": True, "quote_id": quote_id, "total": quote["total"],
                       "lines": delivered})


class SalesAgent(ToolCallingAgent):
    """Worker agent: finalizes a quoted order by recording the sale."""

    def __init__(self, model):
        super().__init__(
            tools=[finalize_sale],
            model=model,
            name="sales_agent",
            description="""Sales specialist. You receive a quote ID and the request date.
            Call finalize_sale ONCE with exactly that quote ID and date. Never invent a quote ID;
            if none was given, do not call the tool. Report the result exactly as returned:
            on success, the total and each line's delivery date; on failure, the reason.""",
        )


# Set up your agents and create an orchestration agent that will manage them.
def recorded_sale_total(quote_id: Optional[str]) -> float:
    """Total actually recorded as sales for a quote; 0.0 if the quote was never finalized.
    Customer-facing totals come from here, so a reply can never state an unrecorded total."""
    quote = QUOTE_STORE.get(quote_id) if quote_id else None
    if not quote or not quote.get("used"):
        return 0.0
    return round(sum(line["net_total"] for line in quote["lines"]), 2)


def customer_quote_payload(quote_id: Optional[str]) -> Optional[Dict]:
    """Customer-facing pricing for a FINALIZED quote only (None if the sale was not recorded).
    The reply's total always comes from the recorded sale, never from an unfinalized quote."""
    quote = QUOTE_STORE.get(quote_id) if quote_id else None
    if not quote or not quote.get("used"):
        return None
    return {
        "lines": [{"item_name": l["item_name"], "units": l["units"], "ships_on": l["ready_date"],
                   "line_total": l["net_total"]} for l in quote["lines"]],
        "total": recorded_sale_total(quote_id),
        "discount_note": quote["discount_note"],
        "precedent_note": quote["precedent_note"],
    }


REPLY_OPENING = "Thank you for your request."
INTERNAL_WORDS = ["dear customer", "q-00", "quote id", "agent", "tool", "stock level", "cash", "error"]


def build_reply_from_facts(lines: List[Dict], payload: Optional[Dict]) -> str:
    """The customer reply built directly from verified facts, in the standard structure.
    Used whenever the LLM's wording breaks a rule, so a wrong reply never reaches a customer."""
    parts = [REPLY_OPENING]
    confirmed = [l for l in lines if l["outcome"] == "confirmed"]
    if confirmed and payload:
        parts.append("Confirmed items:\n" + "\n".join(
            f"{l['our_item']}, {l['units']} units, ships on {l['ships_on']}." for l in confirmed))
        notes = " ".join(n for n in [payload["discount_note"], payload.get("precedent_note")] if n)
        parts.append(f"Total for the confirmed items: ${payload['total']:,.2f}. {notes}")
    other = [l for l in lines if l["outcome"] != "confirmed" or not payload]
    if other:
        parts.append("Unavailable items (not ordered and not included in your total):\n" + "\n".join(
            f"{l['requested']}: {l['note']}" for l in other))
    return "\n\n".join(parts)


def reply_breaks_rules(reply: str, payload: Optional[Dict]) -> List[str]:
    """Check an LLM-written customer reply against the rules in code. Empty list = it passes."""
    problems = []
    if not reply.strip().startswith(REPLY_OPENING):
        problems.append("does not open with the standard first line")
    if payload is None and "$" in reply:
        problems.append("states a dollar amount although nothing was sold")
    if payload is not None and f"${payload['total']:,.2f}" not in reply:
        problems.append("does not state the recorded total")
    lowered = reply.lower()
    problems += [f"mentions '{w}'" for w in INTERNAL_WORDS if w in lowered]
    return problems


def _customer_note(line: Dict) -> str:
    """Translate an internal line result into plain customer language.
    Never reveals stock counts, cash, or internal codes."""
    reason = line.get("reason", "")
    if line["status"] == "available":
        return f"In stock; ships {line['ready_date']}."
    if line["status"] == "restock_ordered":
        return f"Ordered from our supplier; ships {line['ready_date']}."
    if reason == "not_carried":
        # Suggest an alternative only from the SAME product category (never kraft paper for
        # envelopes): the suggestion must be a genuine substitute, not just a word match.
        matches = match_catalog_item(line["requested"])
        if matches:
            category = CATALOG[matches[0]["item_name"]]["category"]
            closest = [c["item_name"] for c in matches[1:]
                       if c["stocked"] and c["score"] >= 0.5
                       and CATALOG[c["item_name"]]["category"] == category]
            if closest:
                return f"We don't carry this exact item; the closest item we stock is {closest[0]}."
        return "We don't currently carry this item."
    if reason.startswith("restock_arrives_"):
        earliest = reason.split("_")[2]
        return f"We can't source this quantity before your deadline (earliest availability: {earliest})."
    if reason == "insufficient_funds_for_restock":
        return "We're unable to source this quantity in time for your order."
    if reason == "invalid_quantity":
        return "The quantity wasn't clear; please confirm how many you need."
    return "We're unable to supply this item at the moment."


def customer_line_summary() -> List[Dict]:
    """Customer-safe view of the current request's lines (built from CURRENT_LINES in code)."""
    summary = []
    sale_failed = CURRENT_QUOTE.get("sale_failed", False)
    for line in CURRENT_LINES:
        confirmed = line["status"] in ("available", "restock_ordered")
        if confirmed and sale_failed:
            # The order could not be recorded: never tell the customer it is confirmed
            outcome, ships_on = "not_confirmed", None
            note = "We couldn't confirm this item right now; our team will contact you shortly."
        else:
            outcome = "confirmed" if confirmed else "unavailable"
            ships_on = line.get("ready_date") if confirmed else None
            note = _customer_note(line)
        summary.append({
            "requested": line["requested"],
            "our_item": line["item_name"] if line["item_name"] != "NONE" else None,
            "units": line.get("units"),
            "outcome": outcome,
            "ships_on": ships_on,
            "note": note,
        })
    return summary


def inventory_result_json() -> str:
    """Inventory delegation result: customer-safe lines plus the next step for the orchestrator."""
    lines = customer_line_summary()
    result = {"lines": lines}
    if any(line["outcome"] == "confirmed" for line in lines):
        result["next_step"] = "Call ask_quoting_agent next to price the confirmed items."
    else:
        result["next_step"] = "Nothing can be supplied. Write a polite reply giving the note for each item."
    return json.dumps(result)


DEFAULT_LEAD_DAYS = 14  # assumed deadline when a request does not state one

REPLY_RULES = """RULES FOR YOUR FINAL REPLY TO THE CUSTOMER (use this exact structure):
        1. First line: "Thank you for your request." No other greeting, no "Dear Customer", no sign-off.
        2. "Confirmed items:" one line per confirmed item: item, quantity, ships-on date.
           Leave this section out if nothing is confirmed.
        3. "Total for the confirmed items: $X.XX" using ONLY the total given, then the discount_note
           (and precedent_note if present). Leave this section out if no total is given; never state
           any other dollar amount.
        4. "Unavailable items (not ordered and not included in your total):" one line per item with
           the note given for it. Leave this section out if none.
        - Speak directly to the customer ("your order"). Only state facts you were given.
        - Never mention stock levels, cash, internal codes, quote IDs, agents, tools or errors."""


class OrchestratorAgent(ToolCallingAgent):
    """Orchestrator: reads each request, delegates to the workers, writes the customer reply."""

    def __init__(self, model):
        self.inventory_agent = InventoryAgent(model)
        self.quoting_agent = QuotingAgent(model)
        self.sales_agent = SalesAgent(model)

        @tool
        def ask_inventory_agent(request_text: str, request_date: str, deadline_date: str) -> str:
            """
            Delegate to the inventory agent: replenish low stock, then decide for every line
            of the request whether it can be supplied and by when.

            Args:
                request_text: The customer's full request text.
                request_date: Date of the request, YYYY-MM-DD.
                deadline_date: Customer's required delivery date, YYYY-MM-DD.

            Returns:
                JSON customer-safe summary of every line (outcome, ship date, plain-language note).
            """
            if CURRENT_LINES:  # already handled for this request: never run inventory twice
                return inventory_result_json()
            try:
                self.inventory_agent.run(
                    f"""Request date: {request_date}. Deadline: {deadline_date}.
                    Handle EVERY item line in this customer request:
                    {request_text}""")
            except Exception:
                pass  # line results already recorded in code are still returned below
            return inventory_result_json()

        @tool
        def ask_quoting_agent(request_date: str, context: str) -> str:
            """
            Delegate to the quoting agent: price the confirmed lines and explain the pricing.
            Safe to call more than once: a request never gets a second quote.

            Args:
                request_date: Date of the request, YYYY-MM-DD.
                context: Short context for precedent search, e.g. the event type ("ceremony").

            Returns:
                JSON with quote_id, line prices, total, discount_note, precedent_note, and next_step.
            """
            if CURRENT_QUOTE["quote_id"] is None:
                try:
                    self.quoting_agent.run(
                        f"Request date: {request_date}. Context: {context}. Price the confirmed lines.")
                except Exception:
                    pass
            return self._quote_summary(request_date)

        @tool
        def ask_sales_agent(quote_id: str, request_date: str) -> str:
            """
            Delegate to the sales agent: finalize the quoted order. Required before replying.

            Args:
                quote_id: The quote ID from ask_quoting_agent.
                request_date: Date of the request, YYYY-MM-DD.

            Returns:
                JSON saying whether the order was confirmed (sale recorded).
            """
            try:
                self.sales_agent.run(f"Finalize quote {quote_id}. Request date: {request_date}.")
            except Exception:
                pass
            finalized = quote_id in QUOTE_STORE and QUOTE_STORE[quote_id]["used"]
            return json.dumps({"quote_id": quote_id, "order_confirmed": finalized})

        super().__init__(
            tools=[ask_inventory_agent, ask_quoting_agent, ask_sales_agent],
            model=model,
            name="orchestrator",
            description="""Munder Difflin order orchestrator. For each customer request:
            1. Find the delivery deadline in the request (YYYY-MM-DD). If none is stated,
               use the deadline given to you.
            2. Call ask_inventory_agent with the FULL request text, request date and deadline.
            3. If at least one line is confirmed, call ask_quoting_agent.
            4. If a quote_id was returned, you MUST call ask_sales_agent with it. An order is not
               confirmed until the sale is recorded.
            5. Only then write the final reply TO the customer.""",
        )

    def _quote_summary(self, request_date: str) -> str:
        """Customer-safe summary of this request's quote. Creates it in code if the quoting
        agent did not, whenever confirmed lines exist (a confirmed line must always be priced)."""
        if CURRENT_QUOTE["quote_id"] is None:
            calculate_quote(request_date)  # no-op message if nothing is confirmed
        quote_id = CURRENT_QUOTE["quote_id"]
        if quote_id is None:
            return json.dumps({"quote_id": None, "message": "Nothing could be quoted."})
        quote = QUOTE_STORE[quote_id]
        total = recorded_sale_total(quote_id) if quote["used"] else quote["total"]
        return json.dumps({"quote_id": quote_id, "lines": quote["lines"],
                           "subtotal": quote["subtotal"], "discount_amount": quote["discount_amount"],
                           "total": total, "discount_note": quote["discount_note"],
                           "precedent_note": quote["precedent_note"],
                           "next_step": f"Call ask_sales_agent with quote_id {quote_id} "
                                        f"to confirm the order BEFORE writing your reply."})

    def _complete_workflow(self, request_date: str) -> bool:
        """Completion check (code). Guarantees every confirmed line is quoted and every quote is
        finalized, by delegating to the worker agents; raw tools are only a last resort.
        Returns True if anything had to be completed here."""
        completed_here = False
        has_confirmed = any(l["status"] in ("available", "restock_ordered") for l in CURRENT_LINES)

        if has_confirmed and CURRENT_QUOTE["quote_id"] is None:
            completed_here = True
            try:
                self.quoting_agent.run(f"Request date: {request_date}. Price the confirmed lines.")
            except Exception:
                pass
            if CURRENT_QUOTE["quote_id"] is None:
                calculate_quote(request_date)

        quote_id = CURRENT_QUOTE["quote_id"]
        if quote_id and not QUOTE_STORE[quote_id]["used"]:
            completed_here = True
            try:
                self.sales_agent.run(f"Finalize quote {quote_id}. Request date: {request_date}.")
            except Exception:
                pass
            if not QUOTE_STORE[quote_id]["used"]:
                finalize_sale(quote_id, request_date)
        return completed_here

    def handle_customer_request(self, request_text: str, request_date: str) -> str:
        """Run one customer request through the workflow and return the customer reply."""
        CURRENT_LINES.clear()
        CURRENT_QUOTE["quote_id"] = None
        CURRENT_QUOTE["sale_failed"] = False
        CURRENT_SPEND["restock"] = 0.0
        CURRENT_REQUEST["text"] = request_text
        request_date = normalize_date(request_date)
        default_deadline = (datetime.strptime(request_date, "%Y-%m-%d")
                            + timedelta(days=DEFAULT_LEAD_DAYS)).strftime("%Y-%m-%d")
        CURRENT_REQUEST["lines"] = parse_request_lines(request_text)
        CURRENT_REQUEST["deadline"] = parse_deadline(request_text, default_deadline)

        # The orchestrator delegates to the workers; its own reply here is only a draft.
        self.run(f"""
        Customer request: {request_text}
        Request date: {request_date}
        If the request states no delivery deadline, use {default_deadline}.
        After the workers are done, give a one-line summary; the customer reply is written next.
        """)

        # Code guarantees the workflow finished (missing quote or sale is completed by the agents).
        safety_net_used = self._complete_workflow(request_date)

        # The customer reply is ALWAYS written from verified facts: line outcomes plus the
        # finalized sale only. A total reaches the customer only if the sale was recorded.
        payload = customer_quote_payload(CURRENT_QUOTE["quote_id"])
        facts = {"lines": customer_line_summary(), "recorded_sale": payload,
                 "order_confirmed": payload is not None}
        reply = str(self.run(f"""
        The customer's request has been fully processed. Do NOT call any tools except final_answer.
        Write the final reply to the customer from these verified facts only:
        {json.dumps(facts)}
        {REPLY_RULES}
        """))

        # Code verifies the reply before it reaches the customer. If the wording breaks any
        # rule, the reply is built directly from the verified facts instead.
        problems = reply_breaks_rules(reply, payload)
        if problems:
            print(f"REPLY CHECK (internal): replaced LLM wording: {'; '.join(problems)}")
            reply = build_reply_from_facts(facts["lines"], payload)

        quote_id = CURRENT_QUOTE["quote_id"]
        fulfilled = bool(quote_id and QUOTE_STORE[quote_id]["used"])

        # Reorder low stock only after a recorded sale, so a rejected request never changes cash
        if fulfilled:
            try:
                self.inventory_agent.run(
                    f"Call replenish_low_stock with as_of_date {request_date}. Do not call any other tool.")
            except Exception:
                pass

        # Every CONFIRMED line must be something the customer actually wrote
        unrequested = [l["requested"] for l in CURRENT_LINES
                       if l["status"] in ("available", "restock_ordered")
                       and _normalize_text(l["requested"]) not in _normalize_text(request_text)]
        if unrequested:
            print(f"LINE CHECK (internal): confirmed lines not in the request: {unrequested}")

        # Every CONFIRMED line must ship by the deadline the customer wrote
        late = [l["requested"] for l in CURRENT_LINES
                if l["status"] in ("available", "restock_ordered")
                and (not l.get("ready_date")          # no ship date = can't be verified: flag it
                     or normalize_date(l["ready_date"]) > CURRENT_REQUEST["deadline"])]
        if late:
            print(f"DEADLINE CHECK (internal): confirmed lines shipping after "
                  f"{CURRENT_REQUEST['deadline']}: {late}")

        RUN_LOG.append({
            "request_date": request_date,
            "unrequested_lines": unrequested,
            "late_lines": late,
            "customer_deadline": CURRENT_REQUEST["deadline"],
            "lines": [dict(line) for line in CURRENT_LINES],
            "quote_total": recorded_sale_total(quote_id),
            "sale_recorded": recorded_sale_total(quote_id),
            "restock_spend": CURRENT_SPEND["restock"],
            "fulfilled": fulfilled,
            "safety_net_used": safety_net_used,
        })
        return reply


# ---------------------------------------------------------------------
# Business advisor (optional 5th agent): runs ONCE at the end, read-only
# ---------------------------------------------------------------------
@tool
def review_business_health(start_date: str, end_date: str) -> str:
    """
    Gather the evidence for a business review of the whole test run (read-only).

    Args:
        start_date: First request date of the run, YYYY-MM-DD.
        end_date: Last request date of the run, YYYY-MM-DD.

    Returns:
        JSON with cash at start and end, inventory value, top sellers, and order outcome statistics.
    """
    report = generate_financial_report(normalize_date(end_date))
    # Cash just before the first request (the day before), so the run's own sales count as change
    day_before = (datetime.strptime(normalize_date(start_date), "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    cash_start = get_cash_balance(day_before)
    # The starter records the opening $50,000 as a "sale" with no item name: not a product
    top_sellers = [p for p in report["top_selling_products"] if p.get("item_name")]

    declined_by_reason: Dict[str, int] = {}
    not_carried_requests: List[str] = []
    for record in RUN_LOG:
        for line in record["lines"]:
            if line["status"] == "declined":
                reason = line["reason"].split("_after_")[0] if line["reason"].startswith("restock_arrives") else line["reason"]
                declined_by_reason[reason] = declined_by_reason.get(reason, 0) + 1
                if line["reason"] == "not_carried":
                    not_carried_requests.append(line["requested"])

    return json.dumps({
        "cash_start": round(cash_start, 2),
        "cash_end": round(report["cash_balance"], 2),
        "inventory_value_end": round(report["inventory_value"], 2),
        "total_assets_end": round(report["total_assets"], 2),
        "top_selling_products": top_sellers,
        "requests_processed": len(RUN_LOG),
        "requests_fulfilled": sum(1 for r in RUN_LOG if r["fulfilled"]),
        "revenue_from_run": round(sum(r["quote_total"] for r in RUN_LOG if r["fulfilled"]), 2),
        # Units made explicit: a request contains several ITEM LINES, so line counts can
        # exceed the number of requests. The advisor must never call lines "requests".
        "total_item_lines_requested": sum(len(r["lines"]) for r in RUN_LOG),
        "declined_item_lines_by_reason": declined_by_reason,
        "requests_with_at_least_one_declined_item": sum(
            1 for r in RUN_LOG if any(l["status"] == "declined" for l in r["lines"])),
        "not_carried_item_lines": not_carried_requests,
        "orders_completed_by_code_check": sum(1 for r in RUN_LOG if r["safety_net_used"]),
        "inventory_value_start": round(generate_financial_report(day_before)["inventory_value"], 2),
        "cash_change": round(report["cash_balance"] - cash_start, 2),
        "inventory_value_change": round(report["inventory_value"]
                                        - generate_financial_report(day_before)["inventory_value"], 2),
        "total_assets_change": round(report["total_assets"]
                                     - generate_financial_report(day_before)["total_assets"], 2),
    }, default=str)


class BusinessAdvisorAgent(ToolCallingAgent):
    """Analyst agent: reviews the run's results and recommends improvements. Never acts."""

    def __init__(self, model):
        super().__init__(
            tools=[review_business_health],
            model=model,
            name="business_advisor",
            description="""Business advisor for Munder Difflin. Call review_business_health ONCE with the
            start and end dates you are given. Use the *_change fields as given (positive means it
            went up, negative means it went down); never work out changes yourself. Each request
            contains several item lines: counts named *item_lines* are item lines, NOT requests, so
            never describe them as requests. Then write 3 to 5
            numbered recommendations to improve
            efficiency and revenue. Each recommendation must cite specific evidence from the review
            (numbers, item names, decline reasons). You advise only; you cannot change anything.""",
        )


# Known request wordings and the catalog item each must resolve to (None = not carried).
# Guards against a customer being told "we don't carry this" about an item we stock.
MATCHER_CHECKS = [
    ("200 sheets of A4 printer paper", "A4 paper"),
    ("1000 sheets of A4 white printer paper", "A4 paper"),
    ("A4 glossy paper", "Glossy paper"),
    ("heavyweight cardstock", "Cardstock"),
    ('8.5"x11" colored paper', "Colored paper"),
    ("large poster paper", "Large poster paper (24x36 inches)"),
    ("kraft paper envelopes", "Envelopes"),
    ("A4 matte paper", "Matte paper"),
    ("A3 glossy paper", None),
    ("printer paper", "Standard copy paper"),
    ("balloons", None),
]


def check_item_matching() -> str:
    """Run the known wordings through the matcher before the evaluation; report any mismatch."""
    failures = []
    for wording, expected in MATCHER_CHECKS:
        found = match_catalog_item(wording, top_n=1)
        got = found[0]["item_name"] if found else None
        if got != expected:
            failures.append(f"{wording!r} -> {got} (expected {expected})")
    if failures:
        return "ITEM MATCHING FAILURES:\n" + "\n".join(failures)
    return f"Item matching: all {len(MATCHER_CHECKS)} known wordings resolve to the correct catalog item."


def audit_results(results: List[Dict]) -> str:
    """Check every row: rejected requests must not change cash, every cash change must equal
    recorded sale minus restock spend, and a fulfilled reply must state the recorded total."""
    issues = []
    for row in results:
        rid = row["request_id"]
        if row.get("late_lines", 0):
            issues.append(f"Request {rid}: {row['late_lines']} confirmed line(s) ship after the customer's deadline")
        if row.get("unrequested_lines", 0):
            issues.append(f"Request {rid}: {row['unrequested_lines']} confirmed line(s) the customer did not request")
        if not row["fulfilled"] and abs(row["cash_delta"]) >= 0.01:
            issues.append(f"Request {rid}: not fulfilled but cash changed by {row['cash_delta']:.2f}")
        if not row["reconciled"]:
            issues.append(f"Request {rid}: cash change {row['cash_delta']:.2f} != sale "
                          f"{row['recorded_sale_total']:.2f} - restock {row['restock_spend']:.2f}")
        if row["fulfilled"]:
            amounts = {round(float(a.replace(",", "")), 2)
                       for a in re.findall(r"\$\s?([\d,]+(?:\.\d{1,2})?)", str(row["response"]))}
            if row["recorded_sale_total"] not in amounts:
                issues.append(f"Request {rid}: reply does not state the recorded total "
                              f"${row['recorded_sale_total']:.2f}")
    if not issues:
        return f"All {len(results)} rows reconcile: cash changes match recorded sales and restocks."
    return "\n".join(issues)


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    orchestrator = OrchestratorAgent(model)
    RUN_LOG.clear()
    QUOTE_STORE.clear()
    print("\n===== ITEM MATCHING CHECK =====")
    print(check_item_matching())

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        cash_before = current_cash
        try:
            response = orchestrator.handle_customer_request(request_with_date, request_date)
        except Exception as error:
            # Never show internal errors to the customer; keep them in the console for the operator
            print(f"INTERNAL ERROR (not shown to customer): {error}")
            response = ("We're sorry, we couldn't complete your request right now. "
                        "Our team will follow up with you shortly.")

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        # Audit fields: every row must reconcile (cash change = recorded sale - restock spend)
        log = RUN_LOG[-1] if RUN_LOG else {"fulfilled": False, "sale_recorded": 0.0, "restock_spend": 0.0}
        cash_delta = round(current_cash - cash_before, 2)  # change caused by THIS request
        expected_delta = round(log["sale_recorded"] - log["restock_spend"], 2)
        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
                "fulfilled": log["fulfilled"],
                "cash_before": round(cash_before, 2),
                "cash_after": round(current_cash, 2),
                "cash_delta": cash_delta,
                "recorded_sale_total": log["sale_recorded"],
                "restock_spend": log["restock_spend"],
                "reconciled": abs(cash_delta - expected_delta) < 0.01,
                "unrequested_lines": len(log.get("unrequested_lines", [])),
                "customer_deadline": log.get("customer_deadline"),
                "late_lines": len(log.get("late_lines", [])),
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)

    # Validation pass: catch ledger/response mismatches before submission
    print("\n===== EVALUATION AUDIT =====")
    print(audit_results(results))

    # Business advisor: one review of the whole run
    try:
        advisor = BusinessAdvisorAgent(model)
        advice = str(advisor.run(
            f"Review the business for the period {initial_date} to {final_date} and recommend improvements."))
        print("\n===== BUSINESS ADVISOR RECOMMENDATIONS =====")
        print(advice)
        with open("advisor_recommendations.txt", "w") as advice_file:
            advice_file.write(advice)
    except Exception as error:
        print(f"Business advisor could not run: {error}")
    return results


if __name__ == "__main__":
    results = run_test_scenarios()