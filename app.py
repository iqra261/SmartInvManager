import os
import sqlite3
from datetime import datetime, date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq

# ---------------------------------------------------------
# Smart Inventory Manager
# Files required: app.py and requirements.txt
# ---------------------------------------------------------

st.set_page_config(
    page_title="Smart Inventory Manager",
    page_icon="📦",
    layout="wide",
)

DB_PATH = "inventory.db"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


# ------------------------- Database -------------------------

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                category TEXT NOT NULL DEFAULT 'Other',
                price REAL NOT NULL CHECK(price >= 0),
                quantity INTEGER NOT NULL DEFAULT 0 CHECK(quantity >= 0),
                min_stock INTEGER NOT NULL DEFAULT 5 CHECK(min_stock >= 0),
                supplier TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                product_name TEXT NOT NULL,
                quantity INTEGER NOT NULL CHECK(quantity > 0),
                unit_price REAL NOT NULL,
                total REAL NOT NULL,
                sold_at TEXT NOT NULL,
                FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE SET NULL
            )
        """)


def read_products():
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT id, name, category, price, quantity, min_stock, supplier, created_at "
            "FROM products ORDER BY name",
            conn,
        )


def read_sales(days=30):
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT id, product_id, product_name, quantity, unit_price, total, sold_at "
            "FROM sales WHERE sold_at >= ? ORDER BY sold_at DESC",
            conn,
            params=(cutoff,),
        )


def add_product(name, category, price, quantity, min_stock, supplier):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO products
               (name, category, price, quantity, min_stock, supplier, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                name.strip(),
                category.strip() or "Other",
                float(price),
                int(quantity),
                int(min_stock),
                supplier.strip(),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )


def restock_product(product_id, amount):
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE products SET quantity = quantity + ? WHERE id = ?",
            (int(amount), int(product_id)),
        )
        if cur.rowcount == 0:
            raise ValueError("Product not found.")


def delete_product(product_id):
    with get_connection() as conn:
        conn.execute("DELETE FROM products WHERE id = ?", (int(product_id),))


def record_sale(product_id, amount):
    amount = int(amount)
    with get_connection() as conn:
        product = conn.execute(
            "SELECT id, name, price, quantity FROM products WHERE id = ?",
            (int(product_id),),
        ).fetchone()
        if product is None:
            raise ValueError("Product not found.")
        if amount < 1:
            raise ValueError("Sale quantity must be at least 1.")
        if amount > product["quantity"]:
            raise ValueError(
                f"Not enough stock. Available quantity: {product['quantity']}."
            )

        total = float(product["price"]) * amount
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "UPDATE products SET quantity = quantity - ? WHERE id = ?",
            (amount, int(product_id)),
        )
        conn.execute(
            """INSERT INTO sales
               (product_id, product_name, quantity, unit_price, total, sold_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                int(product_id),
                product["name"],
                amount,
                float(product["price"]),
                total,
                now,
            ),
        )


init_db()


# ------------------------- Helpers -------------------------

def money(value):
    return f"Rs {float(value):,.2f}"


def inventory_summary(products, sales):
    if products.empty:
        total_units = 0
        inventory_value = 0.0
        low_stock_count = 0
    else:
        total_units = int(products["quantity"].sum())
        inventory_value = float((products["price"] * products["quantity"]).sum())
        low_stock_count = int((products["quantity"] <= products["min_stock"]).sum())

    total_sales = float(sales["total"].sum()) if not sales.empty else 0.0
    units_sold = int(sales["quantity"].sum()) if not sales.empty else 0

    return {
        "product_count": len(products),
        "total_units": total_units,
        "inventory_value": inventory_value,
        "low_stock_count": low_stock_count,
        "sales_total": total_sales,
        "units_sold": units_sold,
    }


def get_groq_key():
    # Recommended: set GROQ_API_KEY as an environment variable or Streamlit secret.
    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key:
        return env_key.strip()
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
        if secret_key:
            return str(secret_key).strip()
    except Exception:
        pass
    # Convenient for local demos. The key is held in the current app session only.
    return st.session_state.get("groq_api_key", "").strip()


def build_inventory_context(products, recent_sales):
    if products.empty:
        product_text = "No products have been added yet."
    else:
        safe_products = products[
            ["name", "category", "price", "quantity", "min_stock", "supplier"]
        ].copy()
        product_text = safe_products.to_string(index=False)

    if recent_sales.empty:
        sales_text = "No sales have been recorded in the selected recent period."
    else:
        safe_sales = recent_sales[
            ["product_name", "quantity", "unit_price", "total", "sold_at"]
        ].head(30)
        sales_text = safe_sales.to_string(index=False)

    summary = inventory_summary(products, recent_sales)
    return f"""
Current inventory summary:
- Number of product types: {summary['product_count']}
- Units currently in stock: {summary['total_units']}
- Estimated inventory value at listed selling prices: {summary['inventory_value']:.2f} PKR
- Products at or below their minimum-stock threshold: {summary['low_stock_count']}
- Sales revenue in the recent 30-day window: {summary['sales_total']:.2f} PKR
- Units sold in the recent 30-day window: {summary['units_sold']}

Product records:
{product_text}

Recent sales records (up to 30 rows):
{sales_text}
"""


# ------------------------- Sidebar -------------------------

st.sidebar.title("📦 Inventory Manager")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Products", "Record a Sale", "Restock Products", "AI Assistant"],
)
st.sidebar.caption("A simple inventory and sales tracker for small businesses.")

st.sidebar.divider()
st.sidebar.subheader("Groq AI settings")
st.sidebar.caption(
    "Enter a replacement Groq API key here, or set GROQ_API_KEY in your environment/secrets."
)
st.sidebar.text_input(
    "Groq API key",
    type="password",
    key="groq_api_key",
    help="Do not share your API key or commit it to a public repository.",
)
model_name = st.sidebar.text_input(
    "Groq model",
    value=st.session_state.get("groq_model", DEFAULT_MODEL),
    key="groq_model",
)
st.sidebar.caption("The AI assistant uses your current inventory and recent sales data.")


# Refresh data for every Streamlit rerun
products = read_products()
sales_30 = read_sales(days=30)
summary = inventory_summary(products, sales_30)


# ------------------------- Dashboard -------------------------

if page == "Dashboard":
    st.title("📊 Smart Inventory Dashboard")
    st.write("Monitor your stock, sales, and products that may need restocking.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Product types", f"{summary['product_count']:,}")
    c2.metric("Units in stock", f"{summary['total_units']:,}")
    c3.metric("Inventory value", money(summary["inventory_value"]))
    c4.metric("Sales (last 30 days)", money(summary["sales_total"]))

    st.subheader("⚠️ Low-stock products")
    if products.empty:
        st.info("No products yet. Open **Products** to add your first item.")
    else:
        low_stock = products[products["quantity"] <= products["min_stock"]].copy()
        if low_stock.empty:
            st.success("No products are currently at or below their minimum-stock level.")
        else:
            low_stock["status"] = low_stock.apply(
                lambda row: "Out of stock" if row["quantity"] == 0 else "Low stock",
                axis=1,
            )
            st.dataframe(
                low_stock[["name", "category", "quantity", "min_stock", "supplier", "status"]],
                use_container_width=True,
                hide_index=True,
            )

    left, right = st.columns(2)
    with left:
        st.subheader("Stock by product")
        if not products.empty:
            stock_chart = products.sort_values("quantity", ascending=False).head(12)
            fig = px.bar(
                stock_chart,
                x="name",
                y="quantity",
                color="category",
                title="Current units in stock (top 12)",
                labels={"name": "Product", "quantity": "Units"},
            )
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add products to see stock charts.")

    with right:
        st.subheader("Sales revenue by day")
        if not sales_30.empty:
            chart_sales = sales_30.copy()
            chart_sales["day"] = pd.to_datetime(chart_sales["sold_at"]).dt.date
            daily = (
                chart_sales.groupby("day", as_index=False)["total"]
                .sum()
                .sort_values("day")
            )
            fig = px.line(
                daily,
                x="day",
                y="total",
                markers=True,
                title="Daily revenue (last 30 days)",
                labels={"day": "Date", "total": "Revenue (PKR)"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Record sales to see the revenue chart.")

    st.subheader("Recent sales")
    if sales_30.empty:
        st.info("No sales recorded in the last 30 days.")
    else:
        st.dataframe(
            sales_30[["sold_at", "product_name", "quantity", "unit_price", "total"]].head(10),
            use_container_width=True,
            hide_index=True,
        )


# ------------------------- Products -------------------------

elif page == "Products":
    st.title("🛍️ Product Management")
    st.write("Add new products, review stock, or remove a product record.")

    with st.expander("➕ Add a product", expanded=products.empty):
        with st.form("add_product_form", clear_on_submit=True):
            name = st.text_input("Product name", placeholder="e.g. Rice 5 kg")
            category = st.selectbox(
                "Category",
                ["Grocery", "Beverages", "Household", "Clothing", "Electronics", "Other"],
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                price = st.number_input("Selling price (PKR)", min_value=0.0, value=100.0, step=10.0)
            with col2:
                quantity = st.number_input("Opening stock (units)", min_value=0, value=10, step=1)
            with col3:
                min_stock = st.number_input("Minimum stock alert", min_value=0, value=5, step=1)
            supplier = st.text_input("Supplier (optional)")
            submitted = st.form_submit_button("Add product", type="primary")

            if submitted:
                if not name.strip():
                    st.error("Please enter a product name.")
                else:
                    try:
                        add_product(name, category, price, quantity, min_stock, supplier)
                        st.success(f"Added **{name.strip()}** successfully.")
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("A product with this name already exists. Use a unique name.")

    st.subheader("All products")
    products = read_products()
    if products.empty:
        st.info("Your product list is empty.")
    else:
        display = products[["id", "name", "category", "price", "quantity", "min_stock", "supplier"]].copy()
        display["Stock status"] = display.apply(
            lambda r: "Out of stock" if r["quantity"] == 0
            else ("Low stock" if r["quantity"] <= r["min_stock"] else "In stock"),
            axis=1,
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.subheader("Delete a product")
        st.warning("Deleting a product removes it from the product list. Existing sales history is retained.")
        options = {f"{row['name']} (ID {row['id']})": int(row["id"]) for _, row in products.iterrows()}
        with st.form("delete_product_form"):
            chosen = st.selectbox("Choose product", list(options.keys()))
            confirm = st.checkbox("I understand this action cannot be undone.")
            delete_submitted = st.form_submit_button("Delete selected product")
            if delete_submitted:
                if confirm:
                    delete_product(options[chosen])
                    st.success("Product deleted.")
                    st.rerun()
                else:
                    st.error("Please confirm before deleting.")


# ------------------------- Record a Sale -------------------------

elif page == "Record a Sale":
    st.title("🧾 Record a Sale")
    st.write("Record a sale to automatically reduce available stock and update sales reports.")
    products = read_products()

    if products.empty:
        st.info("Add a product before recording a sale.")
    else:
        options = {
            f"{row['name']} — {int(row['quantity'])} in stock — {money(row['price'])} each": int(row["id"])
            for _, row in products.iterrows()
        }
        with st.form("sale_form"):
            chosen = st.selectbox("Product", list(options.keys()))
            quantity_sold = st.number_input("Quantity sold", min_value=1, value=1, step=1)
            submitted = st.form_submit_button("Record sale", type="primary")

            if submitted:
                try:
                    record_sale(options[chosen], quantity_sold)
                    st.success("Sale recorded and stock updated.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    st.subheader("Sales history")
    all_sales = read_sales(days=3650)
    if all_sales.empty:
        st.info("No sales have been recorded yet.")
    else:
        st.dataframe(
            all_sales[["sold_at", "product_name", "quantity", "unit_price", "total"]].head(100),
            use_container_width=True,
            hide_index=True,
        )


# ------------------------- Restock Products -------------------------

elif page == "Restock Products":
    st.title("📥 Restock Products")
    st.write("Record newly received stock. This increases inventory without recording a sale.")
    products = read_products()

    if products.empty:
        st.info("Add a product before restocking.")
    else:
        options = {
            f"{row['name']} — current stock: {int(row['quantity'])}": int(row["id"])
            for _, row in products.iterrows()
        }
        with st.form("restock_form"):
            chosen = st.selectbox("Product to restock", list(options.keys()))
            amount = st.number_input("Units received", min_value=1, value=1, step=1)
            submitted = st.form_submit_button("Update stock", type="primary")
            if submitted:
                restock_product(options[chosen], amount)
                st.success("Stock updated successfully.")
                st.rerun()


# ------------------------- AI Assistant -------------------------

elif page == "AI Assistant":
    st.title("🤖 Groq Inventory Assistant")
    st.write(
        "Ask questions about your products, stock levels, and recorded sales. "
        "Answers are based on the current data supplied to the model."
    )

    api_key = get_groq_key()
    if not api_key:
        st.warning(
            "Enter your replacement Groq API key in the sidebar to use the AI assistant. "
            "Never put a real API key directly into app.py."
        )
    else:
        st.caption(f"Model: {model_name or DEFAULT_MODEL}")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input(
            "e.g. Which products need restocking, and why?"
        )
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            current_products = read_products()
            current_sales = read_sales(days=30)
            context = build_inventory_context(current_products, current_sales)

            system_prompt = """
You are a careful inventory assistant for a small business.
Use only the inventory and sales data provided in the current context for factual claims.
Do not invent product names, stock quantities, sales, suppliers, or prices.
If the data needed to answer is missing, clearly say so.
Treat inventory value as an estimate based on listed selling prices, not accounting cost.
The sales data covers the last 30 days unless the context says otherwise.
Give practical, concise recommendations. Clearly label suggestions as estimates, not guarantees.
Do not claim that you have changed inventory records; the app's forms perform those actions.
"""
            try:
                client = Groq(api_key=api_key)
                messages = [{"role": "system", "content": system_prompt + "\n\nCURRENT DATA:\n" + context}]
                # Send a short conversation history to retain context.
                messages.extend(st.session_state.chat_history[-8:])
                with st.chat_message("assistant"):
                    with st.spinner("Thinking with Groq..."):
                        response = client.chat.completions.create(
                            model=(model_name.strip() or DEFAULT_MODEL),
                            messages=messages,
                            temperature=0.2,
                            max_tokens=900,
                        )
                        answer = response.choices[0].message.content or "I couldn't generate a response."
                    st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as exc:
                st.error(
                    "The Groq request failed. Check that the API key is valid, the model name is supported, "
                    "and your internet connection is available. Details: " + str(exc)
                )

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()


st.sidebar.divider()
st.sidebar.caption("Prototype note: data is saved in inventory.db in the app's current working directory.")
