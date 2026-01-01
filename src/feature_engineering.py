import pandas as pd


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transactional data to customer-level features.

    Features created:
    - last_purchase_date
    - num_orders
    - total_quantity
    - total_spent
    - recency_days

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactional dataframe

    Returns
    -------
    pd.DataFrame
        Customer-level feature dataframe
    """

    # Ensure datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Reference date (end of dataset)
    reference_date = df["InvoiceDate"].max()

    # Aggregate per customer
    customer_df = (
        df.groupby("CustomerID")
        .agg(
            last_purchase_date=("InvoiceDate", "max"),
            num_orders=("InvoiceNo", "nunique"),
            total_quantity=("Quantity", "sum"),
            total_spent=("UnitPrice", lambda x: (x * df.loc[x.index, "Quantity"]).sum()),
        )
        .reset_index()
    )

    # Compute recency
    customer_df["recency_days"] = (
        reference_date - customer_df["last_purchase_date"]
    ).dt.days

    return customer_df
