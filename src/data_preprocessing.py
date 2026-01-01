import pandas as pd


def load_and_clean_data(data_path: str) -> pd.DataFrame:
    """
    Load raw Online Retail data and apply basic cleaning steps.

    Cleaning steps:
    - Remove rows with missing CustomerID
    - Remove cancelled invoices (InvoiceNo starting with 'C')
    - Remove negative or zero quantity
    - Remove negative or zero unit price

    Parameters
    ----------
    data_path : str
        Path to the Excel file

    Returns
    -------
    pd.DataFrame
        Cleaned transactional dataframe
    """

    # Load data
    df = pd.read_excel(data_path)

    # Remove missing CustomerID
    df = df.dropna(subset=["CustomerID"])

    # Ensure InvoiceNo is string
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)

    # Remove cancelled invoices
    df = df[~df["InvoiceNo"].str.startswith("C")]

    # Remove invalid quantities and prices
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    return df
