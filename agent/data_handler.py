# Data loading and analysis
import pandas as pd

# Data (sales CSV file) loading and analysis

class DataHandler:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path, parse_dates=["Date"])
        
        # Convert to categories for improved performance
        self.df["Product"] = self.df["Product"].astype("category")
        self.df["Region"] = self.df["Region"].astype("category")
        self.df["Customer_Gender"] = self.df["Customer_Gender"].astype("category")

        # Additional columns for more efficient analysis
        self.df["Year"] = self.df["Date"].dt.year
        self.df["Month"] = self.df["Date"].dt.month
        self.df["Weekday"] = self.df["Date"].dt.day_name()
    
    # Method for producing sales by time period, called by the agent tool
    def sales_by_time_period(self, period: str):
        """
        Aggregate sales total by time period.
        
        Parameters:
            period (str): 
                "ME" = month end
                "QE" = quarter end
                "YE" = year end
        Returns:
            pd.DataFrame: DataFrame with time period and aggregated sales
        """
        if period not in ["ME", "QE", "YE"]:
            raise ValueError("Invalid period. Use 'ME', 'QE', or 'YE'.")

        df_grouped = (
            self.df.set_index("Date")
            .resample(period)["Sales"]
            .sum()
            .reset_index()
            .rename(columns={"Date": "Period", "Sales": "Total Sales"})
        )

        return df_grouped
    
    # Nethod for producing sales by product and region, called by the agent tool
    def sales_by_product_region(self):
        """
        Returns a pivot table showing sales totals by product and region.
        """
        pivot = self.df.pivot_table(
            index="Product",
            columns="Region",
            values="Sales",
            aggfunc="sum"
        ).fillna(0)
        return pivot.to_dict() # Return dict to prevent issues in displaying the table format. Dict follows JSON format. The UI
        #  will convert it back to a table
    
    # Method for producing customer segmentation analysis, called by the agent tool
    def sales_by_cust_segment(self):
        """
        Segments customers by age group and gender, reporting total sales and average satisfaction.
        Returns a DataFrame as JSON for easy visualization.
        """
        # define age bins
        age_bins = [18, 25, 35, 45, 55, 65, 80]
        self.df["Age_Group"] = pd.cut(self.df["Customer_Age"], bins=age_bins, right=False)

        # group by
        segment = self.df.groupby(["Age_Group", "Customer_Gender"]).agg(
            Total_Sales=("Sales", "sum"),
            Average_Satisfaction=("Customer_Satisfaction", "mean")
        ).reset_index()

        return segment.to_dict(orient="records")

    # Method for producing statistical data, called by the agent tool
    def statistical_metrics(self):
        """
        Returns the pandas describe summary as string.
        """
        return self.df.describe().to_string()
    
    # Plotting-related modifications

    # Methods for data preparation for plotting results
    def get_monthly_sales_summary(self):
        """
        Returns a DataFrame with total sales by month for plotting.
        """
        monthly = self.df.set_index("Date").resample("M")["Sales"].sum().reset_index()
        monthly["Month"] = monthly["Date"].dt.strftime("%Y-%m")
        monthly_data = monthly[["Month", "Sales"]].rename(columns={"Sales": "Total Sales"})
        return {
            "type": "plot_data",
            "data": monthly_data.to_dict(orient="records"),
            "x": "Month",
            "y": "Total Sales",
            "title": "Monthly Sales Performance"
        }

    def get_quarterly_sales_summary(self):
        """
        Returns a DataFrame with total sales by quarter for plotting.
        """
        quarterly = self.df.set_index("Date").resample("Q")["Sales"].sum().reset_index()
        quarterly["Quarter"] = quarterly["Date"].dt.to_period("Q").astype(str)
        quarterly_data = quarterly[["Quarter", "Sales"]].rename(columns={"Sales": "Total Sales"})
        return {
            "type": "plot_data",
            "data": quarterly_data.to_dict(orient="records"),
            "x": "Quarter",
            "y": "Total Sales",
            "title": "Quarterly Sales Performance"
        }

    def get_yearly_sales_summary(self):
        """
        Returns a DataFrame with total sales by year for plotting.
        """
        yearly = self.df.set_index("Date").resample("Y")["Sales"].sum().reset_index()
        yearly["Year"] = yearly["Date"].dt.strftime("%Y")
        yearly_data = yearly[["Year", "Sales"]].rename(columns={"Sales": "Total Sales"})
        return {
            "type": "plot_data",
            "data": yearly_data.to_dict(orient="records"),
            "x": "Year",
            "y": "Total Sales",
            "title": "Yearly Sales Performance"
        }

    def get_product_region_sales_summary(self):
        """
        Returns a structured dictionary with sales totals by product and region for plotting.
        """
        pivot = self.df.pivot_table(
            index="Product",
            columns="Region",
            values="Sales",
            aggfunc="sum"
        ).fillna(0)

        plot_data = pivot.reset_index().melt(id_vars="Product", var_name="Region", value_name="Total Sales")

        return {
            "type": "plot_data",
            "data": plot_data.to_dict(orient="records"),
            "x": "Product",
            "y": "Total Sales",
            "hue": "Region",  # Optional: helpful if plotting with seaborn or grouped bars
            "title": "Sales by Product and Region"
        }

    def get_customer_segment_sales_summary(self):
        """
        Returns a dictionary for plotting total sales by age group and gender.
        Structured for grouped bar plots (e.g., gender split within each age group).
        """
        age_bins = [18, 25, 35, 45, 55, 65, 80]
        self.df["Age_Group"] = pd.cut(self.df["Customer_Age"], bins=age_bins, right=False)

        segment = self.df.groupby(["Age_Group", "Customer_Gender"]).agg(
            Total_Sales=("Sales", "sum")
        ).reset_index()

        # Format Age_Group for plotting (as string)
        segment["Age_Group"] = segment["Age_Group"].astype(str)

        return {
            "type": "plot_data",
            "data": segment.to_dict(orient="records"),
            "x": "Age_Group",
            "y": "Total_Sales",
            "hue": "Customer_Gender",  # Optional, in case you want grouped bars
            "title": "Sales by Customer Segment (Age & Gender)"
        }

    def get_statistical_sales_summary(self):
        """
        Returns a structured summary of statistical metrics for numeric features for plotting.
        Filters only numeric columns and standard statistics from df.describe().
        """

        # Generate describe() summary (transposed to have metrics as rows)
        desc_df = self.df.describe(include="all").transpose().reset_index()
        # desc = self.df.describe().T  # Transpose to get features as rows

        # Keep only numeric columns (exclude object and datetime)
        numeric_types = ["int64", "float64"]
        filtered_df = desc_df[desc_df["index"].isin(self.df.select_dtypes(include=numeric_types).columns)]

        # Melt to long format: one row per (column, metric)
        melted = filtered_df.melt(id_vars="index", var_name="Metric", value_name="Value")
        melted = melted.rename(columns={"index": "Column"})

        # Drop any rows with missing or invalid numeric values
        melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")
        melted = melted.dropna(subset=["Value"])

        return {
            "type": "plot_data",
            "data": melted.to_dict(orient="records"),
            "x": "Column",
            "y": "Value",
            "hue": "Metric",
            "title": "Statistical Summary of Numeric Columns"
        }
