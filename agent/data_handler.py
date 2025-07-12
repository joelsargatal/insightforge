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
    # def sales_by_time_period(self, start_date: str, end_date: str): # To-Do - Future implementation: Analysis by custom dates
        """
        Aggregate sales total by time period.
        period:
            "ME" = monthly
            "QE" = quarterly
            "YE" = yearly
        """
        self.df.time = self.df.set_index("Date").resample(period)["Sales"].sum()
        return self.df.time.to_string()

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
        # return pivot.to_string()
    
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

    # later you can add:
    # def sales_by_month(self): ...
    # def pivot_product_region(self): ...