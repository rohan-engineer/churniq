"""
ChurnIQ Exploratory Data Analysis (EDA) Runner
Generates comprehensive EDA charts using Pandas, Matplotlib, and Seaborn,
and exports notebooks/exploratory_analysis.ipynb.
"""

import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from src.data_preprocessing import load_dataset, clean_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Style configuration
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 1.2,
})

COLORS = {
    "primary": "#2563eb",
    "danger": "#ef4444",
    "navy": "#0f172a",
    "palette": ["#2563eb", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"]
}


def run_eda(save_plots: bool = True) -> dict:
    """Performs end-to-end exploratory analysis and saves plots."""
    df = load_dataset()
    cleaned = clean_dataset(df)
    
    total = len(cleaned)
    churn_counts = cleaned["Churn"].value_counts()
    churned = int(churn_counts.get("Yes", 0))
    retained = int(churn_counts.get("No", 0))
    churn_rate = (churned / total) * 100
    
    logger.info("=" * 50)
    logger.info("EXPLORATORY DATA ANALYSIS SUMMARY")
    logger.info(f"Total Customers:    {total}")
    logger.info(f"Retained (No):      {retained} ({100 - churn_rate:.2f}%)")
    logger.info(f"Churned (Yes):      {churned} ({churn_rate:.2f}%)")
    logger.info("=" * 50)
    
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if save_plots:
        # 1. Overall Churn Distribution Pie / Bar
        plt.figure(figsize=(7, 5))
        churn_counts.plot(
            kind="pie",
            autopct="%1.1f%%",
            colors=[COLORS["primary"], COLORS["danger"]],
            labels=["Retained (No)", "Churned (Yes)"],
            explode=[0, 0.08],
            startangle=90,
            textprops={"fontsize": 12, "weight": "bold"}
        )
        plt.title("Overall Customer Churn Distribution", fontsize=14, weight="bold", pad=15)
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(config.REPORTS_DIR / "eda_churn_overview.png", dpi=200)
        plt.close()
        
        # 2. Churn by Contract Type
        plt.figure(figsize=(8, 5))
        contract_order = ["Month-to-month", "One year", "Two year"]
        sns.countplot(
            data=cleaned,
            x="Contract",
            hue="Churn",
            order=contract_order,
            palette=[COLORS["primary"], COLORS["danger"]],
        )
        plt.title("Customer Churn by Contract Type", fontsize=14, weight="bold", pad=15)
        plt.xlabel("Contract Duration", fontsize=12, labelpad=10)
        plt.ylabel("Number of Customers", fontsize=12, labelpad=10)
        plt.legend(title="Churn", labels=["Retained", "Churned"])
        plt.tight_layout()
        plt.savefig(config.REPORTS_DIR / "eda_contract_churn.png", dpi=200)
        plt.close()
        
        # 3. Churn by Payment Method
        plt.figure(figsize=(10, 5))
        sns.countplot(
            data=cleaned,
            y="PaymentMethod",
            hue="Churn",
            palette=[COLORS["primary"], COLORS["danger"]],
        )
        plt.title("Customer Churn by Payment Method", fontsize=14, weight="bold", pad=15)
        plt.xlabel("Number of Customers", fontsize=12, labelpad=10)
        plt.ylabel("Payment Method", fontsize=12, labelpad=10)
        plt.legend(title="Churn", labels=["Retained", "Churned"])
        plt.tight_layout()
        plt.savefig(config.REPORTS_DIR / "eda_payment_churn.png", dpi=200)
        plt.close()
        
        # 4. Tenure vs Churn Distribution (KDE)
        plt.figure(figsize=(9, 5))
        sns.histplot(
            data=cleaned,
            x="tenure",
            hue="Churn",
            kde=True,
            palette=[COLORS["primary"], COLORS["danger"]],
            element="step",
            bins=30,
        )
        plt.title("Customer Tenure Distribution by Churn Status", fontsize=14, weight="bold", pad=15)
        plt.xlabel("Tenure (Months with Company)", fontsize=12, labelpad=10)
        plt.ylabel("Customer Count", fontsize=12, labelpad=10)
        plt.legend(title="Churn", labels=["Churned", "Retained"])
        plt.tight_layout()
        plt.savefig(config.REPORTS_DIR / "eda_tenure_distribution.png", dpi=200)
        plt.close()
        
        # 5. Monthly Charges Distribution
        plt.figure(figsize=(9, 5))
        sns.boxplot(
            data=cleaned,
            x="Churn",
            y="MonthlyCharges",
            palette=[COLORS["primary"], COLORS["danger"]],
        )
        plt.title("Monthly Charges vs Churn Status", fontsize=14, weight="bold", pad=15)
        plt.xlabel("Churn Status", fontsize=12, labelpad=10)
        plt.ylabel("Monthly Charges ($)", fontsize=12, labelpad=10)
        plt.tight_layout()
        plt.savefig(config.REPORTS_DIR / "eda_monthly_charges.png", dpi=200)
        plt.close()
        
        # 6. Service Adoption Analysis
        services = ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection"]
        service_data = []
        for s in services:
            churn_rate_s = cleaned[cleaned[s] == "Yes"]["Churn"].apply(lambda x: 1 if x == "Yes" else 0).mean() * 100
            churn_rate_no_s = cleaned[cleaned[s] == "No"]["Churn"].apply(lambda x: 1 if x == "Yes" else 0).mean() * 100
            service_data.append({"Service": s, "Has Service": "Yes", "Churn Rate (%)": churn_rate_s})
            service_data.append({"Service": s, "Has Service": "No", "Churn Rate (%)": churn_rate_no_s})
            
        serv_df = pd.DataFrame(service_data)
        plt.figure(figsize=(9, 5))
        sns.barplot(
            data=serv_df,
            x="Service",
            y="Churn Rate (%)",
            hue="Has Service",
            palette=[COLORS["primary"], COLORS["danger"]],
        )
        plt.title("Churn Rate by Security & Support Services", fontsize=14, weight="bold", pad=15)
        plt.xlabel("Add-on Service", fontsize=12, labelpad=10)
        plt.ylabel("Churn Rate (%)", fontsize=12, labelpad=10)
        plt.tight_layout()
        plt.savefig(config.REPORTS_DIR / "eda_services.png", dpi=200)
        plt.close()
        
        logger.info("All EDA charts successfully saved in reports/")
        
    return {
        "total": total,
        "churned": churned,
        "retained": retained,
        "churn_rate": churn_rate,
    }


def create_jupyter_notebook():
    """Generates the exploratory_analysis.ipynb file."""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# ChurnIQ: Exploratory Data Analysis\n",
                    "## Telecom Customer Churn Dataset Investigation\n",
                    "This notebook demonstrates exploratory data analysis (EDA) on the IBM Telco Customer Churn dataset."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "sns.set_theme(style='whitegrid')\n",
                    "df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')\n",
                    "print('Dataset Shape:', df.shape)\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Data Cleaning and Type Coercion\n",
                    "`TotalCharges` has empty spaces `' '` for customers with `tenure == 0`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')\n",
                    "df['TotalCharges'] = df['TotalCharges'].fillna(0.0)\n",
                    "print('Missing values after cleaning:\\n', df[['tenure', 'MonthlyCharges', 'TotalCharges']].isna().sum())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Overall Churn Ratio"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "churn_counts = df['Churn'].value_counts()\n",
                    "print(churn_counts)\n",
                    "plt.figure(figsize=(6, 4))\n",
                    "churn_counts.plot(kind='pie', autopct='%1.1f%%', colors=['#2563eb', '#ef4444'], explode=[0, 0.08])\n",
                    "plt.title('Overall Customer Churn')\n",
                    "plt.ylabel('')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Churn by Contract Type"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(8, 4))\n",
                    "sns.countplot(data=df, x='Contract', hue='Churn', palette=['#2563eb', '#ef4444'])\n",
                    "plt.title('Churn by Contract Type')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Churn by Payment Method"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(10, 4))\n",
                    "sns.countplot(data=df, y='PaymentMethod', hue='Churn', palette=['#2563eb', '#ef4444'])\n",
                    "plt.title('Churn by Payment Method')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Tenure and Monthly Charges"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, ax = plt.subplots(1, 2, figsize=(14, 5))\n",
                    "sns.histplot(data=df, x='tenure', hue='Churn', kde=True, ax=ax[0], palette=['#2563eb', '#ef4444'])\n",
                    "ax[0].set_title('Tenure Distribution')\n",
                    "sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=ax[1], palette=['#2563eb', '#ef4444'])\n",
                    "ax[1].set_title('Monthly Charges vs Churn')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    nb_path = config.BASE_DIR / "notebooks" / "exploratory_analysis.ipynb"
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)
    logger.info(f"EDA Jupyter notebook generated at: {nb_path}")


if __name__ == "__main__":
    run_eda(save_plots=True)
    create_jupyter_notebook()
