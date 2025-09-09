import sys
import pandas as pd
import pickle
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QTextEdit, QComboBox, QLineEdit,
    QGroupBox, QGraphicsDropShadowEffect
)
from PyQt5.QtGui import QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

def reshape_amount(X):
    return X.values.reshape(-1, 1)


# -------------------- LOGIC --------------------
class PersonalFinanceManager:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.income = self.df[self.df["Type"] == "Income"]["Amount"].sum()

        # Load ML model (if available)
        self.model = None
        try:
            with open("expense_model2.pkl", "rb") as f:
                self.model = pickle.load(f)
        except Exception as e:
            print("⚠️ Could not load model, using rule-based categorization:", e)

        # Handle expenses
        self.expenses = self.df[self.df["Type"] == "Expense"].copy()
        if not self.expenses.empty:
            if self.model:
                try:
                    self.expenses["Category"] = self.model.predict(
                        self.expenses[["Description", "Amount"]].astype(str)
                    )
                except Exception as e:
                    print("⚠️ Model prediction failed, using rule-based categorization:", e)
                    self.expenses["Category"] = self.expenses["Description"].apply(self.categorize)
            else:
                self.expenses["Category"] = self.expenses["Description"].apply(self.categorize)
        else:
            self.expenses["Category"] = []

    def categorize(self, desc):
        """Fallback rule-based categorization"""
        if not isinstance(desc, str):
            return "Other"
        desc = desc.lower()
        if "salary" in desc or "income" in desc:
            return "Income"
        if any(x in desc for x in ["zomato", "restaurant", "food"]): return "Food"
        if any(x in desc for x in ["uber", "bus", "petrol"]): return "Transport"
        if any(x in desc for x in ["amazon", "shopping"]): return "Shopping"
        if any(x in desc for x in ["bill", "recharge"]): return "Bills"
        if any(x in desc for x in ["movie", "ticket"]): return "Entertainment"
        return "Other"

    def summary(self):
        total_exp = self.expenses["Amount"].sum() if not self.expenses.empty else 0
        return self.income, total_exp, self.income - total_exp

    def expense_chart(self):
        if self.expenses.empty:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.text(0.5, 0.5, "No Expense Data", ha="center", va="center", fontsize=12)
            ax.axis("off")
            return fig

        category_summary = self.expenses.groupby("Category")["Amount"].sum()
        category_summary = category_summary[~category_summary.index.str.contains("income|salary", case=False)]

        fig, ax = plt.subplots(figsize=(4, 4))
        category_summary.plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        ax.set_title("Expense Distribution")
        return fig

    def expense_advisor_text(self):
        if self.expenses.empty:
            return "📊 No expense data available for review."

        inc, total_exp, sav = self.summary()
        category_summary = self.expenses.groupby("Category")["Amount"].sum()
        category_summary = category_summary[~category_summary.index.str.contains("income|salary", case=False)]

        text, high_spent = "📊 Monthly Expense Advisor:\n\n", False

        for cat, amt in category_summary.items():
            pct = (amt / total_exp) * 100 if total_exp else 0  # ✅ percent of expenses
            if pct > 30:
                text += f"⚠️ {cat}: ₹{amt} ({pct:.1f}%) – Too High!\n"; high_spent = True
            elif pct > 15:
                text += f"✔️ {cat}: ₹{amt} ({pct:.1f}%) – Moderate.\n"
            else:
                text += f"✅ {cat}: ₹{amt} ({pct:.1f}%) – Well managed.\n"

        text += f"\n📊 Total Reviewed Expenses: ₹{total_exp}\n"
        text += "\n" + ("⚠️ Overall: Try reducing expenses.\n" if high_spent else "✔️ Overall: Expenses under control!\n")

        # Emergency Fund based on 5% of total expenses
        emergency_fund = round(total_exp * 0.05, 2)
        text += f"\n⚡ Suggested Emergency Fund: ₹{emergency_fund}"

        return text

    def investment_advisor(self, savings, risk, expected_return):
        if savings <= 0: return "⚠️ You have no savings. Try reducing expenses."
        base = f"Based on your savings (₹{savings}), Risk: {risk}, Expected Return: {expected_return}%\n\n"
        options = {
            "Low": "💡 FD, RD, Debt Funds\n📌 60% FD, 30% Debt, 10% Gold",
            "Medium": "💡 SIPs, Balanced Funds\n📌 40% MF, 30% FD, 20% Gold, 10% Stocks",
            "High": "💡 Stocks, Equity MFs, Crypto\n📌 60% Stocks, 20% MFs, 10% Crypto, 10% Gold",
        }
        return base + options[risk]


# -------------------- UI --------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Personal Finance Manager")
        self.setGeometry(100, 100, 900, 700)
        self.manager, self.chart_canvas = None, None
        self.init_ui()

    def add_shadow(self, widget):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15); shadow.setXOffset(3); shadow.setYOffset(3)
        shadow.setColor(QColor(0, 0, 0, 100)); widget.setGraphicsEffect(shadow)

    def groupbox(self, title):
        box = QGroupBox(title)
        box.setStyleSheet("""
            QGroupBox { background:#f5f5f5; border:1px solid #aaa;
                        border-radius:10px; padding:8px; font-weight:bold; }
        """)
        self.add_shadow(box)
        return box

    def button(self, text, slot):
        btn = QPushButton(text); btn.clicked.connect(slot)
        btn.setStyleSheet("background:#dcdcdc; border-radius:6px; padding:5px;")
        return btn

    def init_ui(self):
        self.setStyleSheet("background:#fafafa;")
        main_layout = QVBoxLayout()

        # Upload
        main_layout.addWidget(self.button("📂 Upload CSV", self.upload_file))

        # ---- TOP: Summary + Chart ----
        top_layout = QHBoxLayout()

        # Summary
        expense_box = self.groupbox("Income / Expense / Saving")
        vbox = QVBoxLayout(expense_box)
        self.summary_label = QLabel("💰 Income: ₹0\n💸 Expenses: ₹0\n💵 Savings: ₹0")
        vbox.addWidget(self.summary_label)
        vbox.addWidget(self.button("Expense Review", self.show_expense_advisor))
        self.review_text = QTextEdit(readOnly=True); vbox.addWidget(self.review_text)
        top_layout.addWidget(expense_box)

        # Chart
        chart_box = self.groupbox("Charts")
        self.chart_layout = QVBoxLayout(chart_box)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.button("Pie Chart", self.display_chart))
        self.chart_layout.addLayout(btn_layout)
        top_layout.addWidget(chart_box)

        main_layout.addLayout(top_layout)

        # ---- Investment Advisor ----
        invest_outer = self.groupbox("Investment Advisor")
        vbox = QVBoxLayout(invest_outer)

        input_layout = QHBoxLayout()
        self.return_input = QLineEdit(placeholderText="Expected Return %")
        self.risk_combo = QComboBox(); self.risk_combo.addItems(["Low", "Medium", "High"])
        input_layout.addWidget(self.return_input); input_layout.addWidget(self.risk_combo)
        input_layout.addWidget(self.button("Get Advice", self.show_investment_advice))
        vbox.addLayout(input_layout)

        self.investment_label = QLabel(""); self.investment_label.setWordWrap(True)
        vbox.addWidget(self.investment_label)

        main_layout.addWidget(invest_outer)
        self.setLayout(main_layout)

    # ---- Slots ----
    def upload_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if file:
            self.manager = PersonalFinanceManager(file)
            inc, exp, sav = self.manager.summary()
            self.summary_label.setText(f"💰 Income: ₹{inc}\n💸 Expenses: ₹{exp}\n💵 Savings: ₹{sav}")
            self.display_chart()

    def display_chart(self):
        if not self.manager: return
        if self.chart_canvas:
            self.chart_layout.removeWidget(self.chart_canvas); self.chart_canvas.setParent(None)
        self.chart_canvas = FigureCanvas(self.manager.expense_chart())
        self.chart_layout.addWidget(self.chart_canvas)

    def show_expense_advisor(self):
        if self.manager: self.review_text.setText(self.manager.expense_advisor_text())

    def show_investment_advice(self):
        if self.manager:
            try: expected = float(self.return_input.text())
            except: expected = 0
            _, _, sav = self.manager.summary()
            self.investment_label.setText(self.manager.investment_advisor(sav, self.risk_combo.currentText(), expected))


# -------------------- RUN --------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec_()) 
