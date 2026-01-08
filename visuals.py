import pyqtgraph as pg
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView

def create_sentiment_bar_chart(sentiment_counts: dict) -> pg.PlotWidget:
    """Creates a bar chart for sentiment distribution."""
    sentiments = list(sentiment_counts.keys())
    counts = list(sentiment_counts.values())

    # Define colors for each sentiment
    sentiment_colors = {
        "Very Positive": "g",
        "Positive": "#5cb85c",
        "Neutral": "y",
        "Negative": "#d9534f",
        "Very Negative": "r"
    }
    colors = [sentiment_colors.get(s, 'b') for s in sentiments]

    x_ticks = list(range(len(sentiments)))
    bargraph = pg.BarGraphItem(x=x_ticks, height=counts, width=0.6, brushes=colors)

    plot_widget = pg.PlotWidget()
    plot_widget.addItem(bargraph)
    plot_widget.setTitle("Overall Sentiment Distribution")
    plot_widget.setLabel('left', 'Number of Texts')

    # Set custom x-axis ticks
    ax = plot_widget.getAxis('bottom')
    ticks = [list(zip(x_ticks, sentiments))]
    ax.setTicks(ticks)

    return plot_widget

def create_language_bar_chart(language_counts: dict) -> pg.PlotWidget:
    """Creates a bar chart for language distribution."""
    languages = list(language_counts.keys())
    counts = list(language_counts.values())

    x_ticks = list(range(len(languages)))
    bargraph = pg.BarGraphItem(x=x_ticks, height=counts, width=0.6, brush='b')

    plot_widget = pg.PlotWidget()
    plot_widget.addItem(bargraph)
    plot_widget.setTitle("Language Distribution")
    plot_widget.setLabel('left', 'Number of Texts')

    ax = plot_widget.getAxis('bottom')
    ticks = [list(zip(x_ticks, languages))]
    ax.setTicks(ticks)

    return plot_widget

def create_keywords_table(keywords: dict) -> QTableWidget:
    """Creates a table widget for top keywords."""
    table = QTableWidget()
    sentiments = list(keywords.keys())
    table.setColumnCount(len(sentiments))
    table.setHorizontalHeaderLabels(sentiments)

    # Find the max number of keywords for any sentiment to set row count
    max_keywords = 0
    if keywords:
        max_keywords = max(len(v) for v in keywords.values())
    table.setRowCount(max_keywords)

    for col_idx, sentiment in enumerate(sentiments):
        for row_idx, keyword in enumerate(keywords[sentiment]):
            table.setItem(row_idx, col_idx, QTableWidgetItem(keyword))

    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    return table
