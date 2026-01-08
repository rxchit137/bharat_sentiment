import sys
import json
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QTextEdit, QProgressBar, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtGui import QFont
import time
from PySide6.QtCore import Qt, QThread, Signal
import multiprocessing

from sentiment_logic import analyze
from batch_handler import load_texts_from_file, generate_report, compare_reports
from worker_init import init_worker, analyze_text_worker
from visuals import create_sentiment_bar_chart, create_language_bar_chart, create_keywords_table

class BatchProcessingThread(QThread):
    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, filepath, text_column):
        super().__init__()
        self.filepath = filepath
        self.text_column = text_column
        self.is_running = True

    def run(self):
        try:
            texts = load_texts_from_file(self.filepath, self.text_column)
            total_texts = len(texts)

            pool = multiprocessing.Pool(initializer=init_worker)
            async_results = [pool.apply_async(analyze_text_worker, args=(text,)) for text in texts]

            completed_count = 0
            while completed_count < total_texts and self.is_running:
                completed_count = sum(1 for res in async_results if res.ready())
                progress_value = int((completed_count / total_texts) * 100)
                self.progress.emit(progress_value)
                time.sleep(0.5)

            if not self.is_running:
                pool.terminate()
                pool.join()
                return

            pool.close()
            pool.join()

            results = [res.get() for res in async_results]
            report_path = generate_report(self.filepath, texts, results)
            self.finished.emit(f"Batch processing complete. Report saved to: {report_path}")
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.is_running = False


class BatchAnalysisTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        # File selection
        self.file_label = QLabel("Select a CSV or XLSX file:")
        self.file_path_entry = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.file_path_entry)
        file_layout.addWidget(self.browse_button)

        # Text column selection
        self.column_label = QLabel("Enter the name of the column containing the text:")
        self.column_name_entry = QLineEdit()

        # Action button
        self.run_batch_button = QPushButton("Run Batch Analysis")

        # Progress and results
        self.progress_bar = QProgressBar()
        self.results_label = QLabel("")
        self.results_layout = QVBoxLayout()
        self.results_widget = QWidget()
        self.results_widget.setLayout(self.results_layout)

        layout.addWidget(self.file_label)
        layout.addLayout(file_layout)
        layout.addWidget(self.column_label)
        layout.addWidget(self.column_name_entry)
        layout.addWidget(self.run_batch_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.results_label)
        layout.addWidget(self.results_widget)

        self.browse_button.clicked.connect(self.browse_file)
        self.run_batch_button.clicked.connect(self.run_batch_processing)

    def browse_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select a File", "", "CSV files (*.csv);;Excel files (*.xlsx)")
        if filepath:
            self.file_path_entry.setText(filepath)

    def run_batch_processing(self):
        filepath = self.file_path_entry.text()
        text_column = self.column_name_entry.text()

        if not filepath or not text_column:
            QMessageBox.warning(self, "Input Error", "Please select a file and specify the text column.")
            return

        self.progress_bar.setValue(0)
        self.results_label.setText("Processing...")
        self.run_batch_button.setEnabled(False)

        self.thread = BatchProcessingThread(filepath, text_column)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_batch_finished)
        self.thread.error.connect(self.on_batch_error)
        self.thread.start()

    def on_batch_finished(self, report_path):
        # Clear previous results
        for i in reversed(range(self.results_layout.count())):
            self.results_layout.itemAt(i).widget().setParent(None)

        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)

            # Display visualizations
            self.results_layout.addWidget(QLabel(f"Report for: {report_data['source_file']}"))
            self.results_layout.addWidget(create_sentiment_bar_chart(report_data["overall_sentiment_counts"]))
            self.results_layout.addWidget(create_language_bar_chart(report_data["language_breakdown"]))
            self.results_layout.addWidget(QLabel("Top Keywords:"))
            self.results_layout.addWidget(create_keywords_table(report_data["top_keywords"]))
        except Exception as e:
            self.results_layout.addWidget(QLabel(f"Error displaying report: {e}"))

        self.run_batch_button.setEnabled(True)

    def on_batch_error(self, message):
        # Clear previous results
        for i in reversed(range(self.results_layout.count())):
            self.results_layout.itemAt(i).widget().setParent(None)
        self.results_layout.addWidget(QLabel(f"Error: {message}"))
        self.run_batch_button.setEnabled(True)

class AnalysisModel:
    def __init__(self):
        self.tokenizer = None
        self.session = None
        self._load_model()

    def _load_model(self):
        import os
        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_dir = "./onnx_model"
        model_path = os.path.join(model_dir, "model_quantized.onnx")

        if os.path.exists(model_path):
            self.session = ort.InferenceSession(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        else:
            print("ONNX model not found. Single text analysis will not be available.")

    def analyze(self, text: str) -> dict:
        if not self.session or not self.tokenizer:
            return {"language": "unknown", "sentiment": "ONNX model not loaded."}
        return analyze(text, self.tokenizer, self.session)


class SingleAnalysisTab(QWidget):
    def __init__(self, analysis_model: AnalysisModel):
        super().__init__()
        self.analysis_model = analysis_model
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        self.label = QLabel("Enter Hindi or English text for sentiment analysis:")
        self.text_entry = QLineEdit()
        self.analyze_button = QPushButton("Analyze")
        self.result_label = QLabel("Sentiment: ")

        layout.addWidget(self.label)
        layout.addWidget(self.text_entry)
        layout.addWidget(self.analyze_button)
        layout.addWidget(self.result_label)

        self.analyze_button.clicked.connect(self.perform_analysis)

    def perform_analysis(self):
        text = self.text_entry.text()
        if not text:
            self.result_label.setText("Sentiment: Please enter some text.")
            return

        result = self.analysis_model.analyze(text)
        self.result_label.setText(f"Language: {result['language']}, Sentiment: {result['sentiment']}")

class ReportComparisonThread(QThread):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, report_path1, report_path2):
        super().__init__()
        self.report_path1 = report_path1
        self.report_path2 = report_path2

    def run(self):
        try:
            result_path = compare_reports(self.report_path1, self.report_path2)
            with open(result_path, 'r', encoding='utf-8') as f:
                self.finished.emit(f.read())
        except Exception as e:
            self.error.emit(str(e))

class ReportComparisonTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        # File selection
        self.file1_label = QLabel("Select the first report file (.json):")
        self.file1_path_entry = QLineEdit()
        self.browse1_button = QPushButton("Browse...")
        file1_layout = QHBoxLayout()
        file1_layout.addWidget(self.file1_path_entry)
        file1_layout.addWidget(self.browse1_button)

        self.file2_label = QLabel("Select the second report file (.json):")
        self.file2_path_entry = QLineEdit()
        self.browse2_button = QPushButton("Browse...")
        file2_layout = QHBoxLayout()
        file2_layout.addWidget(self.file2_path_entry)
        file2_layout.addWidget(self.browse2_button)

        # Action button
        self.run_comparison_button = QPushButton("Run Comparison")

        # Results display area
        self.results_layout = QVBoxLayout()
        self.results_widget = QWidget()
        self.results_widget.setLayout(self.results_layout)

        layout.addWidget(self.file1_label)
        layout.addLayout(file1_layout)
        layout.addWidget(self.file2_label)
        layout.addLayout(file2_layout)
        layout.addWidget(self.run_comparison_button)
        layout.addWidget(self.results_widget)

        self.browse1_button.clicked.connect(lambda: self.browse_file(self.file1_path_entry))
        self.browse2_button.clicked.connect(lambda: self.browse_file(self.file2_path_entry))
        self.run_comparison_button.clicked.connect(self.run_comparison)

    def browse_file(self, path_entry_widget):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select a Report File", "", "JSON files (*.json)")
        if filepath:
            path_entry_widget.setText(filepath)

    def run_comparison(self):
        report_path1 = self.file1_path_entry.text()
        report_path2 = self.file2_path_entry.text()

        if not report_path1 or not report_path2:
            QMessageBox.warning(self, "Input Error", "Please select two report files.")
            return

        # Clear previous results
        for i in reversed(range(self.results_layout.count())):
            self.results_layout.itemAt(i).widget().setParent(None)
        self.results_layout.addWidget(QLabel("Comparing reports..."))

        self.run_comparison_button.setEnabled(False)

        self.thread = ReportComparisonThread(report_path1, report_path2)
        self.thread.finished.connect(self.on_comparison_finished)
        self.thread.error.connect(self.on_comparison_error)
        self.thread.start()

    def on_comparison_finished(self, result_json):
        # Clear previous results
        for i in reversed(range(self.results_layout.count())):
            self.results_layout.itemAt(i).widget().setParent(None)

        data = json.loads(result_json)

        # Summary
        summary_text = f"Comparison between {data['source_files'][0]} and {data['source_files'][1]}.\n"
        summary_text += f"Total texts: {data['total_texts_comparison']['report1']} vs {data['total_texts_comparison']['report2']} (Diff: {data['total_texts_comparison']['difference']})"
        summary_box = QTextEdit(summary_text)
        summary_box.setReadOnly(True)
        self.results_layout.addWidget(summary_box)

        # Sentiment Comparison Table
        self.results_layout.addWidget(QLabel("Overall Sentiment Comparison:"))
        sentiment_table = self.create_comparison_table(data["overall_sentiment_comparison"])
        self.results_layout.addWidget(sentiment_table)

        # Language Comparison Table
        self.results_layout.addWidget(QLabel("Language Breakdown Comparison:"))
        language_table = self.create_comparison_table(data["language_breakdown_comparison"])
        self.results_layout.addWidget(language_table)

        # Keyword Comparison Table
        self.results_layout.addWidget(QLabel("Keyword Comparison:"))
        keyword_table = self.create_keyword_comparison_table(data["keyword_comparison"])
        self.results_layout.addWidget(keyword_table)

        self.run_comparison_button.setEnabled(True)

    def create_comparison_table(self, data):
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Metric", "Report 1", "Report 2", "Difference"])
        table.setRowCount(len(data))
        for row, (metric, values) in enumerate(data.items()):
            table.setItem(row, 0, QTableWidgetItem(metric))
            table.setItem(row, 1, QTableWidgetItem(str(values["report1"])))
            table.setItem(row, 2, QTableWidgetItem(str(values["report2"])))
            table.setItem(row, 3, QTableWidgetItem(str(values["difference"])))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        return table

    def create_keyword_comparison_table(self, data):
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Sentiment", "Common Keywords", "Unique to Report 1", "Unique to Report 2"])
        table.setRowCount(len(data))
        for row, (sentiment, values) in enumerate(data.items()):
            table.setItem(row, 0, QTableWidgetItem(sentiment))
            table.setItem(row, 1, QTableWidgetItem(", ".join(values["common_keywords"])))
            table.setItem(row, 2, QTableWidgetItem(", ".join(values["unique_to_report1"])))
            table.setItem(row, 3, QTableWidgetItem(", ".join(values["unique_to_report2"])))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        return table

    def on_comparison_error(self, message):
        # Clear previous results
        for i in reversed(range(self.results_layout.count())):
            self.results_layout.itemAt(i).widget().setParent(None)
        self.results_layout.addWidget(QLabel(f"Error: {message}"))
        self.run_comparison_button.setEnabled(True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Sentiment Analysis Tool")
        self.setGeometry(100, 100, 800, 600)

        # Apply a modern stylesheet
        self.setStyleSheet("""
            QWidget {
                font-size: 14px;
                font-family: Arial, sans-serif;
            }
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTabWidget::pane {
                border-top: 2px solid #C2C7CB;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QLineEdit, QTextEdit {
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 8px;
            }
            QLabel {
                color: #333;
            }
        """)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create and load the model for single analysis
        self.analysis_model = AnalysisModel()

        self.single_analysis_tab = SingleAnalysisTab(self.analysis_model)
        self.tabs.addTab(self.single_analysis_tab, "Single Text Analysis")

        self.batch_tab = BatchAnalysisTab()
        self.tabs.addTab(self.batch_tab, "Batch Analysis")

        self.comparison_tab = ReportComparisonTab()
        self.tabs.addTab(self.comparison_tab, "Report Comparison")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
