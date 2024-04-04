import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QFileDialog, QWidget, QLabel, QAction, QTextEdit, QScrollArea, QPushButton, QDialog, QCheckBox, QMessageBox
from PyQt5.QtGui import QPixmap
import argparse
import matplotlib.pyplot as plt
from ecgprep import preprocess, read_ecg
import os
import pandas as pd
import time

current_dir = os.path.dirname(os.path.abspath(__file__))


class OptionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Display ECG Options")

        layout = QVBoxLayout()

        self.remove_baseline_checkbox = QCheckBox("Remove Baseline")
        layout.addWidget(self.remove_baseline_checkbox)

        self.use_all_leads_checkbox = QCheckBox("Use All Leads")
        layout.addWidget(self.use_all_leads_checkbox)

        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)


class EvaluateECGDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Evaluate ECG")

        layout = QVBoxLayout()

        self.upload_data_button = QPushButton("Upload Evaluation Data (HDF5)")
        self.upload_data_button.clicked.connect(self.upload_data_file)
        layout.addWidget(self.upload_data_button)

        self.upload_model_button = QPushButton("Upload Model (HDF5)")
        self.upload_model_button.clicked.connect(self.upload_model_file)
        layout.addWidget(self.upload_model_button)

        self.upload_annotations_button = QPushButton("Upload Annotations (CSV)")
        self.upload_annotations_button.clicked.connect(self.upload_annotations_file)
        layout.addWidget(self.upload_annotations_button)

        self.run_evaluation_button = QPushButton("Run Model Evaluation")
        self.run_evaluation_button.clicked.connect(self.run_evaluation)
        layout.addWidget(self.run_evaluation_button)

        self.setLayout(layout)

        self.data_path = None
        self.model_path = None
        self.annotations_path = None

    def upload_data_file(self):
        self.data_path, _ = QFileDialog.getOpenFileName(self, 'Upload Evaluation Data (HDF5)', '.', 'HDF5 Files (*.hdf5)')
        if self.data_path:
            print(f"Evaluation data file uploaded: {self.data_path}")

    def upload_model_file(self):
        self.model_path, _ = QFileDialog.getOpenFileName(self, 'Upload Model (HDF5)', '.', 'HDF5 Files (*.hdf5)')
        if self.model_path:
            print(f"Model file uploaded: {self.model_path}")

    def upload_annotations_file(self):
        self.annotations_path, _ = QFileDialog.getOpenFileName(self, 'Upload Annotations (CSV)', '.', 'CSV Files (*.csv)')
        if self.annotations_path:
            print(f"Annotations file uploaded: {self.annotations_path}")

    def run_evaluation(self):
        if self.data_path and self.model_path and self.annotations_path:
            command = f"python3 decode.py --tracings {self.data_path} --model {self.model_path} --annotations {self.annotations_path} --table_name ./tables/eval_tables"
            print("Running command:", command)
            try:
                subprocess.run(command, shell=True, check=True)
                print("Evaluation completed successfully.")
                # Generate and show the evaluation report popup
                report = self.generate_evaluation_report()
                self.show_evaluation_report_popup(report)
            except subprocess.CalledProcessError as e:
                print(f"Error running evaluation: {e}")
        else:
            print("Please upload all required files.")


    def generate_evaluation_report(self):
        csv_file_path = "./tables/eval_tables_scores.csv"
        prev_modification_time = None
        
        # Wait until the CSV file is fully written
        while True:
            current_modification_time = os.path.getmtime(csv_file_path)
            
            # If the modification time is the same as the previous time, the file is likely done being written
            if current_modification_time == prev_modification_time:
                break
            
            prev_modification_time = current_modification_time
            print("Waiting for the CSV file to be generated...")
            time.sleep(1)  # Wait for 1 second before checking again
    
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file_path)
            
            # Print the DataFrame to check its content
            print("DataFrame from CSV file:")
            print(df)
    
            # Convert DataFrame columns to numeric
            df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    
            # Generate the report
            report = f"{'':<10}{'Precision':<10}{'Recall':<10}{'Specificity':<12}{'F1 Score':<10}\n"
            for i in range(len(df)):
                report += f"{df.iloc[i, 0]:<10}{df.iloc[i, 1]:<10.3f}{df.iloc[i, 2]:<10.3f}{df.iloc[i, 3]:<12.3f}{df.iloc[i, 4]:<10.3f}\n"
    
            return report
        except Exception as e:
            print("Error generating report:", e)
            return None


    def show_evaluation_report_popup(self, report):
        msg = QMessageBox()
        msg.setWindowTitle("Evaluation Report")
        msg.setText(report)
        msg.exec_()


class ECGAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ECG Analyzer")
        self.setGeometry(100, 100, 1920, 1080)  # Set window size to 1920 x 1080

        main_layout = QHBoxLayout()

        # Left section for ECG plot with scroll area
        left_layout = QVBoxLayout()
        self.plot_label = QLabel()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)  # Allow scroll area to resize with the widget
        self.scroll_area.setWidget(self.plot_label)
        self.scroll_area.setFixedSize(1152, 566)  # Set fixed size for the scroll area
        left_layout.addWidget(self.scroll_area)
        main_layout.addLayout(left_layout)

        # Right section for text box
        right_layout = QVBoxLayout()
        self.status_label = QLabel("Upload .dat and .hea files to analyze.")
        right_layout.addWidget(self.status_label)

        self.status_textbox = QTextEdit()
        right_layout.addWidget(self.status_textbox)
        main_layout.addLayout(right_layout)

        self.central_widget = QWidget()
        self.central_widget.setLayout(main_layout)
        self.setCentralWidget(self.central_widget)

        self.dat_path = None
        self.hea_path = None
        self.records_path = None  # New attribute to store records.txt file path
        self.tracings_path = None  # New attribute to store tracings HDF5 file path
        self.annotations_path = None  # New attribute to store annotations CSV file path
        self.model_path = None  # New attribute to store model HDF5 file path

        self.init_toolbar()

    def init_toolbar(self):
        # File menu
        file_menu = self.menuBar().addMenu("File")

        upload_dat_action = QAction("Upload .dat File", self)
        upload_dat_action.triggered.connect(self.upload_dat_file)
        file_menu.addAction(upload_dat_action)

        upload_hea_action = QAction("Upload .hea File", self)
        upload_hea_action.triggered.connect(self.upload_hea_file)
        file_menu.addAction(upload_hea_action)

        upload_records_action = QAction("Upload records.txt File", self)
        upload_records_action.triggered.connect(self.upload_records_file)
        file_menu.addAction(upload_records_action)

        '''
        upload_tracings_action = QAction("Upload Tracings HDF5 File", self)
        upload_tracings_action.triggered.connect(self.upload_tracings_file)
        file_menu.addAction(upload_tracings_action)

        upload_annotations_action = QAction("Upload Annotations CSV File", self)
        upload_annotations_action.triggered.connect(self.upload_annotations_file)
        file_menu.addAction(upload_annotations_action)

        upload_model_action = QAction("Upload Model HDF5 File", self)
        upload_model_action.triggered.connect(self.upload_model_file)
        file_menu.addAction(upload_model_action)
        '''

        # Run menu
        run_menu = self.menuBar().addMenu("Run")

        display_ecg_action = QAction("Display ECG", self)
        display_ecg_action.triggered.connect(self.display_ecg)
        run_menu.addAction(display_ecg_action)

        evaluate_ecg_action = QAction("Evaluate Model", self)
        evaluate_ecg_action.triggered.connect(self.evaluate_ecg)
        run_menu.addAction(evaluate_ecg_action)

        generate_h5_action = QAction("Generate HDF5", self)
        generate_h5_action.triggered.connect(self.generate_h5)
        run_menu.addAction(generate_h5_action)

    def upload_dat_file(self):
        self.dat_path, _ = QFileDialog.getOpenFileName(self, 'Upload .dat File', '.', 'Data Files (*.dat)')
        if self.dat_path:
            self.status_label.setText(f".dat file uploaded: {self.dat_path}")
            self.update_status_textbox()

    def upload_hea_file(self):
        self.hea_path, _ = QFileDialog.getOpenFileName(self, 'Upload .hea File', '.', 'Header Files (*.hea)')
        if self.hea_path:
            self.status_label.setText(f".hea file uploaded: {self.hea_path}")
            self.update_status_textbox()

    def upload_records_file(self):
        self.records_path, _ = QFileDialog.getOpenFileName(self, 'Upload records.txt File', '.', 'Text Files (*.txt)')
        if self.records_path:
            self.status_label.setText(f"records.txt file uploaded: {self.records_path}")
            self.update_status_textbox()

    def upload_tracings_file(self):
        self.tracings_path, _ = QFileDialog.getOpenFileName(self, 'Upload Tracings HDF5 File', '.', 'HDF5 Files (*.h5)')
        if self.tracings_path:
            self.status_label.setText(f"Tracings HDF5 file uploaded: {self.tracings_path}")
            self.update_status_textbox()

    def upload_annotations_file(self):
        self.annotations_path, _ = QFileDialog.getOpenFileName(self, 'Upload Annotations CSV File', '.', 'CSV Files (*.csv)')
        if self.annotations_path:
            self.status_label.setText(f"Annotations CSV file uploaded: {self.annotations_path}")
            self.update_status_textbox()

    def upload_model_file(self):
        self.model_path, _ = QFileDialog.getOpenFileName(self, 'Upload Model HDF5 File', '.', 'HDF5 Files (*.h5)')
        if self.model_path:
            self.status_label.setText(f"Model HDF5 file uploaded: {self.model_path}")
            self.update_status_textbox()

    def display_ecg(self):
        options_dialog = OptionsDialog(self)
        if options_dialog.exec_() == QDialog.Accepted:
            remove_baseline = options_dialog.remove_baseline_checkbox.isChecked()
            use_all_leads = options_dialog.use_all_leads_checkbox.isChecked()

            if self.dat_path and self.hea_path:
                file_name_without_extension = os.path.splitext(self.dat_path)[0]
                command = f"python plot_from_ecg.py {file_name_without_extension}"
                if remove_baseline:
                    command += " --remove_baseline"
                if use_all_leads:
                    command += " --use_all_leads"
                command += " --save temp_plot.png"
                print("Running command:", command)  # Debugging print statement
                try:
                    subprocess.run(command, shell=True, check=True)
                    pixmap = QPixmap("./temp_plot.png")
                    self.plot_label.setPixmap(pixmap)
                    self.status_label.setText("ECG Displayed Successfully.")
                except subprocess.CalledProcessError as e:
                    self.status_label.setText(f"Error displaying ECG: {e}")
                finally:
                    # Delete the temporary PNG file
                    os.remove("./temp_plot.png")
            else:
                self.status_label.setText("Please upload both .dat and .hea files.")

    def generate_h5(self):
        if self.records_path:
            command = f"python generate_h5.py --scale 2 --use_all_leads --new_freq 400 --new_len 4096 {self.records_path} example_exams.h5"
            print("Running command:", command)  # Debugging print statement
            try:
                subprocess.run(command, shell=True, check=True)
                self.status_label.setText("HDF5 File Generated Successfully.")
            except subprocess.CalledProcessError as e:
                self.status_label.setText(f"Error generating HDF5 file: {e}")
        else:
            self.status_label.setText("Please upload records.txt file.")

    def update_status_textbox(self):
        status = self.status_label.text()
        self.status_textbox.append(status)

    def evaluate_ecg(self):
        evaluate_dialog = EvaluateECGDialog(self)
        evaluate_dialog.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ECGAnalyzer()
    window.show()
    sys.exit(app.exec_())
