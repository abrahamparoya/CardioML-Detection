import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QFileDialog, QWidget, QLabel, QAction, QTextEdit, QScrollArea, QPushButton, QDialog, QCheckBox
from PyQt5.QtGui import QPixmap
import argparse
import matplotlib.pyplot as plt
from ecgprep import preprocess, read_ecg
import os

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

        upload_tracings_action = QAction("Upload Tracings HDF5 File", self)
        upload_tracings_action.triggered.connect(self.upload_tracings_file)
        file_menu.addAction(upload_tracings_action)

        upload_annotations_action = QAction("Upload Annotations CSV File", self)
        upload_annotations_action.triggered.connect(self.upload_annotations_file)
        file_menu.addAction(upload_annotations_action)

        upload_model_action = QAction("Upload Model HDF5 File", self)
        upload_model_action.triggered.connect(self.upload_model_file)
        file_menu.addAction(upload_model_action)

        # Run menu
        run_menu = self.menuBar().addMenu("Run")

        display_ecg_action = QAction("Display ECG", self)
        display_ecg_action.triggered.connect(self.display_ecg)
        run_menu.addAction(display_ecg_action)

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ECGAnalyzer()
    window.show()
    sys.exit(app.exec_())


