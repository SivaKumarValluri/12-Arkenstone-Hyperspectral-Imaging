# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:30:13 2025

@author: Siva Kumar Valluri
"""
import tkinter as tk
from tkinter import filedialog, messagebox

class Arkenstone_Hyperspectral_GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Arkenstone")
        self.master.configure(bg="#2e3b55")

        # Store result variables
        self.result = None  # Set to None initially, to avoid returning empty values incorrectly

        # Use calibration images selection
        self.use_calibration = tk.StringVar(value="y")
        tk.Label(master, text="Use new calibration?:", bg="#2e3b55", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.use_calibration_menu = tk.OptionMenu(master, self.use_calibration, "y", "n", command=self.toggle_calibration_inputs)
        self.use_calibration_menu.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        # Calibration lamp images selection
        self.calibration_lamp_path = tk.StringVar()
        tk.Label(master, text="Folder of Calibration Lamp Images:", bg="#2e3b55", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.calibration_entry = tk.Entry(master, textvariable=self.calibration_lamp_path, width=40, state="readonly", font=("Helvetica", 10))
        self.calibration_entry.grid(row=1, column=1, padx=10, pady=10)
        self.calibration_button = tk.Button(master, text="Browse", command=self.select_calibration_lamp, bg="#4c5c79", fg="white", font=("Helvetica", 10, "bold"))
        self.calibration_button.grid(row=1, column=2, padx=10, pady=10)

        # Dark field images selection
        self.dark_field_path = tk.StringVar()
        tk.Label(master, text="Folder of Dark Field Images:", bg="#2e3b55", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.dark_field_entry = tk.Entry(master, textvariable=self.dark_field_path, width=40, state="readonly", font=("Helvetica", 10))
        self.dark_field_entry.grid(row=2, column=1, padx=10, pady=10)
        self.dark_field_button = tk.Button(master, text="Browse", command=self.select_dark_field, bg="#4c5c79", fg="white", font=("Helvetica", 10, "bold"))
        self.dark_field_button.grid(row=2, column=2, padx=10, pady=10)

        # Horizontal line separator
        tk.Frame(master, bg="white", height=1).grid(row=3, column=0, columnspan=3, sticky="ew", padx=0, pady=10)


        # Analysis on Hotspots title
        tk.Label(master, text="Analysis on Hotspots:", bg="#2e3b55", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        
        # Hotspot analysis options
        self.hotspot_ignition = tk.StringVar(value="y")
        self.hotspot_temperature = tk.StringVar(value="y")
        self.hotspot_size = tk.StringVar(value="y")
        self.create_option_menu("Probability of ignition as a function of size:", self.hotspot_ignition, 5)
        self.create_option_menu("Temperature of hotspots as a function of size:", self.hotspot_temperature, 6)
        self.create_option_menu("Size of hotspots as a function of size:", self.hotspot_size, 7)

        # Horizontal line separator
        tk.Frame(master, bg="white", height=1).grid(row=8, column=0, columnspan=3, sticky="ew", padx=0, pady=10)

        # Analysis on Growth title
        tk.Label(master, text="Analysis on Growth:", bg="#2e3b55", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=9, column=0, columnspan=3, padx=10, pady=5, sticky="w")

        # Growth analysis options
        self.track_temperature = tk.StringVar(value="y")
        self.track_size = tk.StringVar(value="y")
        self.create_option_menu("Track temperature as a function of time:", self.track_temperature, 10)
        self.create_option_menu("Track size of reacting zone as a function of time:", self.track_size, 11)

        # Submit button
        tk.Button(master, text="Generate", command=self.get_results, bg="#4c5c79", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=12, column=0, columnspan=3, pady=20)

        # Initialize calibration input state
        self.toggle_calibration_inputs("y")

    def create_option_menu(self, text, variable, row):
        tk.Label(self.master, text=text, bg="#2e3b55", fg="white", font=("Helvetica", 12))\
            .grid(row=row, column=0, padx=10, pady=5, sticky="w")
        tk.OptionMenu(self.master, variable, "y", "n").grid(row=row, column=1, padx=10, pady=5, sticky="w")

    def select_calibration_lamp(self):
        folder_selected = filedialog.askdirectory(title="Select Calibration Lamp Images")
        if folder_selected:
            self.calibration_lamp_path.set(folder_selected)

    def select_dark_field(self):
        folder_selected = filedialog.askdirectory(title="Select Dark Field Images")
        if folder_selected:
            self.dark_field_path.set(folder_selected)

    def toggle_calibration_inputs(self, value):
        if value == "y":
            self.calibration_entry.configure(state="normal")
            self.calibration_button.configure(state="normal")
            self.dark_field_entry.configure(state="normal")
            self.dark_field_button.configure(state="normal")
        else:
            self.calibration_lamp_path.set("")
            self.dark_field_path.set("")
            self.calibration_entry.configure(state="disabled")
            self.calibration_button.configure(state="disabled")
            self.dark_field_entry.configure(state="disabled")
            self.dark_field_button.configure(state="disabled")

    def get_results(self):
        if self.use_calibration.get() == "y":
            if not self.calibration_lamp_path.get():
                messagebox.showerror("Error", "Please select a folder for Calibration Lamp Images.")
                return
            if not self.dark_field_path.get():
                messagebox.showerror("Error", "Please select a folder for Dark Field Images.")
                return
            
            self.result = {
                "calibration_lamp": self.calibration_lamp_path.get(),
                "dark_field": self.dark_field_path.get(),
            }
        else:
            self.result = {"calibration_lamp": "", "dark_field": ""}

        self.result.update({
            "hotspot_ignition": self.hotspot_ignition.get(),
            "hotspot_temperature": self.hotspot_temperature.get(),
            "hotspot_size": self.hotspot_size.get(),
            "track_temperature": self.track_temperature.get(),
            "track_size": self.track_size.get()
        })

        self.master.quit()
        self.master.destroy()
