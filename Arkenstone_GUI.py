"""
Created on Sun Dec 29 00:57:50 2024

@author: Siva Kumar Valluri
"""
import tkinter as tk
from tkinter import filedialog, messagebox

class Arkenstone_GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Arkenstone")
        self.master.configure(bg="#2e3b55")

        # Folder selection
        self.folder_path = tk.StringVar()
        tk.Label(master, text="Select Folder:", bg="#2e3b55", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.folder_entry = tk.Entry(master, textvariable=self.folder_path, width=40, state="readonly", font=("Helvetica", 10))
        self.folder_entry.grid(row=0, column=1, padx=10, pady=10)
        tk.Button(master, text="Browse", command=self.select_folder, bg="#4c5c79", fg="white", font=("Helvetica", 10, "bold"))\
            .grid(row=0, column=2, padx=10, pady=10)

        # Emission Type Selection
        self.emission_type = tk.StringVar(value="Optical Emissions")
        tk.Label(master, text="Emission Type:", bg="#2e3b55", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.emission_menu = tk.OptionMenu(master, self.emission_type, "Hyperspectral Emissions", "Optical Emissions", command=self.update_processing_style)
        self.emission_menu.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        # Frame count selection
        self.frame_count = tk.StringVar(value="8")
        tk.Label(master, text="Frame Count (4 or 8):", bg="#2e3b55", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=2, column=0, padx=10, pady=10, sticky="w")
        tk.OptionMenu(master, self.frame_count, "4", "8").grid(row=2, column=1, padx=10, pady=10, sticky="w")

        # Objective used selection
        self.objective_used = tk.StringVar(value="10")
        tk.Label(master, text="Objective Used (2, 4, 5, 10, 20):", bg="#2e3b55", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=3, column=0, padx=10, pady=10, sticky="w")
        tk.OptionMenu(master, self.objective_used, "2", "4", "5", "10", "20").grid(row=3, column=1, padx=10, pady=10, sticky="w")

        # Model selection
        self.model_selection = tk.StringVar(value="Segment Anything Model")
        tk.Label(master, text="Model to Segment Images:", bg="#2e3b55", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.model_menu = tk.OptionMenu(master, self.model_selection, "Random Forest Classifier", "Segment Anything Model")
        self.model_menu.grid(row=4, column=1, padx=10, pady=10, sticky="w")

        # Processing style selection
        self.processing_style = tk.StringVar(value="manual")
        tk.Label(master, text="Processing Style:", bg="#2e3b55", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=5, column=0, padx=10, pady=10, sticky="w")
        self.processing_menu = tk.OptionMenu(master, self.processing_style, "manual", "automated")
        self.processing_menu.grid(row=5, column=1, padx=10, pady=10, sticky="w")

        # Save images selection
        self.save_images = tk.StringVar(value="n")
        tk.Label(master, text="Save Images:", bg="#2e3b55", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=6, column=0, padx=10, pady=10, sticky="w")
        tk.OptionMenu(master, self.save_images, "y", "n").grid(row=6, column=1, padx=10, pady=10, sticky="w")

        # Submit button
        tk.Button(master, text="Generate", command=self.generate_frame_boxes, bg="#4c5c79", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=7, column=0, columnspan=3, pady=20)

    def update_processing_style(self, value):
        if value == "Hyperspectral Emissions":
            self.processing_style.set("automated")
            self.processing_menu.config(state="disabled")

            self.model_selection.set("Segment Anything Model")
            self.model_menu.config(state="disabled")
        else:
            self.processing_menu.config(state="normal")
            self.model_menu.config(state="normal")

    def select_folder(self):
        folder_selected = filedialog.askdirectory(title="Select Folder")
        if folder_selected:
            self.folder_path.set(folder_selected)
    
    def generate_frame_boxes(self):
        self.frame_data = {}
        frame_count = int(self.frame_count.get())
    
        frame_window = tk.Toplevel(self.master)
        frame_window.title("Frame Configuration")
        frame_window.configure(bg="#2e3b55")
    
        # Titles
        tk.Label(frame_window, text="Frame", bg="#2e3b55", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=0, column=0, padx=10, pady=10)
        tk.Label(frame_window, text="Delay (ns)", bg="#2e3b55", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=0, column=1, padx=10, pady=10)
        tk.Label(frame_window, text="Exposure (ns)", bg="#2e3b55", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=0, column=2, padx=10, pady=10)
        tk.Label(frame_window, text="Use", bg="#2e3b55", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=0, column=3, padx=10, pady=10)
        tk.Label(frame_window, text="R", bg="#2e3b55", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=0, column=4, padx=10, pady=10)
    
        self.entries = []
        for i in range(frame_count):
            tk.Label(frame_window, text=f"{i+1}", bg="#2e3b55", fg="white", font=("Helvetica", 10))\
                .grid(row=i+1, column=0, padx=10, pady=5)
    
            delay_entry = tk.Entry(frame_window, width=10, font=("Helvetica", 10))
            delay_entry.grid(row=i+1, column=1, padx=5, pady=5)
            
            exposure_entry = tk.Entry(frame_window, width=10, font=("Helvetica", 10))
            exposure_entry.insert(0, "50")  # Default exposure value
            exposure_entry.grid(row=i+1, column=2, padx=5, pady=5)
    
            use_frame_var = tk.BooleanVar(value=True)  # Default to checked
            use_frame_checkbox = tk.Checkbutton(frame_window, variable=use_frame_var, bg="#2e3b55")
            use_frame_checkbox.grid(row=i+1, column=3, padx=5, pady=5)
    
            red_filter_var = tk.BooleanVar(value=(i >= frame_count // 2))  # Tick last half of the rows
            filter_checkbox = tk.Checkbutton(frame_window, variable=red_filter_var, bg="#2e3b55")
            filter_checkbox.grid(row=i+1, column=4, padx=5, pady=5)
    
            self.entries.append((delay_entry, exposure_entry, use_frame_var, red_filter_var))
    
        tk.Button(frame_window, text="Submit", command=lambda: self.collect_frame_data(frame_window), 
                  bg="#4c5c79", fg="white", font=("Helvetica", 12, "bold"))\
            .grid(row=frame_count+1, column=0, columnspan=5, pady=20)



    def collect_frame_data(self, window):
        self.frame_data = []  # Reset frame data
        for delay_entry, exposure_entry, consider_var, filter_var in self.entries:
            consider = consider_var.get()  # Check if the 'Use' checkbox is selected
            filter_var = filter_var.get() 
            if consider:  # If checkbox is selected, validate input
                delay = delay_entry.get().strip()
                exposure = exposure_entry.get().strip()
    
                if not delay or not exposure:  # Check if values are missing
                    messagebox.showerror("Error", "Please enter valid numerical values for delays and exposures when 'Use' is checked.")
                    return
    
                try:
                    delay = int(delay)
                    exposure = int(exposure)
                except ValueError:  # Catch non-integer values
                    messagebox.showerror("Error", "Delays and Exposures must be numerical values.")
                    return
    
                self.frame_data.append((delay, exposure, consider, filter_var))  # Store valid data
            else:
                self.frame_data.append((None, None, False, False))  # Store (None, None, None) for unchecked rows
    
        window.destroy()  # Close the frame configuration window
        self.master.destroy() #Close master window
    
    def get_results(self):
        return (
            self.folder_path.get(),
            self.emission_type.get(),
            int(self.frame_count.get()),
            self.frame_data,  # Now includes (None, None, None) for unchecked rows
            int(self.objective_used.get()),
            self.model_selection.get(),
            self.processing_style.get(),
            self.save_images.get()
        )





