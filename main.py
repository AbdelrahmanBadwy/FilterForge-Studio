import tkinter as tk
from tkinter import ttk
from gui.app import FilterForgeApp

if __name__ == "__main__":
    root = tk.Tk()
    root.title("FilterForge Studio")
    app = FilterForgeApp(root)
    root.mainloop() 