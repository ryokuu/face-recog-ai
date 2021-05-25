import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
from cv2 import cv2
import numpy as np
import capture
import training
import recognize

#declaring tkinter
window = tk.Tk()
window.title("Door lock system")
window.geometry("550x300")
window.iconphoto(False, tk.PhotoImage(file='./icon/lock.png'))

#header
h1 = tk.Label(window, text="Door Lock Face Recognition", bg= "light blue", font =("Lucida Sans", 15, "bold"))
h1.place(x= 150, y= 10)
#logo
logo = Image.open("icon/logo.png")
resized = logo.resize((250, 250), Image.ANTIALIAS)
logo_new = ImageTk.PhotoImage(resized)
logo1 = tk.Label(window, image= logo_new, height="200")
logo1.pack(side="right")

#buttons
b1 = tk.Button(window,text="Record Data", padx = 10, pady=10, font=("Arial Bold",15),bg='#218B82',fg='black', command=capture.recorder)
b1.config( width= 15)
b1.place(x=20, y=60)

#b1.grid(column=0, row=2)
b2 = tk.Button(window,text="Train Dataset", padx = 10, pady=10,font=("Arial Bold",15),bg='#98D4BB',fg='black',command=training.trainer)
b2.config( width= 15)
b2.place(x=20, y=130)

#b2.grid(column=1, row=3)
b3 = tk.Button(window,text="Recognize Face", padx =10, pady=10,font=("Arial Bold",15),bg='#9AD9DB',fg='black',command=recognize.recognizer)
b3.config( width= 15)
b3.place(x=20, y=200)
#b3.grid(column=2, row=4)

window.mainloop()
