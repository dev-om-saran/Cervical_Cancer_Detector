import customtkinter as ctk
import tkinter as tk
from tkinter import Menu, Label, Button, Frame,messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import shutil

ctk.set_default_color_theme("blue") 

model = load_model("./data/model.h5")

# Create the main tkinter window
root = tk.Tk()
root.configure(bg= "#A9A9A9")
root.geometry("840x615")
root.title("AHRC Cervical Cancer Detector")

# Menu bar
menu_bar = Menu(root)

# File menu
file_menu = Menu(menu_bar, tearoff=0)
file_menu.add_command(label="New")
file_menu.add_command(label="Open")
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)
menu_bar.add_cascade(label="File", menu=file_menu)

# Edit menu
edit_menu = Menu(menu_bar, tearoff=0)
edit_menu.add_command(label="Cut")
edit_menu.add_command(label="Copy")
edit_menu.add_command(label="Paste")
menu_bar.add_cascade(label="Edit", menu=edit_menu)


def tip():
    tk.messagebox.showinfo("Tip", "Kindly do not maximize the windows for good user experience")

def user_manual():
    from os import startfile
    startfile('data\manual.txt')

# Help menu
help_menu = Menu(menu_bar, tearoff=0)
help_menu.add_command(label="Tip",command=tip)
help_menu.add_command(label="User_Manual",command=user_manual)
menu_bar.add_cascade(label="Help", menu=help_menu)

# Attach menu bar to the root window
root.config(menu=menu_bar)

# AHRC Logo
ahrc_org= Image.open('./data/logo.jpg')
ahrc_resized = ahrc_org.resize((280, 100))
# # Create a PhotoImage object for displaying in the Tkinter GUI
ahrc = ImageTk.PhotoImage(ahrc_resized)
ahrc_label = Label(image = ahrc, bg="#A9A9A9")
ahrc_label.grid(row=1, column=3, columnspan=3)


# Centered label

center_label = ctk.CTkLabel(root, text = "Welcome to AHRC Cervical Cancer Screening App", font = ("Helvetica", 20,"bold"), fg_color= "#87CEFA", text_color="#27408B",padx=10, pady=10,corner_radius=10)
center_label.grid(row=1, column=0, columnspan=3,pady =10)

original_image = None
photo = None
file_path = None
predicted_class = None
pic = None
cur_class = None
not_found = None
dyn_res = None

def open():
    global original_image
    global photo
    global image_label
    global file_path
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=(("all files", "*.*"),("png files","*.png"))) 
    if file_path:
        # Load the image with PIL
        original_image = Image.open(file_path)
        image_resized = original_image.resize((224, 224))
        # Create a PhotoImage object for displaying in the Tkinter GUI
        photo = ImageTk.PhotoImage(image_resized)
        
        # Display the image in the GUI
        image_label.config(image = photo)

def grey_convt():
    global original_image
    global grey_dis
    global image_label
    if original_image:
        image_resized = original_image.resize((224, 224))
        grey_image = image_resized.convert('L')
        grey_dis =  ImageTk.PhotoImage(grey_image)
        image_label.config(image = grey_dis)
    
labels = { 1:"Benign",  2:"Malignant", 3:"Suspicious" }

prediction_frame = ctk.CTkFrame(root, bg_color="#B0C4DE",border_width=2,border_color="black")
prediction_frame.grid(row=2, column=2, columnspan = 3,padx=20, pady=10, sticky="nsew")

prediction_label = ctk.CTkLabel(prediction_frame, text="The result will be displayed here", font=("Times New Roman", 20, "bold"), fg_color="#8B814C", text_color="white", padx = 10, pady = 10,corner_radius=10)
prediction_label.pack(pady=20)

validation_frame = ctk.CTkFrame(prediction_frame, bg_color="#B0C4DE")
validation_frame.pack(padx = 10, pady =10)

dyn_img = Image.open('./data/wait.png')
dyn_resized = dyn_img.resize((100, 100))
# # Create a PhotoImage object for displaying in the Tkinter GUI
dyn = ImageTk.PhotoImage(dyn_resized)
dyn_label = Label(validation_frame,image = dyn)
dyn_label.grid(row=0, column=0, columnspan=2)


def predict():
    global predicted_class, model, original_image, prediction_label, pic,file_path
    
    if original_image is not None:

        # Resize and preprocess the image
        image_grey = original_image.convert('L')
        image_resized = image_grey.resize((299, 299))
        image_3 = np.stack((image_resized,) * 3, axis=-1)
        image_array = np.array(image_3)
        image_normalized = image_array / 255.0
        image_expanded = np.expand_dims(image_normalized, axis=0)

        # Make a prediction using the model
        prediction = model.predict(image_expanded)
        predicted_class = np.argmax(prediction)

        # Determine label and color based on prediction
        if predicted_class == 0:
            label_text = "Predicted class is Benign"
            label_color = "dark green"
            label_pic = Image.open("./data/ben.jpg")
        elif predicted_class == 1:
            label_text = "Predicted class is Malignant"
            label_color = "red"
            label_pic = Image.open("./data/mal.jpg")
        elif predicted_class == 2:
            label_text = "Predicted class is Suspicious"
            label_color = "#EE7942"
            label_pic = Image.open("./data/sus.png")

        # Display prediction result
        prediction_label.configure(text = label_text, fg_color = label_color, text_color = "white")
        prediction_label.pack(pady=10)
        
        pic_resized = label_pic.resize((100, 100))
        # # Create a PhotoImage object for displaying in the Tkinter GUI
        pic = ImageTk.PhotoImage(pic_resized)
        dyn_label = Label(validation_frame,image = pic)
        dyn_label.grid(row=0, column=0, columnspan=2)
        file_name = os.path.split(file_path)[-1]
        dest_dir = f"./predictions/{labels[predicted_class+1]}/predicted/"
        dest_path = os.path.join(dest_dir, file_name)
        shutil.copyfile(file_path, dest_path)
        
        

    else:
        tk.messagebox.showwarning("Warning", "No image loaded. Please upload an image first.")
        
def validate():
    global file_path, validation_frame,predicted_class, dyn_label, not_found
    
    if original_image is not None:
        for widget in validation_frame.winfo_children():
            widget.destroy()
        
        
        def find_file(start_dir, target_file):
            for dirpath, _, filenames in os.walk(start_dir):
                if target_file in filenames:
                    return os.path.join(dirpath, target_file)
            return None

        # Example usage:
        start_directory = ".\\predictions"  # Replace with your start directory
        file_to_find = os.path.split(file_path)[-1]  # Replace with your target file name

        found_file = find_file(start_directory, file_to_find)

        if found_file:
            parts = found_file.split('\\')

            # Reverse the list of path parts
            reversed_parts = list(reversed(parts))

            # print(reversed_parts)
            # Access the third element in the reversed list
            cur_class = reversed_parts[2]
            
            prediction_label.configure(text = "The predicted class was " + cur_class, font=("Times New Roman", 20, "bold"), fg_color="#8B814C", text_color="white", padx = 10, pady = 10)
            
            select_label = ctk.CTkLabel(validation_frame, text="Select the Correct Class:", font=("Comic Sans MS", 14),
                                        padx=10, pady=10, corner_radius=10)
            select_label.grid(row=0, column=0)
            
            # Disable button for predicted class
            button_states = {0: "Benign", 1: "Malignant", 2: "Suspicious"}
            ctr=0
            for index, label in button_states.items():
                but_state = None
                if label == cur_class:
                    but_state = "disabled"
                else:
                    but_state = "normal"
                button = ctk.CTkButton(validation_frame, text=label, command=lambda label=label: change_prediction(label,found_file,cur_class),
                                    fg_color="#CD8C95", border_color="black", border_width=2, text_color="white",
                                    corner_radius=5, border_spacing=2,state = but_state)
                button.grid(row = ctr,column=1,pady=5)
                ctr = ctr+1
            
            
        else:
            prediction_label.configure(text = "No prediction made for this image yet")
            for widget in validation_frame.winfo_children():
                widget.destroy()
            temp = Image.open("./data/not_found.png")
            temp_resized = temp.resize((100, 100))
            # # Create a PhotoImage object for displaying in the Tkinter GUI
            not_found = ImageTk.PhotoImage(temp_resized)
            dyn_label = Label(validation_frame,image = not_found)
            dyn_label.grid(row=0, column=0, columnspan=2)
    else:
        tk.messagebox.showwarning("Warning", "No image loaded. Please upload an image first.")
  
def change_prediction(label,found_file,cur_class):
    global file_path
    # print(label +' , ' + cur_class)
    
    if cur_class == label:
        # print("no need")
        reset()
        return

    # Move the file to the correct directory based on the selected class
    file_name = os.path.split(file_path)[-1]
    dest_dir = f"./predictions/{label}/validated/"
    dest_path = os.path.join(dest_dir, file_name)
    shutil.copyfile(file_path, dest_path)
    os.remove(found_file)
    reset()
           
def reset():
    global image_label, pred,logo, prediction_frame, dyn_res, original_image
    image_label.config(image =logo)
    prediction_label.configure( text = "The result will be displayed here", fg_color="#8B814C")
    for widget in validation_frame.winfo_children():
         widget.destroy()
    dyn_img = Image.open('./data/wait.png')
    dyn_resized = dyn_img.resize((100, 100))
    # # Create a PhotoImage object for displaying in the Tkinter GUI
    dyn_res = ImageTk.PhotoImage(dyn_resized)
    dyn_label = Label(validation_frame,image = dyn_res)
    dyn_label.grid(row=0, column=0, columnspan=2)
    
    original_image = None
    

## Image placeholder
logo_org= Image.open('./data/default.png')
logo_resized = logo_org.resize((224, 224))
# # Create a PhotoImage object for displaying in the Tkinter GUI
logo = ImageTk.PhotoImage(logo_resized)
image_label = Label(image = logo)
image_label.grid(row=2, column=0, columnspan=2,pady =10)

# Buttons
button1 = ctk.CTkButton(root, text="Upload New Image",command=open, font=("Comic Sans MS",13),fg_color="#27408B" ,border_color="black",border_width=2 , text_color= "white", corner_radius=10, border_spacing= 10)
button1.grid(row=4, column=0,padx=5)

button3 = ctk.CTkButton(root, text="Predict Now", command=predict,font=("Comic Sans MS",13),fg_color="#27408B" ,border_color="black",border_width=2 , text_color= "white", corner_radius=10, border_spacing= 10)
button3.grid(row=4, column=2,padx=5)

button2 = ctk.CTkButton(root, text="Validate prediction",command = validate,font=("Comic Sans MS",13),fg_color="#27408B" ,border_color="black",border_width=2 , text_color= "white", corner_radius=10, border_spacing= 10)
button2.grid(row=4, column=4,padx=5)

button2 = ctk.CTkButton(root, text="View greyscale image",command = grey_convt,font=("Comic Sans MS",13),fg_color="#27408B" ,border_color="black",border_width=2 , text_color= "white", corner_radius=10, border_spacing= 10)
button2.grid(row=5, column=1,padx=5)


button4 = ctk.CTkButton(root,text = "Make another prediction", command= reset,font=("Comic Sans MS",13),fg_color="#27408B" ,border_color="black",border_width=2 , text_color= "white", corner_radius=10, border_spacing= 10)
button4.grid(row=5, column=3,padx=5)

#result displayed
pred = ctk.CTkLabel(root, text = "Upload an Image and click Predict Now / Validate Prediction", font = ("Times New Roman", 18),fg_color="#2E8B57", corner_radius= 10,text_color="white",padx=10,pady=10)
pred.grid(row=3, column=0, columnspan=5,pady =20)

# Frame for plots

from tkinter import ttk
import os 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

bp = len(os.listdir('./predictions/Benign/predicted'))
bnp = len(os.listdir('./predictions/Benign/validated'))
mp = len(os.listdir('./predictions/Malignant/predicted'))
mnp = len(os.listdir('./predictions/Malignant/validated'))
sp = len(os.listdir('./predictions/Suspicious/predicted'))
snp = len(os.listdir('./predictions/Suspicious/validated'))

def plot():
    # Calculate counts from directories
    bp = len(os.listdir('./predictions/Benign/predicted'))
    bnp = len(os.listdir('./predictions/Benign/validated'))
    sp = len(os.listdir('./predictions/Suspicious/predicted'))
    snp = len(os.listdir('./predictions/Suspicious/validated'))
    mp = len(os.listdir('./predictions/Malignant/predicted'))
    mnp = len(os.listdir('./predictions/Malignant/validated'))
    
    # Check if any data is available
    if bp + bnp  == 0 or sp + snp ==0 or mp + mnp==0 :
        messagebox.showwarning("Message", "Not Enough data available to plot")
        return
    
    # Create a new top-level window
    top = tk.Toplevel()
    top.title("Validation Statistics")
    
    # Frame to hold plots
    plot_frame = ttk.Frame(top)
    plot_frame.pack(padx=10, pady=10)
    
    # Plot Benign vs Suspicious in the first row using pie charts
    fig1, ax1 = plt.subplots(figsize=(4, 2.5))
    ax1.pie([bp, bnp], labels=['Predicted', 'Validated'], autopct='%1.1f%%', startangle=90)
    ax1.set_title('Benign Cases')
    fig2, ax2 = plt.subplots(figsize=(4, 2.5))
    ax2.pie([sp, snp], labels=['Predicted', 'Validated'], autopct='%1.1f%%', startangle=90)
    ax2.set_title('Suspicious Cases')
    
    # Plot Malignant in the second row using pie chart
    fig3, ax3 = plt.subplots(figsize=(4, 2.5))
    ax3.pie([mp, mnp], labels=['Predicted', 'Validated'], autopct='%1.1f%%', startangle=90)
    ax3.set_title('Malignant Cases')
    
    fig4, ax4 = plt.subplots(figsize=(4, 2.5))
    ax4.pie([bp + bnp, mp + mnp, sp + snp], labels=['Benign', 'Malignant','Suspicious'], autopct='%1.1f%%', startangle=90)
    ax4.set_title('Cases Distribution')
    
    # Create Tkinter canvases
    canvas1 = FigureCanvasTkAgg(fig1, master=plot_frame)
    canvas1.draw()

    canvas1.get_tk_widget().grid(row=0, column=0,pady = 5,padx =5)

    # Example for canvas2:
    canvas2 = FigureCanvasTkAgg(fig2, master=plot_frame)
    canvas2.draw()

    canvas2.get_tk_widget().grid(row=0, column=1,pady = 5,padx =5)

    # Example for canvas3:
    canvas3 = FigureCanvasTkAgg(fig3, master=plot_frame)
    canvas3.draw()

    canvas3.get_tk_widget().grid(row=1, column=0,pady = 5,padx =5)
    
    canvas4 = FigureCanvasTkAgg(fig4, master=plot_frame)
    canvas4.draw()

    canvas4.get_tk_widget().grid(row=1, column=1,pady = 5,padx =5)
    
# Function to plot distribution of all cases and overall predictions using pie charts
def plot_all():
    # Calculate counts from directories
    bp_predicted = len(os.listdir('./predictions/Benign/predicted'))
    bp_not_predicted = len(os.listdir('./predictions/Benign/validated'))
    sp_predicted = len(os.listdir('./predictions/Suspicious/predicted'))
    sp_not_predicted = len(os.listdir('./predictions/Suspicious/validated'))
    mp_predicted = len(os.listdir('./predictions/Malignant/predicted'))
    mp_not_predicted = len(os.listdir('./predictions/Malignant/validated'))
    
    # Check if any data is available
    if bp_predicted + bp_not_predicted + sp_predicted + sp_not_predicted + mp_predicted + mp_not_predicted == 0:
        messagebox.showwarning("Message", "Not Enough data available to plot")
        return
    
    # Create a new top-level window for all plots
    top = tk.Toplevel()
    top.title("Distribution and Overall Predictions")
    
    # Frame to hold plots
    plot_frame = ttk.Frame(top)
    plot_frame.pack(padx=10, pady=10)
    
    # Plot distribution of all cases using pie chart
    fig1, ax1 = plt.subplots(figsize=(4, 2.5))
    ax1.pie([bp_predicted + bp_not_predicted, sp_predicted + sp_not_predicted, mp_predicted + mp_not_predicted],
            labels=['Benign', 'Suspicious', 'Malignant'], autopct='%1.1f%%', startangle=90)
    ax1.set_title('Distribution of All Cases')
    
    # Plot overall predictions using pie chart
    fig2, ax2 = plt.subplots(figsize=(4, 2.5))
    ax2.pie([bp_predicted + sp_predicted + mp_predicted, bp_not_predicted + sp_not_predicted + mp_not_predicted],
            labels=['Correct', 'Wrong'], autopct='%1.1f%%', startangle=90)
    ax2.set_title('Overall Predictions')
    
    # Create Tkinter canvases
    canvas1 = FigureCanvasTkAgg(fig1, master=plot_frame)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side=tk.LEFT, padx=10, pady=10)
    
    canvas2 = FigureCanvasTkAgg(fig2, master=plot_frame)
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=tk.LEFT, padx=10, pady=10)




plot_button1 = ctk.CTkButton(root,text = "View Model Validation Statistics", fg_color="#8B475D", text_color ="white",font = ("Comic Sans MS", 14), border_color="black", border_width=2, border_spacing=10, hover_color="#FA8072",command=plot)
plot_button1.grid(row=6, column=0,columnspan=6,pady =20,padx = 10)


from datetime import datetime

def update_time():
    if root:
        current_time = datetime.now().strftime('Developed by Om @ AHRC, IITBBS                                                            ' + '%Y-%m-%d ' + '      ''%H:%M:%S' + '    ')
        status_bar.configure(text=current_time)
        root.after(1000, update_time)  # Update every 1000ms (1 second)


#status baar
status_bar = ctk.CTkLabel(root, text="Status: Ready", anchor=tk.E,fg_color="#515151", text_color="white",padx = 3, pady = 3)
status_bar.grid(row=7, column=0, columnspan=5, sticky=tk.E + tk.W)

update_time()

# Function to handle window close event
def on_closing():
    root.quit()

# Bind window close event to on_closing function
root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the tkinter main loop
root.mainloop()
