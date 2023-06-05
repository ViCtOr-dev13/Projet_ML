import customtkinter as ctk
from PIL import Image, ImageTk
import tkinter as tk

class MainApplication(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry("900x506")
        self.resizable(0, 0)
        self.title('PROJET CLUSTERING')

        self.bg_images = {
            'start': ImageTk.PhotoImage(Image.open("image/VÉRIFIER VOTRE URL.png")),
            'result1': ImageTk.PhotoImage(Image.open("image/URL VÉRIFIÉ1.png")),
            'result3': ImageTk.PhotoImage(Image.open("image/URL VÉRIFIÉ3.png")),
            'detail': ImageTk.PhotoImage(Image.open("image/DÉTAILS.png"))
        }

        # container pour les frames
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Initialize empty dictionary to store frames
        self.frames = {}

        for F in (StartPage, ResultPage1, ResultPage3, TreatmentDetailsPage):
            frame = F(container, self, self.bg_images)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, page_name):
        
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller, bg_images):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.fond = ctk.CTkLabel(master=self, image=bg_images['start'])
        self.fond.place(x=0, y=0)

        self.url1produit = ctk.CTkEntry(master=self, width=400, bg_color=("#000000", "#000000"), fg_color=("#FFFFFF", "#FFFFFF"))
        self.url1produit.insert(0, "ENTRER UN URL")
        self.url1produit.bind("<FocusIn>", self.clear_placeholder)
        self.url1produit.place(x=250, y=230)

        self.valider = ctk.CTkButton(master=self, text='VALIDER', command=self.chargement, bg_color=("#000000", "#000000"), fg_color=("#000000", "#000000"))
        self.valider.place(x=380, y=300)

    def chargement(self):
        url1 = self.url1produit.get()
        print(url1)

        if url1 == "benign":
            self.controller.show_frame(ResultPage1)
        else:
            self.controller.show_frame(ResultPage3)

    def clear_placeholder(self, event):
        if self.url1produit.get() == 'ENTRER UN URL':
            self.url1produit.delete(0, 'end')


class ResultPage1(tk.Frame):
    def __init__(self, parent, controller, bg_images):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.fond = ctk.CTkLabel(master=self, image=bg_images['result1'])
        self.fond.place(x=0, y=0)

        ctk.CTkButton(self, text="ACCÉDER AU LIEN", command=lambda: print("Accessing link"), bg_color=("#000000", "#000000"), fg_color=("#000000", "#000000")).place(x=380, y=300)
        ctk.CTkButton(self, text="DÉTAILS DU TRAITEMENT", command=lambda: controller.show_frame(TreatmentDetailsPage), bg_color=("#000000", "#000000"), fg_color=("#000000", "#000000")).place(x=375, y=350)


class ResultPage3(tk.Frame):
    def __init__(self, parent, controller, bg_images):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.fond = ctk.CTkLabel(master=self, image=bg_images['result3'])
        self.fond.place(x=0, y=0)

        ctk.CTkButton(self, text="RETOUR AU MENU", command=lambda: controller.show_frame(StartPage), bg_color=("#000000", "#000000"), fg_color=("#000000", "#000000")).place(x=385, y=300)
        ctk.CTkButton(self, text="DÉTAILS DU TRAITEMENT", command=lambda: controller.show_frame(TreatmentDetailsPage), bg_color=("#000000", "#000000"), fg_color=("#000000", "#000000")).place(x=375, y=350)


class TreatmentDetailsPage(tk.Frame):
    def __init__(self, parent, controller, bg_images):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.fond = ctk.CTkLabel(master=self, image=bg_images['detail'])
        self.fond.place(x=0, y=0)

        ctk.CTkButton(self, text="Retour", command=lambda: controller.show_frame(StartPage), bg_color=("#000000", "#000000"), fg_color=("#000000", "#000000")).place(x=380, y=300)


if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
