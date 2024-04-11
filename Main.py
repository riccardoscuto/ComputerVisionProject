import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Canvas, colorchooser
import cv2
import json
import numpy as np
from PIL import Image, ImageTk
import requests
import threading

class ColorizationApp:
    def __init__(self, root):
        self.root = root
        self.initialize_variables()
        self.setup_window()

    def setup_window(self):
        self.root.title("Colorization App")
        self.create_ui_components()
        self.layout_ui_components()
        self.update_palette_options_visibility()

    def initialize_variables(self):
        self.selected_file_path = None
        self.file_type = None
        self.colorized_image_pil = None
        self.colorized_frames = []
        self.image_canvas = None
        self.selected_dot_color = None
        self.dots = [] 
        self.palettes = {}
        self.palette_preview_frame = None

    def create_ui_components(self):
        self.color_method_var = tk.StringVar(value="auto")
        self.color_method_var.trace_add("write", self.update_palette_options_visibility)
        self.palette_var = tk.StringVar(value="Palette 1")
        self.palette_color_count_var = tk.IntVar(value=5)
        self.auto_color_radio = tk.Radiobutton(self.root, text="Automatic Colorization", variable=self.color_method_var, value="auto")
        self.dot_mode_button = tk.Radiobutton(self.root, text="Dot Colorization", variable=self.color_method_var, value="dot")
        self.palette_color_radio = tk.Radiobutton(self.root, text="Palette Based Colorization", variable=self.color_method_var, value="palette")
        self.images_frame = tk.Frame(self.root)
        self.images_frame.pack(pady=20)
        self.dot_mode_button.pack(anchor='w')
        self.image_canvas = tk.Canvas(self.images_frame, cursor="cross")
        self.image_canvas.pack(side="left", padx=10)
        self.palette_option_menu = tk.OptionMenu(self.root, self.palette_var, "Palette 1", "Palette 2", "Palette 3")
        self.palette_color_count_menu = tk.OptionMenu(self.root, self.palette_color_count_var,1, 2, 3, 4, 5)
        self.load_palettes_button = tk.Button(self.root, text="Load Palettes", command=self.fetch_palettes)
        self.open_buttons_frame = tk.Frame(self.root)
        self.choose_color_button = tk.Button(self.root, text="Choose Color", command=self.choose_color)
        self.clear_dots_button = tk.Button(self.root, text="Clear dots", command=self.clear_dots)
        self.image_canvas.bind("<Button-1>", self.start_dot)
        self.image_canvas.bind("<ButtonRelease-1>", self.stop_dot)
        self.dot_width_var = tk.IntVar(value=5)
        self.clear_dots_button.pack()
        self.choose_color_button.pack_forget()
        self.open_image_button = tk.Button(self.open_buttons_frame, text="Open Image", command=self.open_image_dialog)
        self.open_video_button = tk.Button(self.open_buttons_frame, text="Open Video", command=self.open_video_dialog)
        self.convert_button = tk.Button(self.root, text="Convert", command=self.convert_image_or_video)
        self.save_buttons_frame = tk.Frame(self.root)
        self.save_image_button = tk.Button(self.save_buttons_frame, text="Save Colorized Image", command=self.save_image)
        self.save_video_button = tk.Button(self.save_buttons_frame, text="Save Colorized Video", command=self.save_video)
        self.original_image_label = tk.Label(self.images_frame, text="Original Image")
        self.colorized_image_label = tk.Label(self.images_frame, text="Colorized Image")

    def layout_ui_components(self):
        self.auto_color_radio.pack(anchor='w', pady=(5, 0))
        self.palette_color_radio.pack(anchor='w', pady=(5, 0))
        self.dot_mode_button.pack(anchor='w', pady=(5, 0))
        self.images_frame.pack()
        self.open_buttons_frame.pack(pady=(0, 0))
        self.open_image_button.pack(side="left", padx=(5, 0))
        self.open_video_button.pack(side="left", padx=(5, 5))
        self.convert_button.pack(pady=(5, 5))
        self.save_buttons_frame.pack(pady=(0, 0))
        self.save_image_button.pack(side="left", padx=(5, 5))
        self.save_video_button.pack(side="left", padx=(5, 5))

    def update_frames(self, original_image_pil, colorized_image_pil):
        if original_image_pil:
            self.original_image_pil = original_image_pil
            self.original_image_pil_resized = self.resize_image(original_image_pil, width=300, height=300)
            self.original_image_tk = ImageTk.PhotoImage(self.original_image_pil_resized)
            if hasattr(self, 'original_image_on_canvas'):
                self.image_canvas.itemconfig(self.original_image_on_canvas, image=self.original_image_tk)
            else:
                self.original_image_on_canvas = self.image_canvas.create_image(0, 0, image=self.original_image_tk, anchor='nw')
            canvas_width = max(self.original_image_pil_resized.width, 600)
            canvas_height = max(self.original_image_pil_resized.height, 300)
            self.image_canvas.config(width=canvas_width, height=canvas_height)
            self.image_canvas.bind("<Button-1>", self.register_dot)
        if colorized_image_pil:
            colorized_image_pil_resized = self.resize_image(colorized_image_pil, width=300, height=300)
            colorized_image_tk = ImageTk.PhotoImage(colorized_image_pil_resized)
            colorized_image_x = self.original_image_pil_resized.width + 10
            colorized_image_y = 0
            if hasattr(self, 'colorized_image_on_canvas'):
                self.image_canvas.itemconfig(self.colorized_image_on_canvas, image=colorized_image_tk)
                self.image_canvas.moveto(self.colorized_image_on_canvas, colorized_image_x, colorized_image_y)
            else:
                self.colorized_image_on_canvas = self.image_canvas.create_image(colorized_image_x, colorized_image_y, image=colorized_image_tk, anchor='nw')
            self.colorized_image_label.image = colorized_image_tk
        self.root.update_idletasks()
        self.root.update()

    def open_image_dialog(self):
        self.selected_file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if self.selected_file_path:
            self.file_type = 'image'
            original_image_pil = Image.open(self.selected_file_path)
            self.update_frames(original_image_pil, None)

    def open_video_dialog(self):
        self.selected_file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if self.selected_file_path:
            self.file_type = 'video'

    def detect_edges(self, grayscale_image):
        return cv2.Canny(grayscale_image, threshold1=70, threshold2=180)

    def create_palette_window(self):
        new_window = Toplevel(self.root)
        new_window.title("Palette Colors")
        canvas = Canvas(new_window, width=300, height=50 * len(self.palettes))
        canvas.pack()
        y_position = 0
        for palette_name, colors in self.palettes.items():
            canvas.create_text(150, y_position + 15, text=palette_name, font=("Helvetica", 16))
            y_position += 30
            for i, color in enumerate(colors):
                x_position = i * (300 / len(colors))
                int_color = tuple(int(c) for c in color)
                canvas.create_rectangle(x_position, y_position, x_position + (300 / len(colors)), y_position + 20, fill=self.format_color_for_display(int_color), outline=self.format_color_for_display(int_color))
            y_position += 40
            canvas.create_rectangle(0, y_position - 40, 300, y_position, fill="", outline="", tags=(palette_name,))
            canvas.tag_bind(palette_name, "<Button-1>", lambda e, name=palette_name: self.set_selected_palette(name))
        canvas.config(scrollregion=canvas.bbox("all"))

    def update_palette_options_visibility(self, *args):
        if self.color_method_var.get() == "palette":
            self.show_palette_options()
            self.clear_dots_button.pack_forget()
        else:
            self.hide_palette_options()
            self.clear_dots_button.pack_forget()
        if self.color_method_var.get() == "dot":
            self.choose_color_button.pack()
            self.clear_dots_button.pack()
        else:
            self.choose_color_button.pack_forget()

    def show_palette_options(self):
        self.palette_option_menu.pack()
        self.palette_color_count_menu.pack()
        self.load_palettes_button.pack()
        if self.palette_preview_frame is not None:
            self.palette_preview_frame.pack()

    def clear_dots(self):
        self.dots.clear()
        self.image_canvas.delete("dot")

    def choose_color(self):
        self.selected_dot_color = colorchooser.askcolor(title="Choose dot color")[1]
        if self.selected_dot_color:
            self.root.config(cursor="cross")
        else:
            self.root.config(cursor="")

    def hide_palette_options(self):
        self.palette_option_menu.pack_forget()
        self.palette_color_count_menu.pack_forget()
        self.load_palettes_button.pack_forget()
        if self.palette_preview_frame is not None:
            self.palette_preview_frame.pack_forget()

    def interpolate_color(self, grayscale_value, palette):
        if grayscale_value <= palette[0][0]:
            return palette[0][1]
        if grayscale_value >= palette[-1][0]:
            return palette[-1][1]
        for i in range(1, len(palette)):
            if grayscale_value < palette[i][0]:
                lower_gray, lower_color = palette[i - 1]
                upper_gray, upper_color = palette[i]
                gray_diff = upper_gray - lower_gray
                weight = (grayscale_value - lower_gray) / gray_diff
                lower_color_np = np.array(lower_color)
                upper_color_np = np.array(upper_color)
                interpolated_color_np = lower_color_np + weight * (upper_color_np - lower_color_np)
                return interpolated_color_np.tolist()
        return palette[-1][1]

    def apply_palette_colorization(self, image_path):
        selected_palette_name = self.palette_var.get()
        selected_palette = self.palettes[selected_palette_name]
        bw_image_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        colorized_image_cv = np.zeros((bw_image_cv.shape[0], bw_image_cv.shape[1], 3), dtype=np.uint8)
        edges = self.detect_edges(bw_image_cv)
        gray_scale_values = np.linspace(0, 255, num=len(selected_palette))
        interpolated_palette = list(zip(gray_scale_values, selected_palette))
        for i in range(bw_image_cv.shape[0]):
            for j in range(bw_image_cv.shape[1]):
                grayscale_value = bw_image_cv[i, j]
                if not edges[i, j]:
                    colorized_image_cv[i, j] = self.interpolate_color(grayscale_value, interpolated_palette)
                else:
                    colorized_image_cv[i, j] = [grayscale_value, grayscale_value, grayscale_value]
        colorized_image_pil = Image.fromarray(colorized_image_cv)
        self.colorized_image_pil = colorized_image_pil
        self.update_frames(None, self.colorized_image_pil)

    def set_selected_palette(self, palette_name):
        self.selected_palette_name = palette_name
        self.palette_var.set(palette_name)

    def format_color_for_display(self, color):
        color = tuple(int(c) for c in color)
        return '#{:02x}{:02x}{:02x}'.format(*color)

    
    def register_dot(self, event):
        if self.selected_dot_color:
            x, y = event.x, event.y
            if x < 300 and y < 300 and x > 0 and y > 0:
                self.dots.append((x, y, self.selected_dot_color))
                self.show_dot_on_image(x, y, self.selected_dot_color)

    def show_dot_on_image(self, x, y, color):
        if isinstance(color, str):
            color = self.hex_to_rgb(color)
        radius = self.dot_width_var.get()
        self.image_canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=self.format_color_for_display(color), outline=self.format_color_for_display(color), tags="dot")

    def create_palette_preview(self, palette_name):
        if self.palette_preview_frame is not None:
            self.palette_preview_frame.destroy()
        self.palette_preview_frame = tk.Frame(self.root)
        self.palette_preview_frame.pack(pady=10)
        palette_colors = self.palettes.get(palette_name, [])
        for color in palette_colors:
            color_hex = self.format_color_for_display(color)
            color_label = tk.Label(self.palette_preview_frame, bg=color_hex, width=20, height=2)
            color_label.pack(side="left", padx=1)

    def update_palette_preview(self):
        selected_palette_name = self.palette_var.get()
        self.create_palette_preview(selected_palette_name)

    def fetch_palettes(self):
        def do_request():
            self.palettes = {}
            color_count = self.palette_color_count_var.get()
            try:
                response = requests.post('http://colormind.io/api/', data=json.dumps({"model": "default"}))
                response.raise_for_status()
                palette_data = response.json()
                palette_name = f"Generated Palette with {color_count} colors"
                self.palettes[palette_name] = palette_data["result"][:color_count]
            except requests.RequestException as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
            self.update_palette_menu()
        thread = threading.Thread(target=do_request)
        thread.start()

    def start_dot(self, event):
        self.last_x, self.last_y = event.x, event.y
        self.image_canvas.bind("<B1-Motion>", self.extend_dot)

    def stop_dot(self, event):
        self.image_canvas.unbind("<B1-Motion>")
        self.last_x = None
        self.last_y = None

    def extend_dot(self, event):
        x, y = event.x, event.y
        if self.selected_dot_color and self.last_x is not None and self.last_y is not None:
            self.image_canvas.create_line(self.last_x, self.last_y, x, y, fill=self.selected_dot_color, width=self.dot_width_var.get(), tags="dot")
            self.last_x, self.last_y = x, y
            self.dots.append(((self.last_x, self.last_y), (x, y), self.selected_dot_color))

    def update_palette_menu(self):
        menu = self.palette_option_menu["menu"]
        menu.delete(0, "end")
        for palette_name in self.palettes.keys():
            menu.add_command(label=palette_name, command=lambda value=palette_name: [self.palette_var.set(value), self.update_palette_preview()])
        self.palette_var.set(next(iter(self.palettes.keys())))
        self.update_palette_preview()

    def convert_image_or_video(self):
        if self.selected_file_path:
            if self.file_type == 'image':
                if self.color_method_var.get() == "auto":
                    self.colorize_image(self.selected_file_path)
                elif self.color_method_var.get() == "palette":
                    self.apply_palette_colorization(self.selected_file_path)
                elif self.color_method_var.get() == "dot":
                    self.process_dot_based_colorization()
            elif self.file_type == 'video':
                self.colorize_video(self.selected_file_path)

    def colorize_image(self, image_path):
        net = self.load_model()
        bw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(bw_image, 100, 200)
        bw_image_colored = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)
        colorized = self.process_image(net, bw_image_colored)
        colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
        edges_mask = edges.astype(bool)
        colorized_array = np.array(colorized_rgb)
        original_array = np.array(bw_image_colored)
        colorized_array[edges_mask] = original_array[edges_mask]
        self.colorized_image_pil = Image.fromarray(colorized_array)
        original_image_pil = Image.open(image_path)
        self.update_frames(original_image_pil, self.colorized_image_pil)

    def colorize_video(self, video_path):
        net = self.load_model()
        cap = cv2.VideoCapture(video_path)
        self.colorized_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            colorized = self.process_image(net, frame)
            self.colorized_frames.append(colorized)
        cap.release()
        self.show_video_preview()

    def load_model(self):
        prototxt_path = 'models/colorization_deploy_v2.prototxt'
        model_path = 'models/colorization_release_v2.caffemodel'
        kernel_path = 'models/pts_in_hull.npy'
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        points = np.load(kernel_path)
        points = points.transpose().reshape(2, 313, 1, 1)
        net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
        net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        return net

    def process_image(self, net, bw_image):
        if len(bw_image.shape) == 2:
            bw_image = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)
        normalized = bw_image.astype("float32") / 255.0
        lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0] - 50
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        return (255.0 * colorized).astype("uint8")

    def segment_image(self, image, threshold=128):
        _, segmented = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        num_labels, labels_im = cv2.connectedComponents(segmented)
        return labels_im

    def get_dot_palette(self, dot_color):
        grayscale_values = np.linspace(0, 255, num=256)
        dot_color_np = np.array(self.hex_to_rgb(dot_color))
        interpolated_palette = [np.clip(dot_color_np * (gray / 255.0), 0, 255).astype(int) for gray in grayscale_values]
        return dict(zip(grayscale_values, interpolated_palette))

    def apply_interpolated_colors(self, image, segment_id, dot_palette, segments):
        indices = np.where(segments == segment_id)
        for i in range(len(indices[0])):
            y, x = indices[0][i], indices[1][i]
            gray_value = image[y, x, 0]
            interpolated_color = dot_palette.get(gray_value, dot_palette[255])
            image[y, x] = interpolated_color

    def process_dot_based_colorization(self):
        if not self.dots or self.selected_file_path is None:
            return
        original_image = cv2.imread(self.selected_file_path)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        segments = self.segment_image(gray_image)
        net = self.load_model()
        colorized_auto = self.process_image(net, original_image)
        colorized_auto = np.array(Image.fromarray(cv2.cvtColor(colorized_auto, cv2.COLOR_BGR2RGB)))
        for dot in self.dots:
            x, y, dot_color = dot
            dot_palette = self.get_dot_palette(dot_color)
            segment_id = segments[y, x]
            self.apply_interpolated_colors(colorized_auto, segment_id, dot_palette, segments)
        self.colorized_image_pil = Image.fromarray(cv2.cvtColor(colorized_auto, cv2.COLOR_BGR2RGB))
        self.colorized_image_pil = Image.fromarray(cv2.cvtColor(np.array(self.colorized_image_pil), cv2.COLOR_BGR2RGB))
        self.update_frames(None, self.colorized_image_pil)

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def resize_image(self, image, width=300, height=300):
        return image.resize((width, height), Image.Resampling.LANCZOS)

    def save_image(self):
        if self.colorized_image_pil:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
            if file_path:
                self.colorized_image_pil.save(file_path)
                messagebox.showinfo("Salvataggio completato", "L'immagine è stata salvata con successo.")

    def show_video_preview(self):
        if self.colorized_frames:
            colorized_image_pil = Image.fromarray(cv2.cvtColor(self.colorized_frames[-1], cv2.COLOR_BGR2RGB))
            colorized_image_pil_resized = self.resize_image(colorized_image_pil, width=300, height=300)
            self.original_image_pil_resized = colorized_image_pil_resized
            self.update_frames(None, colorized_image_pil_resized)
    def save_video(self):
        if self.colorized_frames:
            file_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
            if file_path:
                height, width, layers = self.colorized_frames[0].shape
                size = (width, height)
                out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, size)
                for i in range(len(self.colorized_frames)):
                    out.write(self.colorized_frames[i])
                out.release()
                messagebox.showinfo("Salvataggio completato", "Il video è stato salvato con successo.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorizationApp(root)
    root.mainloop()