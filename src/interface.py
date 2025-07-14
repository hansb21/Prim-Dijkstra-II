import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import matplotlib.patches as patches
import networkx as nx
import mst
import pd
import parse
import os
import glob
import analysis

class SimpleInterface:
   def __init__(self, def_files_pattern="ispd18_sample/*.def"):
       
       self.def_files = glob.glob(def_files_pattern)
       if not self.def_files:
           print(f"Nenhum arquivo DEF encontrado em: {def_files_pattern}")
           self.def_files = ["ispd18_sample/6.def"]
       
       self.def_file_names = [os.path.basename(f) for f in self.def_files]
       self.current_file_index = 0
       
       self.design = None
       self.current_file = None
       
       self.fig, self.ax = plt.subplots(figsize=(14, 8))
       plt.subplots_adjust(bottom=0.2, top=0.85)  
       
       self.current_pd_tree = None
       self.current_pd2_tree = None
       self.current_source = 'U1'
       self.current_alpha = 0.5

       self.setup_top_controls()
       
       button_width = 0.08
       button_spacing = 0.09
       start_x = 0.02
       
       ax_btn1 = plt.axes([start_x + 0*button_spacing, 0.05, button_width, 0.05])
       ax_btn2 = plt.axes([start_x + 1*button_spacing, 0.05, button_width, 0.05])
       ax_btn3 = plt.axes([start_x + 2*button_spacing, 0.05, button_width, 0.05])
       ax_btn4 = plt.axes([start_x + 3*button_spacing, 0.05, button_width, 0.05])
       ax_btn7 = plt.axes([start_x + 6*button_spacing, 0.05, button_width, 0.05])
       ax_btn10 = plt.axes([start_x + 9*button_spacing, 0.05, button_width, 0.05])

       self.btn1 = Button(ax_btn1, 'Prim')
       self.btn2 = Button(ax_btn2, 'Dijkstra')
       self.btn3 = Button(ax_btn3, 'PD')
       self.btn4 = Button(ax_btn4, 'PD-II')
       self.btn7 = Button(ax_btn7, 'Análise')
       self.btn10 = Button(ax_btn10, 'Clear')

       self.btn1.on_clicked(self.function1)
       self.btn2.on_clicked(self.function2)
       self.btn3.on_clicked(self.function3)
       self.btn4.on_clicked(self.function4)
       self.btn7.on_clicked(self.run_analysis)
       self.btn10.on_clicked(self.clear_plot)

       ax_textbox = plt.axes([0.70, 0.12, 0.08, 0.04])
       self.textbox = TextBox(ax_textbox, 'Alpha: ', initial='0.5')

       ax_source = plt.axes([0.79, 0.12, 0.10, 0.04])
       self.source_textbox = TextBox(ax_source, 'Source: ', initial='U1')
       self.source_textbox.on_submit(self.on_source_changed)

       ax_source_btn = plt.axes([0.90, 0.12, 0.08, 0.04])
       self.btn_source_list = Button(ax_source_btn, 'Lista Sources')
       self.btn_source_list.on_clicked(self.show_source_options)

       self.source_info = self.fig.text(0.70, 0.08, 'Clique em um componente, digite no campo Source, ou use Lista Sources', 
                                       fontsize=8, style='italic')

       self.fig.canvas.mpl_connect('button_press_event', self.on_plot_click)

       self.ax.set_title('Routing - Selecione um arquivo')
       self.ax.grid(True, alpha=0.3)
       
       if self.def_files:
           self.load_design_file(self.def_files[0])

   def setup_top_controls(self):
       ax_prev = plt.axes([0.02, 0.92, 0.06, 0.04])
       ax_next = plt.axes([0.09, 0.92, 0.06, 0.04])
       
       self.btn_prev = Button(ax_prev, '← Ant')
       self.btn_next = Button(ax_next, 'Próx →')
       
       self.btn_prev.on_clicked(self.prev_file)
       self.btn_next.on_clicked(self.next_file)
       
       self.file_label = self.fig.text(0.17, 0.94, '', fontsize=10, fontweight='bold')
       self.file_info = self.fig.text(0.17, 0.91, '', fontsize=8)
       
       self.update_file_display()

   def prev_file(self, event):
       if len(self.def_files) > 1:
           self.current_file_index = (self.current_file_index - 1) % len(self.def_files)
           self.load_design_file(self.def_files[self.current_file_index])

   def next_file(self, event):
       if len(self.def_files) > 1:
           self.current_file_index = (self.current_file_index + 1) % len(self.def_files)
           self.load_design_file(self.def_files[self.current_file_index])

   def update_file_display(self):
       if self.def_files:
           current_name = self.def_file_names[self.current_file_index]
           self.file_label.set_text(f"Arquivo: {current_name} ({self.current_file_index + 1}/{len(self.def_files)})")
           
           if self.design:
               num_components = len(self.design['components'])
               num_nets = len(self.design['nets'])
               self.file_info.set_text(f"Componentes: {num_components} | Nets: {num_nets}")
   
   def update_source_options(self):
       if self.design and self.design['components']:
           if self.current_source not in self.design['components']:
               self.current_source = list(self.design['components'].keys())[0]
               self.source_textbox.set_val(self.current_source)

   def on_source_changed(self, text):
       if self.design and text in self.design['components']:
           self.current_source = text
           self.highlight_source_component()
           print(f"Source alterado para: {text}")
       else:
           print(f"Componente '{text}' não encontrado no design")
           if self.design:
               available = list(self.design['components'].keys())
               print(f"   Componentes disponíveis: {available}")
   
   def show_source_options(self, event):
      if not self.design:
          return
      
      print("\n" + "="*40)
      print("COMPONENTES DISPONÍVEIS COMO SOURCE:")
      print("="*40)
      
      components = list(self.design['components'].keys())
      for i, comp in enumerate(components, 1):
          marker = "👑" if comp == self.current_source else "  "
          print(f"{marker} {i:2d}. {comp}")
      
      print(f"\n📍 Source atual: {self.current_source}")
 
   def on_plot_click(self, event):
       if event.inaxes != self.ax or not self.design:
           return
       
       click_x, click_y = event.xdata, event.ydata
       if click_x is None or click_y is None:
           return
       
       min_dist = float('inf')
       closest_comp = None
       
       for comp_id, comp_data in self.design['components'].items():
           comp_x, comp_y = comp_data['x'], comp_data['y']
           dist = ((click_x - comp_x)**2 + (click_y - comp_y)**2)**0.5
           
           if dist < min_dist and dist < 30:
               min_dist = dist
               closest_comp = comp_id
       
       if closest_comp:
           self.current_source = closest_comp
           self.source_textbox.set_val(closest_comp)
           self.highlight_source_component()
           print(f"Source selecionado por clique: {closest_comp}")

   def highlight_source_component(self):
       if not self.design or self.current_source not in self.design['components']:
           return
       
       self.plot_components_only()
       
       source_data = self.design['components'][self.current_source]
       x, y = source_data['x'], source_data['y']
       
       circle = plt.Circle((x, y), 15, fill=False, edgecolor='gold', linewidth=3, zorder=10)
       self.ax.add_patch(circle)
       
       self.ax.annotate('SOURCE', (x, y), xytext=(0, -25), textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold', color='gold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
       
       plt.draw()

   def load_design_file(self, def_file_path):
       try:
           print(f"📂 Carregando: {def_file_path}")
           self.design = parse.parse_design(def_file_path)
           self.current_file = def_file_path
           
           for i, path in enumerate(self.def_files):
               if path == def_file_path:
                   self.current_file_index = i
                   break
           
           self.current_pd_tree = None
           self.current_pd2_tree = None
           
           self.update_file_display()
           self.update_source_options()
           self.plot_components_only()
           
           filename = os.path.basename(def_file_path)
           self.ax.set_title(f'Design: {filename}')
           
           print(f"Arquivo carregado com sucesso!")
           print(f"Componentes: {len(self.design['components'])}")
           print(f"Nets: {len(self.design['nets'])}")
           
       except Exception as e:
           print(f"Erro ao carregar {def_file_path}: {e}")
           self.ax.set_title('Erro ao carregar arquivo')

   def plot_components_only(self):
       if not self.design:
           return
           
       self.ax.clear()
       self.ax.grid(True, alpha=0.3)
       
       x_coords = []
       y_coords = []
       labels = []
       
       for comp_id, comp_data in self.design['components'].items():
           x, y = comp_data['x'], comp_data['y']
           x_coords.append(x)
           y_coords.append(y)
           labels.append(comp_id)
           
           rect = patches.Rectangle((x-5, y-5), 10, 10, 
                                  linewidth=1, edgecolor='blue', 
                                  facecolor='lightblue', alpha=0.7)
           self.ax.add_patch(rect)
       
       self.ax.scatter(x_coords, y_coords, c='red', s=50, zorder=5)
       
       for i, label in enumerate(labels):
           self.ax.annotate(label, (x_coords[i], y_coords[i]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, fontweight='bold')
       
       if x_coords and y_coords:
           margin = 20
           self.ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
           self.ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
       
       self.ax.set_aspect('equal', adjustable='box')
       filename = os.path.basename(self.current_file) if self.current_file else "Unknown"
       self.ax.set_title(f'Componentes - {filename}')
       
       if self.current_source in self.design['components']:
           source_data = self.design['components'][self.current_source]
           x, y = source_data['x'], source_data['y']
           
           circle = plt.Circle((x, y), 15, fill=False, edgecolor='gold', linewidth=3, zorder=10)
           self.ax.add_patch(circle)
           
           self.ax.annotate('SOURCE', (x, y), xytext=(0, -25), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold', color='gold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
       
       plt.draw()

   def check_design_loaded(self):
       if not self.design:
           print("Nenhum design carregado. Selecione um arquivo DEF primeiro.")
           return False
       return True

   def function1(self, event):
       if not self.check_design_loaded():
           return
           
       print("Button 1 clicked - Prim MST")
       G = mst.create_complete_graph(self.design)
       p_mst = pd.prim(G, source=self.current_source)
       mst.draw_manhattan_mst_on_axis(p_mst, ax=self.ax)
       plt.draw()

   def function2(self, event):
       if not self.check_design_loaded():
           return
           
       print("Button 2 clicked - Dijkstra SPT")
       G = mst.create_complete_graph(self.design)
       d_mst, dist, prev = pd.dijkstra(G, source=self.current_source)
       mst.draw_manhattan_mst_on_axis(d_mst, ax=self.ax)
       plt.draw()

   def function3(self, event):
       if not self.check_design_loaded():
           return
           
       print("Button 3 clicked - Prim-Dijkstra")
       try:
           param_value = float(self.textbox.text)
           if 0 <= param_value <= 1:
               self.current_alpha = param_value
               G = mst.create_complete_graph(self.design)
               d_mst, dist, prev = pd.pd(G, source=self.current_source, alpha=param_value)
               self.current_pd_tree = d_mst
               mst.draw_manhattan_mst_on_axis(d_mst, ax=self.ax)
               plt.draw()
               print("✅ Árvore PD salva para análise")
       except ValueError:
           print("invalid alpha value")

   def function4(self, event):
       if not self.check_design_loaded():
           return
           
       self.ax.clear()
       self.ax.set_title('Prim-Dijkstra II')
       self.ax.grid(True, alpha=0.3)
       try:
           param_value = float(self.textbox.text)
           if 0 <= param_value <= 1:
               self.current_alpha = param_value

               if self.current_pd_tree is None:
                   G = mst.create_complete_graph(self.design)
                   d_mst, dist, prev = pd.pd(G, source=self.current_source, alpha=param_value)
                   self.current_pd_tree = d_mst

               pd2_mst = pd.pd_ii(self.current_pd_tree, alpha=param_value, D=1, source=self.current_source)
               self.current_pd2_tree = pd2_mst

               mst.draw_manhattan_mst_on_axis(pd2_mst, ax=self.ax)
               self.fig.canvas.draw()
               print("✅ Árvore PD-II salva para análise")
       except ValueError:
           print("invalid alpha value")

   def run_analysis(self, event):
       if not self.check_design_loaded():
           return
           
       print("Executando análise completa...")

       if self.current_pd_tree is None:
           print("Execute primeiro o Prim-Dijkstra")
           return

       if self.current_pd2_tree is None:
           print("Execute primeiro o Prim-Dijkstra-II")
           return

       try:
           results = analysis.unified_pd_analysis(self.current_pd_tree, self.current_pd2_tree, 
                           self.current_source, self.current_alpha)
       except Exception as e:
           print(f"Erro na análise: {e}")

   def clear_plot(self, event):
       if self.design:
           self.plot_components_only()
       else:
           self.ax.clear()
           self.ax.set_title('Clear')
           self.ax.grid(True, alpha=0.3)
           plt.draw()

   def show(self):
       plt.show()

if __name__ == "__main__":
   interface = SimpleInterface()
   interface.show()
