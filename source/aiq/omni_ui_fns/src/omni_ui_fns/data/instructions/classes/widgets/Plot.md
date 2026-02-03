# omni.ui.Plot

The Plot class displays a line or histogram image. The data of the image is specified as a data array or a provider function.

Here is a list of styles you can customize on Plot:
> border_color (color): the border color if the button or image background has a border
> border_radius (float): the border radius if the user wants to round the button or image
> border_width (float): the border width if the button or image or image background has a border
> margin (float): the distance between the widget content and the parent widget defined boundary
> margin_width (float): the width distance between the widget content and the parent widget defined boundary
> margin_height (float): the height distance between the widget content and the parent widget defined boundary
> color (color): the color of the plot, line color in the line typed plot or rectangle bar color in the histogram typed plot
> selected_color (color): the selected color of the plot, dot in the line typed plot and rectangle bar in the histogram typed plot
> background_color (color): the background color of the plot
> secondary_color (color): the color of the text and the border of the text box which shows the plot selection value
> background_selected_color (color): the background color of the text box which shows the plot selection value

Here are couple of examples of Plots:
```
import math
from omni.ui import color as cl
data = []
for i in range(360):
    data.append(math.cos(math.radians(i)))

def on_data_provider(index):
    return math.sin(math.radians(index))

with ui.Frame(height=20):
    with ui.HStack():
        plot_1 = ui.Plot(ui.Type.LINE, -1.0, 1.0, *data, width=360, height=100,
                style={"Plot":{
                    "color": cl.red,
                    "background_color": cl(0.08),
                    "secondary_color": cl("#aa1111"), #the color of the text and the border of the text box which shows the plot selection value
                    "selected_color": cl.green,
                    "background_selected_color": cl.white,
                    "border_width":5,
                    "border_color": cl.blue,
                    "border_radius": 20
                    }})
        ui.Spacer(width = 20)
        plot_2 = ui.Plot(ui.Type.HISTOGRAM, -1.0, 1.0, on_data_provider, 360, width=360, height=100,
                style={"Plot":{
                    "color": cl.blue,
                    "background_color": cl("#551111"),
                    "secondary_color": cl("#11AA11"), the color of the text and the border of the text box which shows the plot selection value
                    "selected_color": cl(0.67),
                    "margin_height": 10,
                    }})
        plot_2.value_stride = 6
```

