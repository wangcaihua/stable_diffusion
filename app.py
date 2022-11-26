import base64
from PIL import Image
from io import BytesIO
import numpy as np

import dash
from dash import html, dcc, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_canvas import DashCanvas
from dash.dependencies import Input, Output, State
from dash_canvas.utils import array_to_data_url
from stable_diffusion_tf.stable_diffusion import get_or_create_generate
from stable_diffusion_tf.utils import LocalManager, parse_jsonstring

long_callback_manager = LocalManager()


def base65_to_image(b64: str, height: int = None, width: int = None) -> np.ndarray:
  b64_content = b64.split(',', maxsplit=1)[-1]
  im: Image = Image.open(BytesIO(base64.b64decode(b64_content)))
  if height is not None and width is not None:
    width, height = int(width), int(height)
    if im.width != width or im.height != height:
      k = min(im.width / width, im.height / height)
      im = im.resize(size=(int(im.width / k + 0.5), int(im.height / k + 0.5)))
      x, y = int(im.width / 2), int(im.height / 2)
      half_w, half_h = int(width / 2), int(height / 2)
      top, left = max(0, y - half_h), max(0, x - half_w)
      return np.array(im.crop(box=(left, top, left + width, top + height)))
    else:
      return np.array(im)
  else:
    return np.array(im)


app = dash.Dash(name=__name__,
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                long_callback_manager=long_callback_manager)

upload_style = {
  'width': '100%',
  # 'height': '60px',
  # 'lineHeight': '60px',
  'borderWidth': '1px',
  'borderStyle': 'dashed',
  'borderRadius': '5px',
  'textAlign': 'center',
  'margin': '5px'
}
dcc.Checklist()
app.layout = dbc.Container(children=[
  dbc.Row(dbc.Col(html.Label("Stable Diffusion", style={"font-size": 36}),
                  width={"size": 4, 'offset': 4}, style={'text-align': 'center'})),
  dbc.Row(dbc.Col(html.Hr(), width=12)),
  dbc.Row([dbc.Col(html.Label('height')),
           dbc.Col(dcc.Input(value='512', id='height', type="number", style={'width': '100%'}), width=1),
           dbc.Col(html.Label('width')),
           dbc.Col(dcc.Input(value='512', id='width', type="number", style={'width': '100%'}), width=1),
           dbc.Col(html.Label('num_steps')),
           dbc.Col(dcc.Input(value='25', id='num_steps', type="number", style={'width': '100%'}), width=1),
           dbc.Col(html.Label('temperature')),
           dbc.Col(dcc.Input(value='1', id='temperature', type="number", style={'width': '100%'}), width=1),
           dbc.Col(html.Label('guidance_scale')),
           dbc.Col(dcc.Input(value='7.5', id='guidance_scale', type="number", style={'width': '100%'}), width=1),
           dbc.Col(html.Label('image_strength')),
           dbc.Col(dcc.Input(value='0.5', id='image_strength', type="number", style={'width': '100%'}), width=1),
           ], style={'text-align': 'right'}),
  dbc.Row([dbc.Col(html.Label('prompt'), width=2, style={'margin': '5px 0px'}),
           dbc.Col(
             dcc.Input(value='An astronaut riding a horse', placeholder='An astronaut riding a horse',
                       id='prompt', type="text", style={'width': '100%'}),
             width=10,
             style={'margin': '5px 0px'})],
          style={'text-align': 'left'}),
  dbc.Row([dbc.Col(html.Label('negative_prompt'), width=2, style={'margin': '5px 0px'}),
           dbc.Col(dcc.Input(placeholder='negative prompt', id='negative_prompt', type="text", style={'width': '100%'}),
                   width=10,
                   style={'margin': '5px 0px'})],
          style={'text-align': 'left'}
          ),
  dbc.Row([
    dbc.Col(children=[
      dbc.Row(dbc.Col(dcc.Upload(
        id='upload_image',
        children='please upload a image',
        style=upload_style,
        multiple=False
      ), width=12)),
      dbc.Row(dbc.Col(html.Div(children=None, id='image'), width=12))
    ], width=6),
    dbc.Col(children=[
      dbc.Row(
        [dbc.Col(dcc.Dropdown(['Select', 'Reverse'], value='Select', id='dropdown', style={'width': '100%'}), width=2),
         dbc.Col(dcc.Slider(
           id='size_slider', min=0, max=40, step=4, value=10,
         ), width=5),
         dbc.Col(dcc.Slider(
           id='alpha_slider', min=0, max=1, step=0.1, value=0,
         ), width=5)], id='dash_slider', style={"display": 'none'}),
      dbc.Row(children=DashCanvas(
        id='mask',
        hide_buttons=['zoom'],
        width=512,
        height=512,
        lineWidth=10,
        goButtonTitle='Save'
      ), id='dash_canvas', style={"display": 'none'})], width=6)
  ]),
  dbc.Row([
    dbc.Col(html.Button("disable mask", id='msk_stat', style={'width': "100%"}), width={'size': 2, 'offset': 7}),
    dbc.Col(html.Button("disable image", id='img_stat', style={'width': "100%"}), width=2),
    dbc.Col(html.Button("go!", id='submit', style={'width': "100%"}), width=1)
  ], style={'margin': '5px 0px'}),
  dbc.Row(dbc.Col(html.Hr(), width=12)),
  dbc.Row(dbc.Col(dbc.Progress(value=0, striped=True, id='progress', animated=True), width=12),
          id='progress_row', style={"display": "none"}),
  dbc.Row([
    dbc.Col(html.Div(id='progress-img'), width=6),
    dbc.Col(html.Div(id='output-img'), width=6)
  ], style={'margin': '10px'}),
])


@app.callback(Output('image', 'children'),
              Input('upload_image', 'contents'),
              State('height', 'value'),
              State('width', 'value'))
def update_image(upload_content, height, width):
  if upload_content is not None:
    img = base65_to_image(upload_content, int(height), int(width))
    img = array_to_data_url(img)
    return html.Img(src=img)
  else:
    raise PreventUpdate()


@app.callback(Output('mask', 'lineWidth'),
              Input('size_slider', 'value'))
def update_canvas_linewidth(value):
  return value


@app.callback(Output('mask', 'lineColor'),
              Input('alpha_slider', 'value'))
def update_canvas_linecolor(value):
  if value is not None:
    return f'rgba(255,0,0,{1 - value})'
  else:
    raise PreventUpdate()


@app.callback(Output('mask', 'image_content'),
              Output('mask', 'goButtonTitle'),
              Output('dash_canvas', 'style'),
              Output('dash_slider', 'style'),
              Output('image', 'style'),
              Output('img_stat', 'children'),
              Output('msk_stat', 'children'),
              Input('mask', 'trigger'),
              Input('image', 'children'),
              Input('img_stat', 'n_clicks'),
              State('img_stat', 'children'),
              Input('msk_stat', 'n_clicks'),
              State('msk_stat', 'children'),
              State('mask', 'json_data'),
              State('mask', 'image_content'),
              State('mask', 'goButtonTitle'),
              State('height', 'value'),
              State('width', 'value'),
              Input('dropdown', 'value'))
def update_mask(trigger, image_content, img_stat_clicks, img_stat_label, mak_stat_clicks, msk_stat_label,
                json_data, mask_content, button_title, height, width, reverse):
  ctx = dash.callback_context
  component_id = ctx.triggered[0]['prop_id'].split('.')[0]
  style = {'display': 'none'}
  if component_id == 'image':
    return image_content['props']['src'], no_update, None, None, None, 'disable image', 'disable mask'
  elif component_id == 'mask':
    if image_content is None or image_content.get('props') is None or image_content['props'].get('src') is None:
      raise PreventUpdate()
    if button_title == 'Reset':
      return no_update, "Save", no_update, no_update, no_update, no_update, no_update
    elif button_title == 'Save':
      need_reverse = reverse == 'Reverse'
      mask = parse_jsonstring(json_data, (int(width), int(height)))
      mask_content = array_to_data_url(mask if need_reverse else 255 - mask)
      return mask_content, 'Reset', no_update, no_update, no_update, no_update, no_update
    else:
      raise PreventUpdate()
  elif component_id == 'dropdown':
    if button_title == 'Save':
      raise PreventUpdate()
    mask_img = base65_to_image(mask_content, width=width, height=height)
    mask_content = array_to_data_url(255 - mask_img)
    return mask_content, no_update, no_update, no_update, no_update, no_update, no_update
  elif component_id == 'img_stat':
    if image_content is None or image_content.get('props') is None or image_content['props'].get('src') is None:
      raise PreventUpdate()
    if img_stat_label == 'disable image':
      return no_update, no_update, style, style, style, 'enable image', no_update
    elif img_stat_label == 'enable image':
      if msk_stat_label == 'disable mask':
        return no_update, no_update, None, None, None, 'disable image', no_update
      elif msk_stat_label == 'enable mask':
        return no_update, no_update, style, style, None, 'disable image', no_update
      else:
        raise PreventUpdate()
    else:
      raise PreventUpdate()
  elif component_id == 'msk_stat':
    if image_content is None or image_content.get('props') is None or image_content['props'].get('src') is None:
      raise PreventUpdate()
    if img_stat_label == 'disable image':
      if msk_stat_label == 'disable mask':
        return no_update, no_update, style, style, None, no_update, 'enable mask'
      elif msk_stat_label == 'enable mask':
        return no_update, no_update, None, None, None, no_update, 'disable mask'
      else:
        raise PreventUpdate()
    else:
      raise PreventUpdate()
  else:
    raise PreventUpdate()


@app.callback(Output('progress_row', 'style'),
              Input('submit', 'n_clicks'),
              prevent_initial_call=True)
def show_progress(n_clicks):
  if n_clicks is not None:
    return None
  else:
    raise PreventUpdate()


@app.callback(Output('output-img', 'children'),
              Input('submit', 'n_clicks'),
              State('prompt', 'value'),
              State('negative_prompt', 'value'),
              State('height', 'value'),
              State('width', 'value'),
              State('num_steps', 'value'),
              State('temperature', 'value'),
              State('guidance_scale', 'value'),
              State('image_strength', 'value'),
              State('image', 'children'),
              State('mask', 'image_content'),
              State('mask', 'goButtonTitle'),
              State('img_stat', 'children'),
              State('msk_stat', 'children'),
              running=[(Output("submit", "disabled"), True, False),
                       (Output('output-img', 'style'), {"visibility": "hidden"}, {"visibility": "visible"}),
                       ],
              progress=[Output('progress', 'value'), Output('progress-img', 'children')],
              prevent_initial_call=True,
              background=True)
def do_compute(set_progress, n_clicks, prompt, negative_prompt, height, width, num_steps, temperature, guidance_scale,
               image_strength, image, mask, button_title, img_stat_label, msk_stat_label):
  if n_clicks is None:
    raise PreventUpdate()
  if img_stat_label == 'enable image':
    image, mask = None, None
  if image is not None and image.get('props') and image['props'].get('src'):
    image = image['props']['src']
  else:
    image = None
  
  if image is not None and msk_stat_label == 'disable mask':
    mask = mask
  else:
    mask = None
  set_progress((0, None))
  
  def progress_callback(progress, latent, input_image_array, input_mask_array, model):
    decoded = model.decoder.predict_on_batch(latent)
    decoded = ((decoded + 1) / 2) * 255
    if input_mask_array is not None:
      decoded = input_image_array * (1 - input_mask_array) + np.array(decoded) * input_mask_array
    img = np.clip(decoded, 0, 255).astype("uint8")
    # out_img = Image.fromarray(np.squeeze(img))
    # out_img.save(f'/Users/fitz/code/stable_diffusion/assets/image_{progress}.png')
    set_progress((progress, html.Img(src=array_to_data_url(img[0]))))
  
  generator = get_or_create_generate(
    img_height=int(height),
    img_width=int(width)
  )
  
  img = generator.generate(
    prompt=prompt or "iron men fly in the sky",
    negative_prompt=negative_prompt,
    num_steps=int(num_steps),
    unconditional_guidance_scale=float(guidance_scale),
    temperature=float(temperature),
    batch_size=1,
    input_image_strength=float(image_strength),
    input_image=image,
    input_mask=None if button_title == 'Save' else mask,
    progress_call_back=progress_callback
  )
  return html.Img(src=array_to_data_url(img[0]))


if __name__ == '__main__':
  app.run_server(debug=False, port=8051)
