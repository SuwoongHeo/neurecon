import plotly.graph_objects as go
import plotly.io as pio

def plotly_viscorres3D(vertices, faces, vnear=None, query=None, pp_color=None, faces_tanframe=None, draw_edge=False):
    pio.renderers.default = "browser"

    plotdata = []

    x, y, z, = vertices.cpu().chunk(3, -1)
    plmesh = go.Mesh3d(x=x.numpy()[:, 0], y=y.numpy()[:, 0], z=z.numpy()[:, 0], i=faces.numpy()[:, 0],
                       j=faces.numpy()[:, 1], k=faces.numpy()[:, 2], color='grey', opacity=.6,
                       lighting=dict(ambient=0.2, diffuse=0.8), lightposition=dict(x=0, y=0, z=-1))
    plotdata.append(plmesh)
    if draw_edge:
        xe, ye, ze = [], [], []
        tri = vertices[faces]
        for T in tri:
            xe.extend([T[k % 3][0] for k in range(4)] + [None])
            ye.extend([T[k % 3][1] for k in range(4)] + [None])
            ze.extend([T[k % 3][2] for k in range(4)] + [None])
        edges = go.Scatter3d(x=xe, y=ye, z=ze, mode='lines', name='', line=dict(color = 'rgb(25,25,25)', width=1))
        plotdata.append(edges)

    if vnear is not None:
        pp = vnear.view(-1, 3).cpu().numpy()
        pp_marker = go.Scatter3d(x=pp[:, 0], y=pp[:, 1], z=pp[:, 2], marker=go.scatter3d.Marker(size=3, color=pp_color),
                                 mode='markers')
        plotdata.append(pp_marker)

    if query is not None:
        qq = query.view(-1, 3).cpu().numpy()
        qq_marker = go.Scatter3d(x=qq[:, 0], y=qq[:, 1], z=qq[:, 2], marker=go.scatter3d.Marker(size=3, color=pp_color),
                             mode='markers')
        plotdata.append(qq_marker)

    x_lines, y_lines, z_lines = list(), list(), list()
    if (vnear is not None) and (query is not None):
        for qqq, ppp in zip(qq, pp):
            x_lines.extend([qqq[0], ppp[0], None])
            y_lines.extend([qqq[1], ppp[1], None])
            z_lines.extend([qqq[2], ppp[2], None])
        lines = go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines')
        plotdata.append(lines)

    if faces_tanframe is not None:
        frames = []
        frames_np = faces_tanframe.view(-1, 3, 3).cpu().numpy()
        for ax in range(frames_np.shape[-2]):
            # x', y', z'
            x_lines, y_lines, z_lines = list(), list(), list()
            ff = pp + frames_np[:, ax, :]
            for fff, ppp in zip(ff, pp):
                x_lines.extend([fff[0], ppp[0], None])
                y_lines.extend([fff[1], ppp[1], None])
                z_lines.extend([fff[2], ppp[2], None])
            frames.append(go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines'))
        plotdata.append(*frames)

    invisible_scale = go.Scatter3d(name="", visible=True, showlegend=False, opacity=0, hoverinfo='none',
                                   x=[-1.2, 1.2], y=[-1.2, 1.2], z=[-1.2, 1.2])
    plotdata.append(invisible_scale)
    # fig = go.Figure(data=[invisible_scale, plmesh, pp_marker, qq_marker, lines, *frames])
    fig = go.Figure(data=plotdata)
    # fig['layout']['scene']['aspectmode'] = "data"
    # fig['layout']['scene']['xaxis']['range'] = [-.5, .5]
    # fig['layout']['scene']['yaxis']['range'] = [-1., 1.]
    # fig['layout']['scene']['zaxis']['range'] = [-.5, .5]
    fig.show()

