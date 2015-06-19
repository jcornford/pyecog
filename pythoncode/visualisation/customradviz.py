def radviz2(data, labels, variable_names, ax=None, labelsize = 12, **kwds):
    """
    RadViz - a multivariate data visualization algorithm
    This function returns the coordinates for radviz
    Parameters:
    -----------
    data : 
        data array
        rows are samples
        columns are dimensions
    labels:
        Column vector containing class names
    ax: Matplotlib axis object, optional
    color: list or tuple, optional
        Colors to use for the different classes
    colormap : str or matplotlib colormap object, default None
        Colormap to select colors from. If string, load colormap with that name
        from matplotlib.
    kwds: keywords
        Options to pass to matplotlib scatter plotting method
    Returns:
    --------
    ax: Matplotlib axis object
    """
    if ax is None:
        ax = plt.gca(xlim=[-1, 1], ylim=[-1, 1])

    def normalize(series):
        a = np.min(series, axis = 0)
        b = np.max(series, axis = 0)
        return np.divide((series - a),(b - a))

    m = data.shape[0]
    n = data.shape[1]
    data = normalize(data)


    to_plot = {}
    for kls in [1,2,3,4]:
        to_plot[kls] = [[], [],[]]

    s = np.array([(np.cos(t), np.sin(t))
                  for t in [2.0 * np.pi * (i / float(n))
                            for i in range(n)]])

    for i in range(m):
        label = labels[i,0]
        row = data[i,:]
        row_ = np.repeat(np.expand_dims(row, axis=1), 2, axis=1)
        y = (s * row_).sum(axis=0) / row.sum()
        
        kls = labels[i,0]
        to_plot[kls][0].append(y[0])
        to_plot[kls][1].append(y[1])
        to_plot[kls][2].append(label)
    
    #cols= x,y,color
    states =  np.hstack((np.array(to_plot[1]),
                         np.array(to_plot[2]),
                         np.array(to_plot[3]),
                         np.array(to_plot[4]),
                        ))
    states = states.T
    print states.shape
    ax.scatter(states[:,0],states[:,1],c = states[:,2])
    
    #colors = ['r','g','b','k']
    #for i, kls in enumerate([1,2,3,4]):
    #    ax.scatter(to_plot[kls][0], to_plot[kls][1],c = colors[i], **kwds)

    ax.add_patch(patches.Circle((0.0, 0.0), radius=1.0, facecolor='none'))

    for xy, name in zip(s, variable_names):
        ax.add_patch(patches.Circle(xy, radius=0.025, facecolor='black'))

        if xy[0] < 0.0 and xy[1] < 0.0:
            ax.text(xy[0] - 0.025, xy[1] - 0.025, name,
                    ha='right', va='top', size=labelsize)
        elif xy[0] < 0.0 and xy[1] >= 0.0:
            ax.text(xy[0] - 0.025, xy[1] + 0.025, name,
                    ha='right', va='bottom', size=labelsize)
        elif xy[0] >= 0.0 and xy[1] < 0.0:
            ax.text(xy[0] + 0.025, xy[1] - 0.025, name,
                    ha='left', va='top', size=labelsize)
        elif xy[0] >= 0.0 and xy[1] >= 0.0:
            ax.text(xy[0] + 0.025, xy[1] + 0.025, name,
                    ha='left', va='bottom', size=labelsize)

    ax.axis('equal')
    return ax

names = [u'kurtosis', u'skew', u'variation', u'coastline', u'Network States']
ax = radviz_points(dataobj.features,dataobj.label_colarray, names)