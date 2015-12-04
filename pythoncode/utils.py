import numpy as np
mc = {'b' :(77, 117, 179),
      'r' :(210, 88, 88),
      'k' :(38,35,35),
      'white':(255,255,255),
     'grey':(197,198,199)}

for key in mc.keys():
    mc[key] = [x / 255.0 for x in mc[key]]
    print key, mc[key]

def plot_scalebars(ax, div=3.0, labels=True,
                    xunits="", yunits="", nox=False,
                    sb_xoff=0, sb_yoff=0, rotate_yslabel=False,
                    linestyle="-k", linewidth=4.0,
                    textcolor='k', textweight='normal', labelfontsize = 13):
    '''
    Stolen from C. Schmit Hieber (sp)
    Args:
        ax:
        div:
        labels:
        xunits:
        yunits:
        nox:
        sb_xoff:
        sb_yoff:
        rotate_yslabel:
        linestyle:
        linewidth:
        textcolor:
        textweight:
        labelfontsize:

    Returns:

    '''

    scale_dist_x = 0.02
    scale_dist_y = 0.02
    graph_width = 6.0
    graph_height = 4.0
    key_dist = 0.01
    # print dir(ax.dataLim)
    xmin = ax.dataLim.xmin
    xmax = ax.dataLim.xmax
    ymin = ax.dataLim.ymin
    ymax = ax.dataLim.ymax
    xscale = xmax-xmin
    yscale = ymax-ymin

    xoff = (scale_dist_x + sb_xoff) * xscale
    yoff = (scale_dist_y - sb_yoff) * yscale

    # plot scale bars:
    xlength = prettyNumber((xmax-xmin)/div)
    xend_x, xend_y = xmax, ymin
    if not nox:
        xstart_x, xstart_y = xmax-xlength, ymin
        scalebarsx = [xstart_x+xoff, xend_x+xoff]
        scalebarsy = [xstart_y-yoff, xend_y-yoff]
    else:
        scalebarsx=[xend_x+xoff,]
        scalebarsy=[xend_y-yoff]

    ylength = prettyNumber((ymax-ymin)/div)
    yend_x, yend_y = xmax, ymin+ylength
    scalebarsx.append(yend_x+xoff)
    scalebarsy.append(yend_y-yoff)

    ax.plot(scalebarsx, scalebarsy, linestyle, linewidth=linewidth, solid_joinstyle='miter')

    if labels:
        # if textcolor is not None:
        #     color = "\color{%s}" % textcolor
        # else:
        #     color = ""
        if not nox:
            # xlabel
            if xlength >=1:
                xlabel = r"%d$\,$%s" % (xlength, xunits)
            else:
                xlabel = r"%g$\,$%s" % (xlength, xunits)
            xlabel_x, xlabel_y = xmax-xlength/2.0, ymin
            xlabel_y -= key_dist*yscale
            ax.text(xlabel_x+xoff, xlabel_y-yoff, xlabel, ha='center', va='top',
                    weight=textweight, color=textcolor, fontsize = labelfontsize) #, [pyx.text.halign.center,pyx.text.valign.top])
        # ylabel
        if ylength >=1:
            ylabel = r"%d$\,$%s" % (ylength,yunits)
        else:
            ylabel = r"%g$\,$%s" % (ylength,yunits)
        if not rotate_yslabel:
            ylabel_x, ylabel_y = xmax, ymin + ylength/2.0
            ylabel_x += key_dist*xscale
            ax.text(ylabel_x+xoff, ylabel_y-yoff, ylabel, ha='left', va='center',
                    weight=textweight, color=textcolor,fontsize = labelfontsize)
        else:
            ylabel_x, ylabel_y = xmax, ymin + ylength/2.0
            ylabel_x += key_dist*xscale
            ax.text(ylabel_x+xoff, ylabel_y-yoff, ylabel, ha='center', va='top', rotation=90,
                    weight=textweight, color=textcolor, fontsize = labelfontsize)

def prettyNumber(f):
    fScaled = f
    if fScaled < 1:
        correct = 10.0
    else:
        correct = 1.0

    # set stepsize
    nZeros = int(np.log10(fScaled))
    prev10e = 10.0**nZeros / correct
    next10e = prev10e * 10

    if fScaled / prev10e  > 7.5:
        return next10e
    elif fScaled / prev10e  > 5.0:
        return 5 * prev10e
    else:
        return round(fScaled/prev10e) * prev10e