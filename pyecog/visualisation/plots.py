import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
# These are rubbish!
class scatter_matrix():
    def __init__(self, dataobj):
        import seaborn as sns
        import pandas as pd
        sns.set(style="white")
        sns.set_context("talk", font_scale=1.00)
        df = pd.DataFrame(dataobj.features, columns = dataobj.feature_names)
        df['Network States'] = dataobj.label_colarray
        self.pg = sns.pairplot(df, vars=dataobj.feature_names, size = 1.8, hue="Network States")
        sns.plt.show()



class radviz():
    'This needs work, currently written as a function. Just exporting from the ipython notbeook'
    def __init__(self,dataobj):
        # This should have all been done with a scatter!
        plot_dict = self._radviz_points(dataobj.features,dataobj.label_colarray)

        fig = plt.figure(figsize = (10,10))
        ax1 = plt.subplot(111)
        marker_size = 8
        marker_edge = 0.6
        alpha_val = 0.5
        ax1.plot(plot_dict[4][0],plot_dict[4][1],marker = 'o', linestyle = '', markersize = marker_size,label = 'Baseline',
                 color = '#a8a495' ,markeredgecolor = 'white',markeredgewidth=marker_edge,alpha = alpha_val)
        ax1.plot(plot_dict[1][0],plot_dict[1][1],marker = 'o', linestyle = '', markersize = marker_size, label = 'State 1',
                 color = '#39ad48',markeredgecolor = 'white',markeredgewidth=marker_edge, alpha = alpha_val)

        ax1.plot(plot_dict[2][0],plot_dict[2][1],marker = 'o', linestyle = '',markersize = marker_size,label = 'State 2',
                 color = '#d9544d',markeredgecolor = 'white',markeredgewidth=marker_edge,alpha = alpha_val)

        ax1.plot(plot_dict[3][0],plot_dict[3][1],marker = 'o', linestyle = '',markersize = marker_size,label = 'State 3',
                 color = '#3b5b92',markeredgecolor = 'white',markeredgewidth=marker_edge,alpha = alpha_val)


        ax1.legend(frameon = False,numpoints = 1)

        #dataobj.feature_names = ['kurtosis', 'skew', 'variation', 'coastline']
        names = dataobj.feature_names
        self._radviz_circle(ax1, names)
        ax1.axis('equal')
        ax1.axis('off')
        fig.show()

    def _normalize(self,series):
            a = np.min(series, axis = 0)
            b = np.max(series, axis = 0)
            return np.divide((series - a),(b - a))

    def _radviz_points(self,data, labels):
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
        Returns:
        --------
        plotting dictionary
        """
        

        m = data.shape[0]
        n = data.shape[1]
        data = self._normalize(data)


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
        return to_plot
        
    def _radviz_circle(self,ax, variable_names, labelsize = 12):
        n = len(variable_names)
        if ax is None:
            ax = plt.gca(xlim=[-1, 1], ylim=[-1, 1])
        
        
        s = np.array([(np.cos(t), np.sin(t))
                      for t in [2.0 * np.pi * (i / float(n))
                                for i in range(n)]])

        ax.add_patch(patches.Circle((0.0, 0.0), radius=1.0, facecolor='none'))

        for xy, name in zip(s, variable_names):
            ax.add_patch(patches.Circle(xy, radius=0.025, facecolor='grey'))

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

        
        return ax


        