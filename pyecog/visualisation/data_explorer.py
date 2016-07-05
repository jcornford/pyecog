import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mpld3
import pandas as pd
from mpld3 import plugins, utils


class LinkedView(plugins.PluginBase):
    """A simple plugin showing how multiple axes can be linked"""

    JAVASCRIPT = """
    mpld3.register_plugin("linkedview", LinkedViewPlugin);
    LinkedViewPlugin.prototype = Object.create(mpld3.Plugin.prototype);
    LinkedViewPlugin.prototype.constructor = LinkedViewPlugin;
    LinkedViewPlugin.prototype.requiredProps = ["idpts", "idline", "data"];
    LinkedViewPlugin.prototype.defaultProps = {}
    function LinkedViewPlugin(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    LinkedViewPlugin.prototype.draw = function(){
      var pts = mpld3.get_element(this.props.idpts);
      var line = mpld3.get_element(this.props.idline);
      var data = this.props.data;

      function mouseover(d, i){
        line.data = data[i];
        line.elements().transition()
            .attr("d", line.datafunc(line.data))
            .style("stroke", this.style.fill);
      }
      pts.elements().on("mouseover", mouseover);
    };
    """

    def __init__(self, points, line, linedata):
        if isinstance(points, matplotlib.lines.Line2D):
            suffix = "pts"
        else:
            suffix = None

        self.dict_ = {"type": "linkedview",
                      "idpts": utils.get_id(points, suffix),
                      "idline": utils.get_id(line),
                      "data": linedata}
class ClickInfo(plugins.PluginBase):
    """Plugin for getting info on click"""
    
    JAVASCRIPT = """
    mpld3.register_plugin("clickinfo", ClickInfo);
    ClickInfo.prototype = Object.create(mpld3.Plugin.prototype);
    ClickInfo.prototype.constructor = ClickInfo;
    ClickInfo.prototype.requiredProps = ["id"];
    function ClickInfo(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };
    
    ClickInfo.prototype.draw = function(){
        var obj = mpld3.get_element(this.props.id);
        obj.elements().on("mousedown",
                          function(d, i){alert("clicked on points[" + i + "]");});
    }
    """
    def __init__(self, points):
        self.dict_ = {"type": "clickinfo",
                      "id": utils.get_id(points)}


# Basic script to run network state classfication
from load_seizure_data  import LoadSeizureData
from classifier_tester import ClassifierTester
from basicFeatures    import BasicFeatures
from randomForestClassifier import RandomForest
from freqfeatures     import FreqFeatures

dirpath = '/Users/Jonathan/Documents/PhD /Seizure_related/Network_states/VMData/Classified'
dataobj = LoadSeizureData(dirpath)
dataobj.load_data()
basicStatsExtractor = BasicFeatures()
dataobj.extract_feature_array([basicStatsExtractor])

from sklearn.preprocessing import normalize
norm_array = normalize(dataobj.data_array, axis = 1)
x2 = norm_array[:,:].reshape((301,1,5120))

fig = plt.figure(figsize = (8,8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

# scatter periods and amplitudes
np.random.seed(0)
P = 0.2 + np.random.random(size=301)
A = np.random.random(size=301)
x = np.linspace(0, 10, 5120/2.)

data = np.array([[x, Ai * np.sin(x / Pi)] for (Ai, Pi) in zip(A, P)])
data[:,1,:] = x2[:,0,::2]*20
print data.shape
A = np.random.randn(301)
A = dataobj.features[:,3]
P = np.arange(301)

points = ax2.scatter(P, A, s =50, alpha=0.5, c = dataobj.label_colarray)
#points = radviz2(df,'Network States',dataobj, ax = ax, color = cmap, edgecolor = 'white')
ax2.set_xlabel('Feature 2')
ax2.set_ylabel('Feature 1')

# create the line object
lines = ax1.plot(x, 0 * x, '-w', lw=1, alpha=0.8)
ax1.set_ylim(-1, 1)

ax1.set_title("Hover over points to see lines")

# transpose line data and add plugin
linedata = data.transpose(0, 2, 1).tolist()


labels = ["Trace {0}".format(i) for i in range(301)]
tooltip = plugins.PointLabelTooltip(points, labels)

plugins.connect(fig,LinkedView(points, lines[0], linedata))
plugins.connect(fig, ClickInfo(points))
#plugins.connect(fig,tooltip)
mpld3.display()